// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the introductory
    dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples.  In this
    example we are going to show how to create inception networks. 

    An inception network is composed of inception blocks of the form:

               input from SUBNET
              /        |        \
             /         |         \
          block1    block2  ... blockN 
             \         |         /
              \        |        /
          concatenate tensors from blocks
                       |
                    output
                 
    That is, an inception block runs a number of smaller networks (e.g. block1,
    block2) and then concatenates their results.  For further reading refer to:
    Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of
    the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
*/

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

#include "../dnn_dist_data.h"

using namespace std;
using namespace dlib;

std::vector<device> loadSlaves (char *filename) {
	std::ifstream f;
	f.open (filename);

	std::vector<device> ret;

	int number = 0;
	std::string ip;
	int port = -1;

	while (f >> ip >> port) {
		device temp (number, ip, port);
		ret.push_back (temp);
	}

	f.close();
	return ret;
}


// Inception layer has some different convolutions inside.  Here we define
// blocks as convolutions with different kernel size that we will use in
// inception layer block.
template <typename SUBNET> using block_a1 = relu<con<10,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_a2 = relu<con<10,3,3,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a3 = relu<con<10,5,5,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a4 = relu<con<10,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;

// Here is inception layer definition. It uses different blocks to process input
// and returns combined output.  Dlib includes a number of these inceptionN
// layer types which are themselves created using concat layers.  
template <typename SUBNET> using incept_a = inception4<block_a1,block_a2,block_a3,block_a4, SUBNET>;

// Network can have inception layers of different structure.  It will work
// properly so long as all the sub-blocks inside a particular inception block
// output tensors with the same number of rows and columns.
template <typename SUBNET> using block_b1 = relu<con<4,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_b2 = relu<con<4,3,3,1,1,SUBNET>>;
template <typename SUBNET> using block_b3 = relu<con<4,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;
template <typename SUBNET> using incept_b = inception3<block_b1,block_b2,block_b3,SUBNET>;

// Now we can define a simple network for classifying MNIST digits.  We will
// train and test this network in the code below.
using net_type = loss_multiclass_log<
        fc<10,
        relu<fc<32,
        max_pool<2,2,2,2,incept_b<
        max_pool<2,2,2,2,incept_a<
        input<matrix<unsigned char>>
        >>>>>>>>;

int main(int argc, char** argv) try
{
	if (argc < 2) {
		cout << "Master program has invalid argumnets" << endl;
		return 1;
	}

	char *data_path;					// Training & Testing data
	char *slave_path;					// File contains all slave ip and port information

	int ismaster = 0;
	device me;
	device master;

	std::vector<device> slave_list;

	me.ip = argv[1];
	me.port =atoi (argv[2]);
	me.number = atoi (argv[3]);

	for (int i = 1; i < argc; i++) {
		if (strcmp (argv[i], "-d") == 0) {
			data_path = argv[i + 1];
			std::cout << "Dataset:\t" << data_path << std::endl;
		}

		if (strcmp (argv[i], "-s") == 0) {
			slave_path = argv[i + 1];
			std::cout << "Slaveset:\t" << slave_path << std::endl;
		}
	}

	// Get slaves
	slave_list = loadSlaves (slave_path);

	// Print self information
	std::cout << "Local Machine info:\n";
	std::cout << "slave" << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

	// Get data
	dataset<matrix<unsigned char>, unsigned long> training (load_mnist_training_data, data_path);
	dataset<matrix<unsigned char>, unsigned long> testing (load_mnist_testing_data, data_path);
	training = training.split (0, 1000);
	dataset<matrix<unsigned char>, unsigned long> local_training = training = training.split_by_group (slave_list.size(), me.number);

    // Make an instance of our inception network.
    net_type net;
    cout << "The net has " << net.num_layers << " layers in it." << endl;
    cout << net << endl;


    cout << "Traning NN..." << endl;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.set_synchronization_file("inception_sync", std::chrono::seconds(20));
    // Train the network.  This might take a few minutes...
    // trainer.train(training_images, training_labels);
	
		char sync_filename[30];
	sprintf (sync_filename, "backup.%d.mm", me.number);
	trainer.set_synchronization_file (sync_filename, std::chrono::seconds (20));

	// HPZ: Setup synchronized protocol and test for the connection availablitiy.
	using trainer_type = dnn_trainer<net_type>;
	dnn_worker<trainer_type> syncer (&trainer, 0);
	syncer.set_this_device (me);

	// TODO: Wait for master connect
	if (!syncer.wait_for_master_init()) {
		std::cerr << "Error happens when master send init message" << std::endl;
		exit (0);
	}

	trainer.isDistributed = 1;

	// HPZ: Manually check if any problems happened in the init
	sleep ((unsigned int) 0);

	int epoch = 0, batch = 0;
	int mark = 0;
	auto time = 0;

	// sleep ((unsigned int) (me.number % 2) * 10);

	while (true) {
		while (trainer.synchronization_status != 4) {};

		mark += 1;

		auto epoch_time = system_clock::now();  // HPZ: Counting

		while (trainer.status_lock.trylock() == 0);

		if (trainer.synchronization_status != 3)
			std::cout << "Something wrong with sync lock: current: " << trainer.synchronization_status << "\t Going to set: 0" << std::endl;

		trainer.synchronization_status = 0;
		std::cout << "[dnn_master]: init done, may start to train" << std::endl;
		trainer.status_lock.unlock();

		epoch += trainer.train_one_batch (local_training.getData(), local_training.getLabel());

		// Wait for ready
		std::cout << "Im here" << std::endl;

		while (trainer.synchronization_status != 1) {
			asm ("");
		}//std::cout<<"wait to sync" << std::endl;}

		// std::cout << "[dnn_master]: start to sync" << std::endl;

		std::cout << "(train time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting
		// std::cout << "[Before]" << std::endl;
		// accuracy(net, local_training_images, local_training_labels);
		// accuracy(net, testing_images, testing_labels);

		auto sync_time = system_clock::now();  // HPZ: Counting
		syncer.sn_sync();
		std::cout << "(sync time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - sync_time).count() << std::endl;  // HPZ: Counting

		// serialize(trainer, std::cout);

		// Wait for all devices send back to their paramaters

		while (trainer.synchronization_status != 4) {} //std::cout <<"wait to update"<<std::endl;}

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is "
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count();

		std::cout << trainer.learning_rate << std::endl;
		// std::cout << "[After]" << std::endl;
		// local_training.accuracy (net);
		// accuracy(net, testing_images, testing_labels);
		//

		if (trainer.learning_rate <= 0.001) {
			std::cout << "---------------------------" << std::endl;
			std::cout << "|Exit because l_rate      |" << std::endl;
			std::cout << "---------------------------" << std::endl;
			break;
		}

		if (epoch >= 30) {
			std::cout << "---------------------------" << std::endl;
			std::cout << "|Exit because 30 epochs   |" << std::endl;
			std::cout << "---------------------------" << std::endl;
			break;
		}


	}

	// trainer.train(training_images, training_labels);

	// local_training.accuracy (net);
	// testing.accuracy (net);
	std::cout << "All time: " << time << std::endl;
	std::cout << trainer << std::endl;
	sleep ((unsigned int) 3600);

    net.clean();
    serialize("mnist_network_inception.dat") << net;
    // Now if we later wanted to recall the network from disk we can simply say:
    // deserialize("mnist_network_inception.dat") >> net;


    // Now let's run the training images through the network.  This statement runs all the
    // images through it and asks the loss layer to convert the network's raw output into
    // labels.  In our case, these labels are the numbers between 0 and 9.

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

