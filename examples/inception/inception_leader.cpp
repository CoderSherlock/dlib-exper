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

// Load Slave function copied from lenet dist leader
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
    // This example is going to run on the MNIST dataset.
	if (argc < 2) {
		cout << "Master program has invalid argumnets" << endl;
		return 1;
	}

	char *data_path;					// Training & Testing data
	char *slave_path;					// File contains all slave ip and port information

	int ismaster = 1;
	device me;
	device master;

	std::vector<device> slave_list;

	// Get the mode, ip and port
	me.ip = argv[1];
	me.port = atoi (argv[2]);
	me.number = atoi (argv[3]);

	// Print self information
	std::cout << "Local Machine info:\n";
	std::cout << "master" << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

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

	dataset<matrix<unsigned char>, unsigned long> training (load_mnist_training_data, data_path);
	dataset<matrix<unsigned char>, unsigned long> testing (load_mnist_testing_data, data_path);
	training = training.split (0, 1000);


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
	
    net.clean();



	using trainer_type = dnn_trainer<net_type>;

	dnn_leader<trainer_type> syncer (&trainer, 0);
	syncer.set_this_device (me);
	syncer.set_isMaster (1);

	for (int i = 0; i < slave_list.size(); i++) {
		syncer.add_slave (slave_list[i]);
	}

	trainer.isDistributed = 1;

	// HPZ: Manually check if any problems happened in the init
	sleep ((unsigned int) 0);

	trainer.synchronization_status = 0;
	trainer.train_one_batch (training.getData(), training.getLabel());

	while (trainer.synchronization_status != 1) { }

	trainer.synchronization_status = 2;

	while (trainer.synchronization_status != 4) { }

	syncer.init_slaves();

	std::cout << "Finished Initialization, now start training procedures" << std::endl;
	syncer.print_slaves_status();
	std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;

	auto time = 0;
	int epoch = 0, batch = 0;
	int mark = 0;
	int ending = ceil ((float)training.getData().size() / syncer.get_running_slaves_num() / 128) * 30;


	while (1) {
		mark += 1;
		trainer.train_noop();
		auto epoch_time = system_clock::now();  // HPZ: Counting
		// epoch += trainer.train_one_batch(local_training_images, local_training_labels);

		syncer.sn_sync();
		epoch ++;

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is "
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count();


		training.accuracy (net);


		if (ismaster) {
			if (trainer.learning_rate <= 0.001) {
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because l_rate      |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}

			if (epoch >= ending) {
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because 30 epochs   |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}
		}
	}

	std::cout << "All time: " << time << std::endl;

	training.accuracy (net);
	testing.accuracy (net);

	std::cout << trainer << std::endl;

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

