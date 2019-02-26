// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
   This is an example illustrating the use of the deep learning tools from the
   dlib C++ Library.  In it, we will train the venerable LeNet convolutional
   neural network to recognize hand written digits.  The network will take as
   input a small image and classify it as one of the 10 numeric digits between
   0 and 9.

   The specific network we will run is from the paper
   LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
   Proceedings of the IEEE 86.11 (1998): 2278-2324.
   except that we replace the sigmoid non-linearities with rectified linear units.

   These tools will use CUDA and cuDNN to drastically accelerate network
   training and testing.  CMake should automatically find them if they are
   installed and configure things appropriately.  If not, the program will
   still run but will be much slower to execute.
   */


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <csignal>

#include "../dnn_dist_data.h"

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

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

int main (int argc, char **argv) try {

	// signal (SIGINT, to_exit);

	// This example is going to run on the MNIST dataset.
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


	std::cout << local_training.getData().size() << std::endl;


	/*
	 * Define net_type (original by dlib)
	 */
	using net_type = loss_multiclass_log <
					 fc<10,
					 relu<fc<84,
					 relu<fc<120,
					 max_pool<2, 2, 2, 2, relu<con<16, 5, 5, 1, 1,
					 max_pool<2, 2, 2, 2, relu<con<6, 5, 5, 1, 1,
					 input<matrix<unsigned char>>
					 >>>>>>>>>> >>;

	net_type net;

	dnn_trainer<net_type> trainer (net);

	trainer.set_learning_rate (0.01);
	trainer.set_min_learning_rate (0.00001);
	trainer.set_mini_batch_size (128);
	trainer.be_verbose();

	// char sync_filename[30];
	// sprintf (sync_filename, "backup.%d.mm", me.number);
	// trainer.set_synchronization_file (sync_filename, std::chrono::seconds (20));

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

	// At this point our net object should have learned how to classify MNIST images.  But
	// before we try it out let's save it to disk.  Note that, since the trainer has been
	// running images through the network, net will have a bunch of state in it related to
	// the last batch of images it processed (e.g. outputs from each layer).  Since we
	// don't care about saving that kind of stuff to disk we can tell the network to forget
	// about that kind of transient data so that our file will be smaller.  We do this by
	// "cleaning" the network before saving it.
	net.clean();
	serialize ("mnist_network.dat") << net;
	// Now if we later wanted to recall the network from disk we can simply say:
	// deserialize("mnist_network.dat") >> net;


	// Now let's run the training images through the network.  This statement runs all the
	// images through it and asks the loss layer to convert the network's raw output into
	// labels.  In our case, these labels are the numbers between 0 and 9.





	// Let's also see if the network can correctly classify the testing images.  Since
	// MNIST is an easy dataset, we should see at least 99% accuracy.
	/*
	   predicted_labels = net(testing_images);
	   num_right = 0;
	   num_wrong = 0;
	   for (size_t i = 0; i < testing_images.size(); ++i)
	   {
	   if (predicted_labels[i] == testing_labels[i])
	   ++num_right;
	   else
	   ++num_wrong;

	   }
	   cout << "testing num_right: " << num_right << endl;
	   cout << "testing num_wrong: " << num_wrong << endl;
	   cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
	   */

	// Finally, you can also save network parameters to XML files if you want to do
	// something with the network in another tool.  For example, you could use dlib's
	// tools/convert_dlib_nets_to_caffe to convert the network to a caffe model.
	net_to_xml (net, "lenet.xml");
} catch (std::exception &e) {
	cout << e.what() << endl;
}

