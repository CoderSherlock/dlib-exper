/*
 *	This Network is an experiemental neural network for permission analysis
 *
 *
 *
 *
 */



#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <csignal>
#include <fstream>

#include "dnn_dist_data.h"

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

int load_permission_training_data(char* dataset, std::vector<matrix<int>> &data, std::vector<unsigned long> &label) {	
	std::fstream in(dataset);
	std::string line;
	while(in >> line) {
		matrix<int> instance;
		size_t pos = 0;
		int index = 0;
		while((pos = line.find(",")) != std::string::npos) {
			std::string token = line.substr(0, pos);
			instance.set_size(1, 212);
			instance(0, index++) = stoi(token);
			line.erase(0, pos + 1);
		}
		unsigned long instance_label = (unsigned long) stoi(line);	
		data.push_back(instance);
		label.push_back(instance_label);
	}
	return 1;
}

int main (int argc, char **argv) try {

	// signal (SIGINT, to_exit);

	// This example is going to run on the MNIST dataset.
	if (argc < 2) {
		cout << "Master program has invalid argumnets" << endl;
		return 1;
	}
	
	char* data_path = argv[1];

	// Get data
	dataset<matrix<int>, unsigned long> training (load_permission_training_data, data_path);
	training = training.split (0, 1000);
	dataset<matrix<int>, unsigned long> local_training = training.split (0, 900);
	dataset<matrix<int>, unsigned long> local_testing = training.split (900, 1000);

	std::cout << local_training.getData().size() << std::endl;
	std::cout << local_testing.getData().size() << std::endl;

	int all = 0, ben = 0;
	std::vector<unsigned long> temp = local_testing.getLabel();
	for(auto i = temp.begin(); i != temp.end(); i++) {
		if(*i == 0) ben++;
		all++;	
	}
	std::cout << ben << "/" << all << std::endl;


	/*
	 * Define net_type (By CoderSherlock)
	 */
	using net_type = loss_multiclass_log <
					 fc<2,
					 relu<fc<16,
					 relu<fc<64,
					 relu<fc<64,
					 relu<fc<256,
					 relu<fc<256,
					 input<matrix<int>>
					 >>>>>>>>>>>>;

	net_type net;

	dnn_trainer<net_type> trainer (net);

	trainer.set_learning_rate (0.01);
	trainer.set_min_learning_rate (0.00001);
	trainer.set_mini_batch_size (128);
	trainer.be_verbose();

	char sync_filename[30];
	sprintf (sync_filename, "backup.%s.mm", "pe_test");
	trainer.set_synchronization_file (sync_filename, std::chrono::seconds (20));

	// HPZ: Setup synchronized protocol and test for the connection availablitiy.
	using trainer_type = dnn_trainer<net_type>;

	trainer.isDistributed = 0;

	int epoch = 0, batch = 0;
	int mark = 0;
	auto time = 0;

	while (true) {
		mark += 1;

		auto epoch_time = system_clock::now();  // HPZ: Counting

		trainer.train_one_epoch (local_training.getData(), local_training.getLabel());
		epoch ++;

		// sleep((unsigned int) 0);

		std::cout << "(train time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is "
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count();

		std::cout << trainer.learning_rate << std::endl;
		// std::cout << "[After]" << std::endl;
		local_training.accuracy (net);

		if (trainer.learning_rate <= 0.00001) {
			std::cout << "---------------------------" << std::endl;
			std::cout << "|Exit because l_rate      |" << std::endl;
			std::cout << "---------------------------" << std::endl;
			break;
		}

		// if (epoch >= 120) {
		//     std::cout << "---------------------------" << std::endl;
		//     std::cout << "|Exit because 120 epochs   |" << std::endl;
		//     std::cout << "---------------------------" << std::endl;
		//     break;
		// }


	}

	// trainer.train(training_images, training_labels);
	local_training.accuracy (net);
	local_testing.accuracy (net);
	std::cout << "All time: " << time << std::endl;
	std::cout << trainer << std::endl;
	// sleep ((unsigned int) 3600);

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

