/*
 *	This Network is an experiemental neural network for permission analysis (Distributed ver.)
 *	Writer: CoderSherlock
 */

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <csignal>
#include <fstream>

#include "../dnn_dist_data.h"

#define ASYNC 0

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

int load_permission_data(char* dataset, std::vector<matrix<int>> &data, std::vector<unsigned long> &label) {	
	std::fstream in(dataset);
	std::string line;
	while(in >> line) {
		matrix<int> instance;
		size_t pos = 0;
		int index = 0;
		instance.set_size(1, 427);
		while((pos = line.find(",")) != std::string::npos) {
			std::string token = line.substr(0, pos);
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

	char *training_data_path, *testing_data_path;		// Training & Testing data
	char *slave_path;									// File contains all slave ip and port information

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
			training_data_path = argv[i + 1];
			std::cout << "Dataset:\t" << training_data_path << std::endl;
			testing_data_path = argv[i + 2];
			std::cout << "Dataset:\t" << testing_data_path << std::endl;
		}

		if (strcmp (argv[i], "-s") == 0) {
			slave_path = argv[i + 1];
			std::cout << "Slaveset:\t" << slave_path << std::endl;
		}
	}

	// Get slaves
	slave_list = loadSlaves (slave_path);
	
	// Get data
	dataset<matrix<int>, unsigned long> training (load_permission_data, training_data_path);
	dataset<matrix<int>, unsigned long> testing (load_permission_data, testing_data_path);

	std::cout << training.getData().size() << std::endl;
	std::cout << testing.getData().size() << std::endl;


	int all = 0, ben = 0;

	/*
	 * Define net_type (By CoderSherlock)
	 */

	using net_type = loss_multiclass_log <
					 fc<2,
					 relu<fc<32,
					 relu<fc<32,
					 relu<fc<128,
					 relu<fc<128,
					 relu<fc<512,
					 relu<fc<512,
					 input<matrix<int>>
					 >>>>>>>>>>>>>>;

	net_type net;

	dnn_trainer<net_type> trainer (net);

	trainer.set_learning_rate (0.01);
	trainer.set_min_learning_rate (0.00001);
	trainer.set_mini_batch_size (128);
	trainer.be_verbose();

	// HPZ: Setup synchronized protocol and test for the connection availablitiy.
	using trainer_type = dnn_trainer<net_type>;

#if !ASYNC
	dnn_leader<trainer_type> syncer(&trainer, 0);
#else
	dnn_async_leader<trainer_type> sync(&trainer, 0);
#endif

	syncer.set_this_device(me);
	syncer.set_isMaster(1);

	trainer.isDistributed = 1;

	int epoch = 0, batch = 0;
	int mark = 0;
	auto time = 0;

	while (true) {
		mark += 1;

		auto epoch_time = system_clock::now();  // HPZ: Counting

		trainer.train_one_epoch (training.getData(), training.getLabel());
		epoch ++;

		// sleep((unsigned int) 0);

		std::cout << "(train time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is "
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count();

		std::cout << trainer.learning_rate << std::endl;
		// std::cout << "[After]" << std::endl;
		training.accuracy (net);

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
	std::cout << argv[1] << std::endl;
	training.accuracy (net);
	std::cout << std::endl;
	std::cout << argv[2] << std::endl;
	testing.accuracy (net);
	std::cout << std::endl;
	// std::cout << argv[3] << std::endl;
	// benign.accuracy (net);
	// std::cout << std::endl;
	// std::cout << argv[4] << std::endl;
	// malware.accuracy (net);
	// std::cout << std::endl;
	// std::cout << argv[5] << std::endl;
	// b1.accuracy (net);
	// std::cout << std::endl;
	// std::cout << argv[6] << std::endl;
	// m1.accuracy (net);
	// std::cout << std::endl;
	// std::cout << argv[7] << std::endl;
	// b2.accuracy (net);
	// std::cout << std::endl;
	// std::cout << argv[8] << std::endl;
	// m2.accuracy (net);
	// std::cout << std::endl;

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

