/*
 *	This Network is an experiemental neural network for permission analysis
 *  Author & mantainer: CoderSherlock
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

#define VL 4276

int load_function_data(char* dataset, std::vector<matrix<int>> &data, std::vector<unsigned long> &label) {	
	std::fstream in(dataset);
	std::string line;
	while(in >> line) {
		matrix<int> instance;
		size_t pos = 0;
		int index = 0;
		instance.set_size(1, VL);
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
	
	// Get data
	dataset<matrix<int>, unsigned long> training (load_function_data, argv[1]);
	dataset<matrix<int>, unsigned long> testing (load_function_data, argv[2]);
	dataset<matrix<int>, unsigned long> benign (load_function_data, argv[3]);
	dataset<matrix<int>, unsigned long> malware (load_function_data, argv[4]);


	std::cout << training.getData().size() << std::endl;
	std::cout << testing.getData().size() << std::endl;
	std::cout << benign.getData().size() << std::endl;
    std::cout << malware.getData().size() << std::endl;

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
					 relu<fc<4096,
					 relu<fc<4096,
					 input<matrix<int>>
					 >>>>>>>>>>>>>>>>>>;

	net_type net;

	dnn_trainer<net_type> trainer (net);

	trainer.set_learning_rate (0.01);
	trainer.set_min_learning_rate (0.00001);
	trainer.set_mini_batch_size (128);
	trainer.be_verbose();

	char sync_filename[30];
	sprintf (sync_filename, "backup.%s.mm", "func_test");
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

	std::cout << argv[1] << std::endl;
	training.accuracy (net);
	std::cout << std::endl;
	std::cout << argv[2] << std::endl;
	testing.accuracy (net);
	std::cout << std::endl;
	std::cout << argv[3] << std::endl;
	benign.accuracy (net);
	std::cout << std::endl;
	std::cout << argv[4] << std::endl;
	malware.accuracy (net);
	std::cout << std::endl;


	std::cout << "All time: " << time << std::endl;
	std::cout << trainer << std::endl;
	// sleep ((unsigned int) 3600);


	net.clean();
	serialize ("mnist_network.dat") << net;

	net_to_xml (net, "lenet.xml");
} catch (std::exception &e) {
	cout << e.what() << endl;
}
