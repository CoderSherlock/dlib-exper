/*
 *	This Network is an experiemental neural network for permission analysis (Distributed ver.)
 *	Writer: CoderSherlock
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
	std::cout << "slave" << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

	for (int i = 1; i < argc; i++) {
		if (strcmp (argv[i], "-d") == 0) {
			training_data_path = argv[i + 1];
			std::cout << "Training dataset:\t" << training_data_path << std::endl;
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

	dataset<matrix<int>, unsigned long> local_training = training.split_by_group( slave_list.size(), me.number);

	std::cout << training.getData().size() << std::endl;
	std::cout << local_training.getData().size() << std::endl;


	/*
	 * Define net_type (By CoderSherlock)
	 */

	using net_type = loss_multiclass_log <
					 fc<73,
					 relu<fc<128,
					 relu<fc<128,
					 relu<fc<256,
					 relu<fc<256,
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

	dnn_worker<trainer_type> syncer (&trainer, 0);

	syncer.set_this_device(me);
	
	// TODO: Wait for master connect
	if (!syncer.wait_for_master_init()) {
		std::cerr << "Error happens when master send init message" << std::endl;
		exit (0);
	}

	trainer.isDistributed = 1;


	int epoch = 0, batch = 0;
	int mark = 0;
	auto time = 0;

	while (true) {
		while(trainer.synchronization_status != 4) {};

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
	
	std::cout << "All time: " << time << std::endl;

	std::cout << argv[1] << std::endl;
	training.accuracy (net);
	std::cout << std::endl;

	std::cout << trainer << std::endl;
	sleep ((unsigned int) 3600);

	net.clean();
	serialize ("permission_network.dat") << net;

	net_to_xml (net, "permission.xml");

} catch (std::exception &e) {
	cout << e.what() << endl;
}

