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

#define ASYNC 1

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

std::vector<device> loadSlaves(char *filename)
{
	std::ifstream f;
	f.open(filename);

	std::vector<device> ret;

	int number = 0;
	std::string ip;
	int port = -1;

	while (f >> ip >> port)
	{
		device temp(number, ip, port);
		ret.push_back(temp);
	}

	f.close();
	return ret;
}

int load_permission_data(char *dataset, std::vector<matrix<int>> &data, std::vector<unsigned long> &label)
{
	std::fstream in(dataset);
	std::string line;
	while (in >> line)
	{
		matrix<int> instance;
		size_t pos = 0;
		int index = 0;
		instance.set_size(1, 427);
		while ((pos = line.find(",")) != std::string::npos)
		{
			std::string token = line.substr(0, pos);
			instance(0, index++) = stoi(token);
			line.erase(0, pos + 1);
		}
		unsigned long instance_label = (unsigned long)stoi(line);
		data.push_back(instance);
		label.push_back(instance_label);
	}
	return 1;
}

int main(int argc, char **argv) try
{

	// signal (SIGINT, to_exit);

	if (argc < 2)
	{
		cout << "Master program has invalid argumnets" << endl;
		return 1;
	}

	char *training_data_path, *testing_data_path; // Training & Testing data
	char *slave_path;							  // File contains all slave ip and port information

	int ismaster = 1;
	device me;
	device master;

	std::vector<device> slave_list;

	// Get the mode, ip and port
	me.ip = argv[1];
	me.port = atoi(argv[2]);
	me.number = atoi(argv[3]);

	// Print self information
	std::cout << "Local Machine info:\n";
	std::cout << "master"
			  << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-d") == 0)
		{
			training_data_path = argv[i + 1];
			std::cout << "Training dataset:\t" << training_data_path << std::endl;
			testing_data_path = argv[i + 2];
			std::cout << "Testing dataset:\t" << testing_data_path << std::endl;
		}

		if (strcmp(argv[i], "-s") == 0)
		{
			slave_path = argv[i + 1];
			std::cout << "Slaveset:\t" << slave_path << std::endl;
		}
	}

	// Get slaves
	slave_list = loadSlaves(slave_path);

	// Get data
	dataset<matrix<int>, unsigned long> training(load_permission_data, training_data_path);
	dataset<matrix<int>, unsigned long> testing(load_permission_data, testing_data_path);

	std::cout << training.getData().size() << std::endl;
	std::cout << testing.getData().size() << std::endl;

	int all = 0, ben = 0;

	/*
	 * Define net_type (By CoderSherlock)
	 */

	using net_type = loss_multiclass_log<
		fc<73,
		   relu<fc<128,
				   relu<fc<128,
						   relu<fc<256,
								   relu<fc<256,
										   relu<fc<512,
												   relu<fc<512,
														   input<matrix<int>>>>>>>>>>>>>>>>;

	net_type net;

	dnn_trainer<net_type> trainer(net);

	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(12);
	trainer.be_verbose();

	// HPZ: Setup synchronized protocol and test for the connection availability.
	using trainer_type = dnn_trainer<net_type>;

	char sync_filename[30];
	sprintf(sync_filename, "backup.%s.mm", "pe_test");
	trainer.set_synchronization_file(sync_filename, std::chrono::seconds(60));

#if !ASYNC
	dnn_leader<trainer_type> syncer(&trainer, 0);
#else
	dnn_async_leader<trainer_type> syncer(&trainer, 0);
#endif

	syncer.set_this_device(me);
	syncer.set_isMaster(1);
	syncer.exper = 1;

	for (int i = 0; i < slave_list.size(); i++)
	{
		syncer.add_slave(slave_list[i]);
	}

	trainer.isDistributed = 1;

	while (trainer.ready_status < 1)
	{
	};

	trainer.train_one_batch(training.getData(), training.getLabel());

	trainer.distributed_signal.get_mutex().lock();
	trainer.ready_status = 2;
	trainer.status_lock.unlock();
	trainer.distributed_signal.signal();

	while (trainer.ready_status < 4)
	{
	};

	std::cout << "Finish initialization training, it takes " << 0 << " seconds" << std::endl;

#if !ASYNC
	syncer.init_slaves();
#else
	syncer.init_slaves();
	syncer.init_receiver_pool();
#endif

	std::cout << "Finished Initialization, now start training procedures" << std::endl;
	syncer.print_slaves_status();
	std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;

#if !ASYNC
	int epoch = 0, batch = 0;
	int mark = 0;
	auto time = 0;
	int ending = ceil((float)training.getData().size() / syncer.get_running_slaves_num() / 128) * 5;

	while (true)
	{
		mark += 1;

		trainer.train_noop();

		auto epoch_time = system_clock::now(); // HPZ: Counting

		syncer.sn_sync();
		// trainer.train_one_epoch (training.getData(), training.getLabel());
		epoch++;

		// sleep((unsigned int) 0);

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count();

		// std::cout << "[After]" << std::endl;
		// training.accuracy (net);

		// if (trainer.learning_rate <= 0.00001) {
		// 	std::cout << "---------------------------" << std::endl;
		// 	std::cout << "|Exit because l_rate      |" << std::endl;
		// 	std::cout << "---------------------------" << std::endl;
		// 	break;
		// }

		if (epoch >= ending)
		{
			std::cout << "---------------------------" << std::endl;
			std::cout << "|Exit because 120 epochs   |" << std::endl;
			std::cout << "---------------------------" << std::endl;
			break;
		}
	}

	std::cout << "All time: " << time << std::endl;

	syncer.shut_slaves();

#else
	auto real_time = system_clock::now();
	auto print_time = 0;
	syncer.ending_time = ceil((float)training.getData().size() / syncer.get_running_slaves_num() / 128) * 12;
	std::cout << syncer.ending_time << std::endl;

	syncer.sync((unsigned long)training.getData().size());
	print_time = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - real_time).count();
	std::cout << "All time: " << print_time << std::endl;

#endif

	std::cout << training_data_path << std::endl;
	training.accuracy(net);
	std::cout << std::endl;
	std::cout << testing_data_path << std::endl;
	testing.accuracy(net);
	std::cout << std::endl;

	std::cout << trainer << std::endl;
	//sleep ((unsigned int) 3600);

	net.clean();
	serialize("permission_network.dat") << net;

	net_to_xml(net, "permission.xml");
}
catch (std::exception &e)
{
	cout << e.what() << endl;
}
