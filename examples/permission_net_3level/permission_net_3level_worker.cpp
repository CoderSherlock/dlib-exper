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

#define ASYNC 1

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

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

	char *config_path; // File contains all slave ip and port information

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
	std::cout << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-c") == 0)
		{
			config_path = argv[i + 1];
			std::cout << "Slaveset:\t" << config_path << std::endl;
		}
	}

	// Get slaves
	dt_config distributed_trainer_config;
	distributed_trainer_config.read_config(config_path);

	int role = distributed_trainer_config.get_role(me.ip, me.port);
	std::cout << "I'm a " << (role == 0 ? "worker" : (role == 1 ? "leader" : (role == 2 ? "supleader" : "undecided"))) << std::endl;

	// Get data
	char *training_data_path = strdup(distributed_trainer_config.training_dataset_path.c_str());

	dataset<matrix<int>, unsigned long> training(load_permission_data, training_data_path);
	std::list<dataset<matrix<int>, unsigned long>> testings;

	for (auto testing_data_itr = distributed_trainer_config.testing_dataset_path.begin(); testing_data_itr != distributed_trainer_config.testing_dataset_path.end(); ++testing_data_itr)
	{
		char *testing_data_path = strdup(testing_data_itr->c_str());
		dataset<matrix<int>, unsigned long> testing(load_permission_data, testing_data_path);
		testings.push_back(testing);
	}

	std::cout << training.getData().size() << std::endl;

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

	if (role == device_role::worker)
	{
		dnn_worker<trainer_type> syncer(&trainer);

		syncer.set_this_device(me);

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

		// TODO: Wait for master connect
		if (!syncer.wait_for_master_init())
		{
			std::cerr << "Error happens when master send init message" << std::endl;
			exit(0);
		}

		dataset<matrix<int>, unsigned long> local_training;

		int epoch = 0, batch = 0;
		int mark = 0;
		auto time = 0;

		while (true)
		{
			mark += 1;

			task_op operation = syncer.wait_for_task();

			switch (operation.opcode)
			{
			case task_type::train_one_batch:
			{
				unsigned long start = *(unsigned long *)&operation.operand1, end = *(unsigned long *)&operation.operand2;
				std::cout << start << "~" << end << std::endl;
				std::cout << "diff:" << end - start << std::endl;
				local_training = training.split(start, end);
				trainer.epoch_pos = 0;
				trainer.set_mini_batch_size(end - start);
				std::cout << "mini_batch:" << trainer.get_mini_batch_size() << std::endl;
				std::cout << "data_size:" << local_training.getData().size() << std::endl;
				trainer.train_one_batch(local_training.getData(), local_training.getLabel());
				syncer.pre_train(operation);

				trainer.distributed_signal.get_mutex().lock();
				if (trainer.ready_status != 3)
					trainer.distributed_signal.wait();

				trainer.status_lock.unlock();

				syncer.send_gradients_to_master();

				std::cout << "Learning rate is " << trainer.learning_rate << std::endl;
				break;
			}
			default:
			{
				// HPZ: TODO
				std::cout << "Error op" << std::endl;
				epoch = 99999;
			}
			}

			if (trainer.learning_rate <= 0.001)
			{
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because l_rate      |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}

			if (epoch >= 12)
			{
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because 30 epochs   |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}
		}
	}
	else if (role == device_role::leader)
	{
		dnn_async_leader<trainer_type> syncer(&trainer, 0);
		syncer.set_this_device(me);
		syncer.set_role(role);
		syncer.exper = 1;

		for (int i = 0; i < slave_list.size(); i++)
		{
			syncer.add_slave(slave_list[i]);
		}

		trainer.isDistributed = 1;

		while (trainer.ready_status < 1)
		{
		};
		std::cout << 0 << std::endl;

		trainer.train_one_batch(training.getData(), training.getLabel());
		std::cout << 1 << std::endl;

		trainer.distributed_signal.get_mutex().lock();
		trainer.ready_status = 2;
		std::cout << 2 << std::endl;

		trainer.status_lock.unlock();
		trainer.distributed_signal.signal();
		std::cout << 3 << std::endl;

		while (trainer.ready_status < 4)
		{
		};

		std::cout << "Finish initialization training, it takes " << 0 << " seconds" << std::endl;

		syncer.init_slaves();
		syncer.init_receiver_pool();
	}
	else if (role == device_role::supleader)
	{
		dnn_async_leader<trainer_type> syncer(&trainer, 0);
		syncer.set_this_device(me);
		syncer.set_role(role);
		syncer.exper = 1;

		for (int i = 0; i < slave_list.size(); i++)
		{
			syncer.add_slave(slave_list[i]);
		}

		trainer.isDistributed = 1;

		while (trainer.ready_status < 1)
		{
		};
		std::cout << 0 << std::endl;

		trainer.train_one_batch(training.getData(), training.getLabel());
		std::cout << 1 << std::endl;

		trainer.distributed_signal.get_mutex().lock();
		trainer.ready_status = 2;
		std::cout << 2 << std::endl;

		trainer.status_lock.unlock();
		trainer.distributed_signal.signal();
		std::cout << 3 << std::endl;

		while (trainer.ready_status < 4)
		{
		};

		std::cout << "Finish initialization training, it takes " << 0 << " seconds" << std::endl;

		syncer.init_slaves();
		syncer.init_receiver_pool();

		std::cout << "Finished Initialization, now start training procedures" << std::endl;
		syncer.print_slaves_status();
		std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;

		auto real_time = system_clock::now();
		auto print_time = 0;
		syncer.ending_time = 60;
		std::cout << syncer.ending_time << std::endl;

		syncer.sync((unsigned long)training.getData().size());
		print_time = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - real_time).count();
		std::cout << "All time: " << print_time << std::endl;
	}
	else
	{
	}

	// Validating training dataset
	std::cout << distributed_trainer_config.training_dataset_path << std::endl;
	training.accuracy(net);
	std::cout << std::endl;

	// Validating testing dataset
	auto name = distributed_trainer_config.testing_dataset_path.begin();
	auto data = testings.begin();
	for (; name != distributed_trainer_config.testing_dataset_path.end(); ++name, ++data)
	{
		std::cout << *name << std::endl;
		data->accuracy(net);
		std::cout << std::endl;
	}
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
