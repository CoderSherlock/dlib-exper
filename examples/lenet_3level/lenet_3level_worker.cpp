/*
 *	This Network is an experiemental neural network for permission analysis (Distributed ver.)
 *	Writer: CoderSherlock
 */

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <csignal>

// #include "../dnn_dist_data.h"

#define ASYNC 1

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

int get_comp_ability(int number, std::list<device> device_list)
{
	int ret = 0;
	for (auto t = device_list.begin(); t != device_list.end(); ++t)
	{
		if (t->master == number)
		{
			ret += get_comp_ability(t->number, device_list);
		}
	}
	if (ret == 0)
		return 1;
	return ret;
}

int main(int argc, char **argv) try
{

	// signal (SIGINT, to_exit);

	if (argc < 2)
	{
		cout << "Master program has invalid arguments" << endl;
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
	me.number = distributed_trainer_config.get_number(me.ip, me.port);
	std::cout << "I'm a " << (role == 0 ? "worker" : (role == 1 ? "leader" : (role == 2 ? "supleader" : "undecided"))) << std::endl;
	for (auto i = distributed_trainer_config.device_list.begin(); i != distributed_trainer_config.device_list.end(); ++i)
	{
		if (i->master == me.number)
		{
			i->comp_ability = get_comp_ability(i->number, distributed_trainer_config.device_list);
			slave_list.push_back(*i);
		}
	}

	me.comp_ability = 0;
	for (auto i = slave_list.begin(); i != slave_list.end(); ++i)
	{
		me.comp_ability += i->comp_ability;
	}

	// Get data
	char *training_data_path = strdup(distributed_trainer_config.training_dataset_path.begin()->c_str());

	dataset<matrix<unsigned char>, unsigned long> training(load_mnist_training_data, training_data_path);

	char *testing_data_path = strdup(distributed_trainer_config.testing_dataset_path.begin()->c_str());
	dataset<matrix<unsigned char>, unsigned long> testing(load_mnist_testing_data, testing_data_path);

	training = training.split(0, 6000);
	std::cout << training.getData().size() << std::endl;

	int all = 0, ben = 0;

	/*
	 * Define net_type (By CoderSherlock)
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

	dnn_trainer<net_type> trainer(net);

	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(12);
	trainer.be_verbose();

	// HPZ: Setup synchronized protocol and test for the connection availability.
	using trainer_type = dnn_trainer<net_type>;

	// char sync_filename[30];
	// sprintf(sync_filename, "backup.%s.mm", "pe_test");
	// trainer.set_synchronization_file(sync_filename, std::chrono::seconds(60));

	if (role == device_role::worker)
	{
		dnn_worker<trainer_type> syncer(&trainer);

		syncer.set_this_device(me);
		syncer.set_role(role);

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
		std::cout << "Wait for leader to init" << std::endl;
		if (!syncer.wait_for_master_init())
		{
			std::cerr << "Error happens when master send init message" << std::endl;
			exit(0);
		}

		// sleep((unsigned int)3600);

		dataset<matrix<unsigned char>, unsigned long> local_training;

		int epoch = 0, batch = 0;
		int mark = 0;
		auto time = 0;

		while (true)
		{

			mark += 1;

			task_op operation = syncer.wait_for_task();

			auto batch_time = system_clock::now(); // *_*
			auto breakdown = system_clock::now();  // *_*

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

				std::cout << "(prepare " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*
				breakdown = system_clock::now();

				trainer.train_one_batch(local_training.getData(), local_training.getLabel());
				syncer.pre_train(operation);

				trainer.distributed_signal.get_mutex().lock();
				if (trainer.ready_status != 3)
					trainer.distributed_signal.wait();

				trainer.status_lock.unlock();

				std::cout << "(train+recv " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*
				// if(me.number % 2 == 1) sleep((unsigned int)10);

				syncer.notify_train_finish();
				syncer.wait_to_send();
				breakdown = system_clock::now();
				syncer.send_parameters_to_master();

				std::cout << "(send " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*

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

			std::cout << "Time for batch is " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - batch_time).count() << std::endl; // *_*

			if (trainer.learning_rate <= 0.001)
			{
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because l_rate      |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}

			if (epoch >= 5000)
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
		dnn_leader<trainer_type> syncer(&trainer, 0);
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

		sleep((unsigned int)5);

		syncer.init_slaves();

		std::cout << "Finished Initialization, now start training procedures" << std::endl;
		syncer.print_slaves_status();
		std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;

		syncer.wait_for_master_init();

		std::cout << "Connected by master" << std::endl;

		// sleep((unsigned int)3600);

		int epoch = 0, batch = 0;

		while (true)
		{
			task_op operation = syncer.wait_for_task();

			switch (operation.opcode)
			{
			case task_type::train_one_batch:
			{
				unsigned long start = *(unsigned long *)&operation.operand1, end = *(unsigned long *)&operation.operand2;
				std::cout << start << "~" << end << std::endl;
				std::cout << "diff:" << end - start << std::endl;

				// HPZ: Sync lateset parameters
				std::vector<resizable_tensor> latest_parameters;
				syncer.receive_latest_parameters(latest_parameters);

				std::vector<tensor *> temp(syncer.trainer->num_computational_layers);
				for (size_t i = 0; i < temp.size(); i++)
				{
					// TODO : Deal with 0
					temp[i] = &latest_parameters[i];
				}
				syncer.update(temp);

				// TODO: Dispatch jobs
				syncer.subdispatch(start, end);

				syncer.sync();

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

			if (epoch >= 5000)
			{
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because 30 epochs   |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}
		}
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

		sleep((unsigned int)10);

		syncer.init_slaves();
		syncer.init_receiver_pool();

		std::cout << "Finished Initialization, now start training procedures" << std::endl;
		syncer.print_slaves_status();
		std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;

		// sleep((unsigned int)3600);

		auto real_time = system_clock::now();
		auto print_time = 0;
		syncer.ending_time = distributed_trainer_config.ending_epoch;
		std::cout << syncer.ending_time << std::endl;

		dataset<matrix<unsigned char>, unsigned long> testing = training;

		syncer.sync((unsigned long)training.getData().size(), &testing);
		print_time = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - real_time).count();
		std::cout << "All time: " << print_time << std::endl;
	}
	else
	{
	}

	// Validating training dataset
	std::cout << *(distributed_trainer_config.training_dataset_path.begin()) << std::endl;
	training.accuracy(net);
	std::cout << std::endl;

	// Validating testing dataset
	std::cout << *(distributed_trainer_config.testing_dataset_path.begin()) << std::endl;
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
