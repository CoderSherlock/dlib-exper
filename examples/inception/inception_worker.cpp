/*
 *	This Network is an experiemental neural network for lenet (Distributed ver.)
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

	char *config_path; 						// File contains all slave ip and port information
	device me;								// device information include ip and port
	device master;							// parent device information
	std::vector<device> slave_list;			// children device information
	dt_config distributed_trainer_config;	// All-topology network structure information


	// Get the mode, ip and port
	me.ip = argv[1];
	me.port = atoi(argv[2]);
	me.number = atoi(argv[3]);
	me.sync_type = device_sync_type::async;	// Default as sync

	//////////////////////////////////////////////////////////////////////////////////////
	/* Print self information *///////////////////////////////////////////////////////////
	std::cout << "Local Machine info:\n";												//
	std::cout << " " << me.ip << ":" << me.port << " " << me.number << std::endl;		//
																						//
	for (int i = 1; i < argc; i++)														//
	{																					//
		if (strcmp(argv[i], "-c") == 0)													//
		{																				//
			config_path = argv[i + 1];													//
			std::cout << "Slaveset:\t" << config_path << std::endl;						//
		}																				//
	}																					//
	//////////////////////////////////////////////////////////////////////////////////////



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* Fetch distributed-learning-system's configuration *////////////////////////////////////////////////////////////////////////////////
	distributed_trainer_config.read_config(config_path);																				//
																																		//
	device_role role = distributed_trainer_config.get_role(me.ip, me.port);																		//
	me.number = distributed_trainer_config.get_number(me.ip, me.port);
	master = distributed_trainer_config.get_my_master(me);																	//
																																		//
	// Validating current node's role -> worker or leader or supleader?																	//
	std::cout << "I'm a " << (role == device_role::worker ? "worker" : (role == device_role::leader ? "leader" : (role == device_role::supleader ? "supleader" : "undecided"))) << std::endl;	//
																																		//
	// Get slaves																														//
	for (auto i = distributed_trainer_config.device_list.begin(); i != distributed_trainer_config.device_list.end(); ++i)				//
	{																																	//
		if (i->master == me.number)																										//
		{																																//
			i->comp_ability = get_comp_ability(i->number, distributed_trainer_config.device_list);										//
			slave_list.push_back(*i);																									//
		}																																//
	}																																	//
																																		//
	me.comp_ability = 0;																												//
	for (auto i = slave_list.begin(); i != slave_list.end(); ++i)																		//
	{																																	//
		me.comp_ability += i->comp_ability;																								//
	}																																	//
																																		//
	// Get training data loaded																											//
	char *training_data_path = strdup(distributed_trainer_config.training_dataset_path.begin()->c_str());								//
																																		//
	dataset<matrix<unsigned char>, unsigned long> training(load_mnist_training_data, training_data_path);								//
																																		//
	char *testing_data_path = strdup(distributed_trainer_config.testing_dataset_path.begin()->c_str());									//
	dataset<matrix<unsigned char>, unsigned long> testing(load_mnist_testing_data, testing_data_path);									//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	training = training.split(0, 6000);						// Since lenet used mnist dataset with amount of 60000, we cut it to 1/10 of
	std::cout << training.getData().size() << std::endl;	// the original for fast training, not necessary
	
	
	
	int all = 0, ben = 0;

	/*
	 * Define net_type (By CoderSherlock)
	 */
	net_type net;

	dnn_trainer<net_type> trainer(net);

	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(12);
	trainer.be_verbose();

	// HPZ: Setup synchronized protocol and test for the connection availability.
	using trainer_type = dnn_trainer<net_type>;

	/* Setup memory-disk synchronization file 				*
	 * Commented since this is not working on odroid linux	*/
	// char sync_filename[30];
	// sprintf(sync_filename, "backup.%s.mm", "pe_test");
	// trainer.set_synchronization_file(sync_filename, std::chrono::seconds(60));

	if (role == device_role::worker)
	{
		dnn_worker<trainer_type, matrix<unsigned char>, unsigned long> worker(&trainer);
		int finished_batch = 0;

		worker.set_this_device(me);
		worker.set_role(role);
		worker.set_master_device(master);

		worker.trainer->isDistributed = 1;
		worker.default_dataset = &training;
		worker.init_thread_pool();
		worker.init_trainer(training);

		while (true)
		{
			int do_task_status = worker.do_one_task_without_wait();
			if(do_task_status == -1)		// Recv an error task -> very likely to be a stop training signal 
				break;
			else if (do_task_status == 1){
				finished_batch += 1;
				continue;
			}
			
			
			if (trainer.learning_rate <= 0.001)
			{
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because l_rate      |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}

			if (finished_batch >= 500000)
			{
				std::cout << "------------------------------" << std::endl;
				std::cout << "|Exit because 500k batches   |" << std::endl;
				std::cout << "------------------------------" << std::endl;
				break;
			}
		}
	}
	else if (role == device_role::leader)
	{
		dnn_leader<trainer_type, matrix<unsigned char>, unsigned long> leader(&trainer, device_role(0));
		me.sync_type = device_sync_type::sync;
		leader.set_this_device(me);
		leader.set_role(role);
		leader.set_master_device(master);
		leader.exper = 1;

		for (int i = 0; i < slave_list.size(); i++)
		{
			leader.add_slave(slave_list[i]);
		}

		leader.trainer->isDistributed = 1;
		leader.default_dataset = &training;
		leader.init_trainer(training);
		leader.init_thread_pool();
		
		unsigned long record_epoch = 0;
		while(1) { 
			// if (leader.epoch != record_epoch) {
			// 	leader.trainer->read_lock.lock();
			// 	training.accuracy(net);
			// 	leader.trainer->read_lock.unlock();
			// 	record_epoch = leader.epoch;

			// 	if (record_epoch == 1) exit(0);
			// }
		}

	}
	else if (role == device_role::supleader)
	{
		dnn_full_leader<trainer_type, matrix<unsigned char>, unsigned long> leader(&trainer, device_role(0));
		me.sync_type = device_sync_type::async;
		leader.set_this_device(me);
		leader.set_role(role);
		leader.exper = 1;

		for (int i = 0; i < slave_list.size(); i++)
		{
			leader.add_slave(slave_list[i]);
		}

		trainer.isDistributed = 1;
		leader.default_dataset = &training;
		leader.init_trainer();
		leader.init_thread_pool();

		std::cout << "Finish initialization training, it takes " << 0 << " seconds" << std::endl;

		// leader.init_slaves();
		// leader.init_receiver_pool();

		std::cout << "Finished Initialization, now start training procedures" << std::endl;

		auto real_time = system_clock::now();
		auto print_time = 0;
		leader.ending_time = distributed_trainer_config.ending_epoch;
		std::cout << "Training is set to be finished after " << leader.ending_time << " epochs" << std::endl;

		dataset<matrix<unsigned char>, unsigned long> testing = training;

		unsigned long record_epoch = 0;
		while(1) { 
			if (leader.epoch != record_epoch) {
				leader.trainer->read_lock.lock();
				// training.accuracy(net);
				leader.trainer->read_lock.unlock();
				record_epoch = leader.epoch;

				if (record_epoch == distributed_trainer_config.ending_epoch) {
					leader.listener_status = 0;	
					break;
				}
			}
		}
		// leader.sync(&testing);
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
	exit(0);
}
catch (std::exception &e)
{
	cout << e.what() << endl;
}
