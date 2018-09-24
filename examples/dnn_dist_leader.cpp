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

#include "dnn_dist_data.h"

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

std::vector<device> loadSlaves(char* filename) {
    std::ifstream f;
    f.open(filename);

	std::vector<device> ret;

    int number = 0;
    std::string ip;
    int port = -1;
    while(f >> ip >> port) {
		device temp(number, ip, port);
		ret.push_back(temp);
	}

	f.close();
	return ret;
}

int main(int argc, char** argv) try
{

	// signal (SIGINT, to_exit);

	// This example is going to run on the MNIST dataset.  
	if (argc < 2)
	{
		cout << "Master program has invalid argumnets" << endl;
		return 1;
	}

	char* data_path;					// Training & Testing data
	char* slave_path;					// File contains all slave ip and port information

	int ismaster = 1;
	device me;
	device master;
	int slave_number;

	std::vector<device> slave_list;

	// Get the mode, ip and port
	me.ip = argv[1];
	me.port = atoi(argv[2]);
	me.number = atoi(argv[3]);

	// Print self information
	std::cout << "Local Machine info:\n";
	std::cout << "master" << " " << me.ip << ":" << me.port << " " << me.number << std::endl;

	for(int i =1; i < argc; i++) {
		if(strcmp(argv[i], "-d")==0){
			data_path = argv[i+1];
			std::cout << "Dataset:\t" << data_path << std::endl;
		}

		if(strcmp(argv[i], "-s")==0){
			slave_path = argv[i+1];
			std::cout << "Slaveset:\t" << slave_path << std::endl;
		}
	}

	// Get slaves
	slave_list = loadSlaves(slave_path);

	// Get data
	dataset<matrix<unsigned char>, unsigned long> training(load_mnist_training_data, data_path);
	dataset<matrix<unsigned char>, unsigned long> testing(load_mnist_testing_data, data_path);
	training = training.split(0, 1000);


	/*
	 *  Define net_type (original by dlib)
	 */
	using net_type = loss_multiclass_log<
		fc<10,        
		relu<fc<84,   
		relu<fc<120,  
		max_pool<2,2,2,2,relu<con<16,5,5,1,1,
		max_pool<2,2,2,2,relu<con<6,5,5,1,1,
		input<matrix<unsigned char>> 
			>>>>>>>>>>>>;

	net_type net;

	dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(128);
	trainer.be_verbose(); 

	char sync_filename[30];
	sprintf(sync_filename, "backup.%d.mm", me.number);
	trainer.set_synchronization_file(sync_filename, std::chrono::seconds(20));

	/*
	 * HPZ: Setup synchronized protocol and test for the connection availablitiy.
	 */
	using trainer_type = dnn_trainer<net_type>;
	dnn_syncer<trainer_type> syncer(&trainer, 0);
	syncer.set_this_device(me);
	syncer.set_isMaster(1);
	for(int i=0; i < slave_list.size(); i++){
		syncer.add_slave(slave_list[i]);
	}


	std::cout << "Now we have " << syncer.get_running_slaves_num() << " slaves" << std::endl;
	syncer.print_slaves_status();

	trainer.isDistributed = 1;
	
	// HPZ: Manually check if any problems happened in the init
	sleep((unsigned int) 0);

	trainer.synchronization_status = 0;
	trainer.train_one_batch(training.getData(), training.getLabel());
	while(trainer.synchronization_status != 1) {asm("");}//std::cout<<"wait to sync" << std::endl;}
	trainer.synchronization_status = 2;
	while(trainer.synchronization_status != 3) {}

	syncer.init_slaves();

	// std::cout << syncer << std::endl;

	int epoch = 0, batch = 0;
	int mark = 0;
	/*
	 *	Record Overall time
	 */
	auto time = 0;
	while(1){
		mark += 1;
		auto epoch_time = system_clock::now();  // HPZ: Counting
		// trainer.train_one_epoch(local_training_images, local_training_labels);
		
		syncer.sn_sync();

		std::cout << "Finish batch " << batch++ << std::endl;
		std::cout << "Time for batch is " 
			<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting
		time += std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count();

		std::cout << trainer.learning_rate << std::endl;
		// std::cout << "[After]" << std::endl;
		// testing.accuracy(net);
        //
		
		if(ismaster)
		{
			if (trainer.learning_rate <= 0.001) {
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because l_rate      |" << std::endl;
				std::cout << "---------------------------" << std::endl;
                break;
            }

			if (epoch >= 60) {
				std::cout << "---------------------------" << std::endl;
				std::cout << "|Exit because 60 epochs   |" << std::endl;
				std::cout << "---------------------------" << std::endl;
				break;
			}
		}
	}

	training.accuracy(net);
	testing.accuracy(net);
	std::cout << "All time: " << time << std::endl;
	std::cout << trainer << std::endl;

	net.clean();
	serialize("mnist_network.dat") << net;
	net_to_xml(net, "lenet.xml");
}
catch(std::exception& e)
{
	cout << e.what() << endl;
}

