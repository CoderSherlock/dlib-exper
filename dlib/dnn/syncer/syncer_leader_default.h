#ifndef DLIB_DNn_SYNCER_LEADER_DEFAULT_H_
#define DLIB_DNn_SYNCER_LEADER_DEFAULT_H_

#include "syncer.h"

using std::chrono::system_clock;

namespace dlib {

/*
 *	Leader Manage functions, by PZH
 *
 *  set_isMaster(int)
 */

template <typename trainer_type>
void dnn_syncer<trainer_type>::set_isMaster (int ism) {
	this->ismaster = ism;
}

template <typename trainer_type>
void dnn_syncer<trainer_type>::set_this_device (device me_) {
	this->me = me_;
}

/*
 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
 *	void(*)
 */
template<typename trainer_type>
void dnn_syncer<trainer_type>::print_slaves_status() {
	DLIB_CASSERT (ismaster == 1, "Slave deivce doesn't have the right to get slaves' status.");

	for (int i = 0; i < this->slaves_list.size(); i++ ) {
		std::cout << "[" << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << "]\t";
		std::cout << this->slaves_conns[i] << "\t" << this->slaves_status[i] << std::endl;
	}
}


template<typename trainer_type>
int dnn_syncer<trainer_type>::get_running_slaves_num() {
	DLIB_CASSERT (this->ismaster == 1, "Slave deivce doesn't have the right to get running_slaves_num.");
	int ret = 0;

	for (int i = 0; i < slaves_list.size(); i++) {
		if (this->slaves_status[i] == 1)
			ret ++;
	}

	return ret;
}

/*==================================================================================
 *	Leader Manage functions, by PZH
 *
 *  add_slave
 *  remove_slave
 *  init_slaves
 *  send_parameters: send parameters to specified slave(one)
 *  send_parameters_to_slaves_serialism: send parameters to all slaves in serial
 *  send_parameters_to_slaves_paralized: send parameters to all slaves in parali
 *  sn_sync: sync procedure/ call every round
 *  recieve_gradients_from_one: recieve gradients from specific slave
 ===================================================================================*/
template <typename trainer_type>
void dnn_leader<trainer_type>::add_slave (device slave) {
	DLIB_CASSERT (this->ismaster == 1, "Slave deivce doesn't have the right to add slaves.");

	this->slaves_list.push_back (slave);
	connection *c;
	this->slaves_conns.push_back (c);
	this->slaves_status.push_back (slaveStatus::Initlize);
}

template <typename trainer_type>
void dnn_leader<trainer_type>::remove_slave (size_t index) {
	DLIB_CASSERT (this->ismaster == 1, "Slave deivce doesn't have the right to remove slaves.");

	if (index < 0 || index >= this->slaves_list.size()) {
		std::cerr << "Removing an invalid index" << std::endl;
	}

	this->slaves_list.erase (this->slaves_list.begin() + index);
	this->slaves_conns.erase (this->slaves_conns.begin() + index);
	this->slaves_status.erase (this->slaves_status.begin() + index);
} // End of syncer::remove_slave

template <typename trainer_type>
void dnn_leader<trainer_type>::init_slaves() {
	DLIB_CASSERT (this->ismaster == 1, "Slave deivce doesn't have the right to init slaves.");

	for (int i = 0; i < this->slaves_list.size(); i++ ) {
		if (create_connection (this->slaves_conns[i], (unsigned short)this->slaves_list[i].port, this->slaves_list[i].ip, (unsigned short)this->me.port + i, this->me.ip)) {
			std::cerr << "Create failed on " << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << std::endl;
			this->slaves_status[i] = slaveStatus::NotConn;
			continue;
		}

		connection *conn = this->slaves_conns[i];

		// Greeting messages
		char init_msg[30];
		snprintf (init_msg, sizeof (init_msg), "%s:%d\n", &this->me.ip[0], this->me.port);
		conn->write (init_msg, 30);
		char reply_msg[30];
		conn->read (reply_msg, 30);

		// Validating if slave ip and port is correct
		char *cptr = strchr (reply_msg, ':');
		char *eptr = strchr (reply_msg, '\n');

		if (! (std::string (reply_msg, cptr - reply_msg) == this->slaves_list[i].ip &&
				atoi (std::string (cptr + 1, eptr - cptr + 1).c_str()) == this->slaves_list[i].port)) {
			std::cerr << "Error in validating slaves" << std::endl;
			this->slaves_status[i] = slaveStatus::FailVal;
			continue;
		}

		this->slaves_status[i] = slaveStatus::Running;
	}
}


// TODO:: slave is dependent on index
template<typename trainer_type>
void dnn_leader<trainer_type>::send_parameters (connection *slave, std::vector<tensor *> parameters) {
	for (size_t i = 0; i < parameters.size(); i++) {
		if (parameters[i]->size() != 0) {
			// this->print_tensor(parameters[i], 10);
			network::send_compressed_tensor (slave, parameters[i]);
		}
	}

}

/******************************************************
 *	Serialized send all tensors to all alive slaves
 *
 ******************************************************/
template<typename trainer_type>
void dnn_leader<trainer_type>::send_parameters_to_slaves_serialised (std::vector<tensor *> parameters) {
	if (this->get_running_slaves_num() != 0) {

		for (int s_c = 0, s_c_max = this->slaves_status.size(); s_c < s_c_max ; s_c ++) {
			if (this->slaves_status[s_c] == slaveStatus::Running) {
				send_parameters (this->slaves_conns[s_c], parameters);
			}
		}
	}
}


/******************************************************
 *	Paralized send all tensors to all alive slaves
 *
 ******************************************************/
template<typename trainer_type>
void dnn_leader<trainer_type>::send_parameters_to_slaves_paralized (std::vector<tensor *> parameters) {
	std::vector<std::thread *> recievers;
	recievers.resize (this->slaves_list.size());

	for (size_t i = 0; i < recievers.size(); i++) {
		if (this->slaves_status[i] == slaveStatus::Running)
			recievers[i] = new std::thread (&dnn_leader::send_parameters, this, this->slaves_conns[i], parameters);
	}

	for (size_t i = 0; i < recievers.size(); i++) {
		recievers[i]->join();
	}
}

template<typename trainer_type>
void dnn_leader<trainer_type>::sn_sync() {

	std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager> (this->trainer->num_computational_layers);

	std::vector<std::vector<resizable_tensor>> all_tensors;

	////////////////////////////////////////////////////////////
	auto epoch_time = system_clock::now();  // HPZ: Counting
	////////////////////////////////////////////////////////////

	// recieve_gradients_serialism(all_tensors);
	this->recieve_gradients_parallism (all_tensors);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (this->exper)																															//
		std::cout << "(Time for recieve_tensor) is "																							//
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;   				 	//

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (this->num_debug) {
		for (size_t i = 0; i < all_tensors.size(); i++) {
			std::cout << "Gradient from slave " << i << std::endl;

			for (size_t j = 0; j < all_tensors[i].size(); j++) {
				if (all_tensors[i][j].size() != 0) {
					//	std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
					print_tensor (&all_tensors[i][j], 10);
				}
			}
		}
	}

	////////////////////////////////////////////////////
	epoch_time = system_clock::now();  // HPZ: Counting
	////////////////////////////////////////////////////

	this->average (all_tensors);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (this->exper)																																//
		std::cout << "(Time for average) is "																										//
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;  // HPZ: Counting   	//

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	if (this->num_debug) {
		std::cout << "Averaged gradient: " << std::endl;

		for (size_t j = 0; j < all_tensors[0].size(); j++) {
			if (all_tensors[0][j].size() != 0) {
				//									std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
				print_tensor (&all_tensors[0][j], 10);
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	epoch_time = system_clock::now();  // HPZ: Counting																							  //
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::vector<tensor *> temp (this->trainer->num_computational_layers);

	for (size_t i = 0; i < temp.size(); i++) {
		// TODO : Deal with 0
		temp[i] = &all_tensors[0][i];
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (this->exper)																															//
		std::cout << "(Time for update) is "																								  	//
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;   					//

	epoch_time = system_clock::now();  // HPZ: Counting																						  	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	send_parameters_to_slaves_paralized (temp);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (this->exper)																															 //
		std::cout << "(Time for syncback) is "																									 //
				  << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - epoch_time).count() << std::endl;   					 //

	epoch_time = system_clock::now();  // HPZ: Counting																							 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (this->num_debug) {
		for (size_t i = 0; i < all_tensors.size(); i++) {
			for (size_t j = 0; j < all_tensors[i].size(); j++) {
				//									 print_tensor(&all_tensors[i][j], 10);
			}
		}
	}


	std::cout << "Sync finished" << std::endl;
	//	sleep(1000);
}

template<typename trainer_type>
int dnn_leader<trainer_type>::recieve_gradients_from_one (int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors) {

	for (size_t i = 0; i < cli_tensors[slave_index].size(); i++) {
		if (cli_tensors[slave_index][i].size() != 0) {
			// this->recieve_tensor(this->slave_conns[slave_index], &cli_tensors[slave_index][i]);
			network::recieve_compressed_tensor (this->slaves_conns[slave_index], &cli_tensors[slave_index][i]);

			// print_tensor(&cli_tensors[slave_index][i], cli_tensors[slave_index][i].size());
		}
	}

	return 1;
}

template<typename trainer_type>
void dnn_leader<trainer_type>::init_before_recieving (std::vector<std::vector<resizable_tensor>> &all_tensors) {
	// Get the pointer of gradients from current device
	std::vector<tensor *> tensors;
	tensors.resize (this->trainer->num_computational_layers);
	visit_layer_parameters (this->trainer->devices[0]->net, [&] (size_t i, tensor & t) {
		tensors[i] = &t;
	});

	// Initialize temporary gradients contrainer from all other devices
	all_tensors.resize (this->slaves_status.size());

	for (size_t i = 0; i < all_tensors.size(); i++) {
		all_tensors[i].resize (this->trainer->num_computational_layers);
		std::cout << "layers:" << this->trainer->num_computational_layers << std::endl;

		for (size_t j = 0; j < all_tensors[i].size(); j++) {
			if (this->slaves_status[i] == slaveStatus::Running) {
				std::cout << "layer size:" << tensors[j]->size() << std::endl;
				all_tensors[i][j].copy_size (*tensors[j]);
			}
		}
	}
}

template<typename trainer_type>
void dnn_leader<trainer_type>::recieve_gradients_serialism (std::vector<std::vector<resizable_tensor>> &all_tensors) {
	init_before_recieving (all_tensors);

	// Get gradients if there exists slave machine
	if (this->get_running_slaves_num() != 0) {
		for (int s_c = 0, s_c_max = this->slaves_status.size(); s_c < s_c_max ; s_c ++) {
			if (this->slaves_status[s_c] == slaveStatus::Running) {
				std::cout << "Reciveing from " << s_c << std::endl;
				recieve_gradients_from_one (s_c, all_tensors);
			}
		}
	}
}


template<typename trainer_type>
void dnn_leader<trainer_type>::recieve_gradients_parallism (std::vector<std::vector<resizable_tensor>> &all_tensors) {
	init_before_recieving (all_tensors);
	std::vector<std::thread *> recievers;
	recievers.resize (all_tensors.size());

	for (size_t i = 0; i < recievers.size(); i++) {
		if (this->slaves_status[i] == slaveStatus::Running)
			recievers[i] = new std::thread (&dnn_leader::recieve_gradients_from_one, this, i, std::ref (all_tensors));
	}

	for (size_t i = 0; i < recievers.size(); i++) {
		recievers[i]->join();
	}
}


} // End of Namespace dlib

#endif
