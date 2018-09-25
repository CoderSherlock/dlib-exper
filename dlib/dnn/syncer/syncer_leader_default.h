#ifndef DLIB_DNn_SYNCER_DEFAULT_H_  
#define DLIB_DNn_SYNCER_DEFAULT_H_  

#include "syncer.h"

using std::chrono::system_clock;

/*
 *	Leader Manage functions, by PZH
 *
 *  set_isMaster(int)
 */

template <typename trainer_type>
void dlib::dnn_syncer<trainer_type>::set_isMaster(int ism) {
	this->ismaster = ism;
}

template <typename trainer_type>
void dlib::dnn_syncer<trainer_type>::set_this_device(device me_) {
	this->me = me_;
}

template <typename trainer_type>
void dlib::dnn_leader<trainer_type>::add_slave(device slave) {
	DLIB_CASSERT(this->ismaster == 1, "Slave deivce doesn't have the right to add slaves.");

	this->slaves_list.push_back(slave);
	connection* c;
	this->slaves_conns.push_back(c);
	this->slaves_status.push_back(slaveStatus::Initlize);
}

template <typename trainer_type>
void dlib::dnn_leader<trainer_type>::remove_slave(size_t index) {
	DLIB_CASSERT(this->ismaster == 1, "Slave deivce doesn't have the right to remove slaves.");

	if (index < 0 || index >= this->slaves_list.size()) {
		std::cerr << "Removing an invalid index" << std::endl;
	}
	this->slaves_list.erase(this->slaves_list.begin() + index);
	this->slaves_conns.erase(this->slaves_conns.begin() + index);
	this->slaves_status.erase(this->slaves_status.begin() + index);
} // End of syncer::remove_slave

template <typename trainer_type>
void dlib::dnn_leader<trainer_type>::init_slaves() {
	DLIB_CASSERT(this->ismaster == 1, "Slave deivce doesn't have the right to init slaves.");

	for(int i = 0; i < this->slaves_list.size(); i++ ) {
		if(create_connection(this->slaves_conns[i], (unsigned short)this->slaves_list[i].port, this->slaves_list[i].ip, (unsigned short)this->me.port + i, this->me.ip)) {
			std::cerr << "Create failed on " << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << std::endl;
			this->slaves_status[i] = slaveStatus::NotConn;
			continue;
		}

		connection* conn = this->slaves_conns[i];

		// Greeting messages
		char init_msg[30];
		snprintf(init_msg, sizeof(init_msg), "%s:%d\n", &this->me.ip[0], this->me.port);
		conn->write(init_msg, 30);
		char reply_msg[30];
		conn->read(reply_msg, 30);

		// Validating if slave ip and port is correct
		char* cptr = strchr(reply_msg, ':');
		char* eptr = strchr(reply_msg, '\n');
		if(!(std::string(reply_msg, cptr - reply_msg)==this->slaves_list[i].ip &&
					atoi(std::string(cptr + 1, eptr - cptr + 1).c_str()) == this->slaves_list[i].port)) {
			std::cerr << "Error in validating slaves" << std::endl;
			this->slaves_status[i] = slaveStatus::FailVal;
			continue;
		}

		this->slaves_status[i] = slaveStatus::Running;
	}
}

/*
 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
 *	void(*)
 */
template<typename trainer_type>
void dlib::dnn_syncer<trainer_type>::print_slaves_status(){
	DLIB_CASSERT(ismaster == 1, "Slave deivce doesn't have the right to get slaves' status.");

	for(int i = 0; i < this->slaves_list.size(); i++ ){
		std::cout << "[" << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << "]\t";
		std::cout << this->slaves_conns[i] << "\t" << this->slaves_status[i] << std::endl;
	}
}


template<typename trainer_type>
int dlib::dnn_syncer<trainer_type>::get_running_slaves_num(){
	DLIB_CASSERT(this->ismaster == 1, "Slave deivce doesn't have the right to get running_slaves_num.");
	int ret = 0;
	for(int i = 0; i < slaves_list.size(); i++){
		if(this->slaves_status[i] == 1)
			ret ++;
	}
	return ret;
}

#endif
