#ifndef DLIB_DNn_SYNCER_ASYNC_H_
#define DLIB_DNn_SYNCER_ASYNC_H_

#include "syncer.h"

namespace dlib {

template<typename trainer_type>
void dnn_async_leader<trainer_type>::init_reciever_pool() {

	// Initiliaze parameters storage for each thread
	std::vector<tensor *> tensors;
	tensors.resize (this->trainer->num_computational_layers);
	visit_layer_parameters (this->trainer->devices[0]->net, [&] (size_t i, tensor & t) {
		tensors[i] = &t;
	});

	this->send_back_paras.resize (this->get_running_slaves_num());
	this->send_back_flags.resize (this->get_running_slaves_num());

	for (size_t i = 0; i < this->send_back_paras.size(); i++) {
		this->send_back_paras[i].resize (this->trainer->num_computational_layers);

		for (size_t j = 0; j < this->send_back_paras[i].size(); j++) {
			this->send_back_paras[i][j].copy_size (*tensors[j]);
		}

		this->send_back_flags[i] = 0;
	}

	// Initialize reciever threads
	this->recievers.resize (this->get_running_slaves_num());

	for (size_t i = 0; i < this->recievers.size(); i++) {
		this->recievers[i] = new std::thread (&dnn_async_leader::async_thread, this, i);
	}
};

template<typename trainer_type>
void dnn_async_leader<trainer_type>::sync() {
	int last = -1;

	while (1) {
		int now = 0;

		while (this->tq.queue_lock.trylock() == 0) {};

		for (auto i = this->tq.queue.begin(); i != this->tq.queue.end(); i ++) {
			if ((*i).ready == 1) {
				now ++;
			}
		}

		this->tq.queue_lock.unlock();

		if (now != last) {
			std::cout << "Now we have " << now << " tasks." << std::endl;
			last = now;
		}

	}

	for (size_t i = 0; i < this->recievers.size(); i ++) {
		this->recievers[i]->join();
	}
};

template<typename trainer_type>
void dnn_async_leader<trainer_type>::async_thread (int slave_index) {

	// Initialize the reciever structure
	std::vector<tensor *> tensors;
	tensors.resize (this->trainer->num_computational_layers);
	visit_layer_parameters (this->trainer->devices[0]->net, [&] (size_t i, tensor & t) {
		tensors[i] = &t;
	});

	std::vector<resizable_tensor> gradients;
	gradients.resize (this->trainer->num_computational_layers);

	for (size_t i = 0; i < gradients.size(); i++) {
		gradients[i].copy_size (*tensors[i]);
	}

	while (1) {
		this->recieve_gradients_from_one (slave_index, gradients);
		std::cout << "Recieved from slave " << slave_index << std::endl;

		task t (slave_index, 1, gradients);
		this->tq.add_task (t);

		while (this->send_back_flags[slave_index] == 0) {}

		// this->send_parameters();

		this->send_back_flags[slave_index] = 0;
	}
};

template<typename trainer_type>
int dnn_async_leader<trainer_type>::recieve_gradients_from_one (int slave_index, std::vector<resizable_tensor> &cli_tensors) {
	std::cout << slave_index << ":"<<&this->slaves_conns << std::endl;
	for (size_t i = 0; i < cli_tensors.size(); i++) {
		if (cli_tensors[i].size() != 0) {
			network::recieve_compressed_tensor (this->slaves_conns[slave_index], &cli_tensors[i]);
		}
	}

	return 1;
};



}

#endif
