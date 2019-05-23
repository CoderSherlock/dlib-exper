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
	// this->send_back_flags.resize (this->get_running_slaves_num());
	this->send_back_flags = new int[this->get_running_slaves_num()];

	for (size_t i = 0; i < this->send_back_paras.size(); i++) {
		this->send_back_paras[i].resize (this->trainer->num_computational_layers);

		for (size_t j = 0; j < this->send_back_paras[i].size(); j++) {
			this->send_back_paras[i][j].copy_size (*tensors[j]);
		}

		this->send_back_flags[i] = 0;
	}

	// Just for exper
	this->counter.resize (this->get_running_slaves_num());

	for (auto i : this->counter) {
		i = 0;
	}

	// Initialize reciever threads
	this->recievers.resize (this->get_running_slaves_num());

	for (size_t i = 0; i < this->recievers.size(); i++) {
		this->recievers[i] = new std::thread (&dnn_async_leader::async_thread, this, i);
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

		if (this->slaves_status[slave_index] != slaveStatus::Running)
			break;

		std::cout << "Recieved from slave " << slave_index << std::endl;

		auto process_time = system_clock::now();

		task t (slave_index, 1, gradients);
		this->tq.add_task (t);
		this->trainer->train_noop();	// HPZ: Very important

		while (this->send_back_flags[slave_index] == 0) {}

		std::cout << "(proc time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - process_time).count() << std::endl;  // HPZ: Counting


		auto sync_time = system_clock::now();
		this->send_parameters (slave_index, this->send_back_paras[slave_index]);
		std::cout << "(send time " << std::chrono::duration_cast<std::chrono::milliseconds> (system_clock::now() - sync_time).count() << std::endl;  // HPZ: Counting


		this->send_back_flags[slave_index] = 0;

		// Just for exper
		this->counter[slave_index] ++;

		if (this->counter[slave_index] >= this->ending_time)
			break;
	}
};

template<typename trainer_type>
int dnn_async_leader<trainer_type>::recieve_gradients_from_one (int slave_index, std::vector<resizable_tensor> &cli_tensors) {
	// std::cout << slave_index << ":" << &this->slaves_conns << std::endl;

	try {
		for (size_t i = 0; i < cli_tensors.size(); i++) {
			if (cli_tensors[i].size() != 0) {
				network::recieve_compressed_tensor (this->slaves_conns[slave_index], &cli_tensors[i]);
			}
		}
	} catch (...) {
		std::cout << "It seems that slave " << slave_index << " closed" << std::endl;
		this->slaves_status[slave_index] = slaveStatus::NotConn;
		close_gracefully (this->slaves_conns[slave_index], 1);
	}

	return 1;
};


template<typename trainer_type>
void dnn_async_leader<trainer_type>::send_parameters (int slave_index, std::vector<resizable_tensor> &tensors) {

	for (size_t i = 0; i < tensors.size(); i++) {
		if (tensors[i].size() != 0) {
			// print_tensor (&tensors[i], 10);
			network::send_compressed_tensor (this->slaves_conns[slave_index], &tensors[i]);
		}
	}

}


template<typename trainer_type>
void dnn_async_leader<trainer_type>::sync() {

	int threshold = 3 * this->ending_time / 12;


	while (1) {

		while (this->tq.queue_lock.trylock() == 0) {};

		auto i = this->tq.queue.begin();

		for (i = this->tq.queue.begin(); i != this->tq.queue.end(); i ++) {
			if ((*i).ready == 1) {

				while (this->trainer->status_lock.trylock() == 0);

				this->trainer->synchronization_status = 0;
				this->trainer->status_lock.unlock();

				// Let's rock it!
				(*i).ready = 0;
				this->tq.queue_lock.unlock();

				// Update to trainer
				std::vector<tensor *> temp (this->trainer->num_computational_layers);

				for (size_t j = 0; j < temp.size(); j ++) {
					temp[j] = & ((*i).tensors[j]);
				}

				while (this->trainer->synchronization_status != 1) { }

				this->update_gradients (temp);

				while (this->trainer->status_lock.trylock() == 0);

				if (this->trainer->synchronization_status != 1)
					std::cout << "Something wrong with sync lock: current: " << this->trainer->synchronization_status << "\t Going to set: 2" << std::endl;

				this->trainer->synchronization_status = 2;
				this->trainer->status_lock.unlock();

				// Wait for result
				while (this->trainer->synchronization_status != 4) { }

				std::vector<tensor *> sys_tensors_ptrs;
				sys_tensors_ptrs.resize(this->trainer->num_computational_layers);
				visit_layer_parameters (this->trainer->devices[0]->net, [&] (size_t k, tensor & t) {
					// std::cout << "SP get parameteres from" << &t << std::endl;
					//this->send_back_paras[ (*i).slave_index][k] = t;
					sys_tensors_ptrs[k] = &t;
				});
				for (size_t layer = 0; layer < sys_tensors_ptrs.size(); layer ++) {
					if (sys_tensors_ptrs[layer]->size() != 0) {
						for (auto l = this->send_back_paras[(*i).slave_index][layer].begin(), m = sys_tensors_ptrs[layer]->begin(); m != sys_tensors_ptrs[layer]->end(); l++, m++) {
							*l = *m;
						}
					}
				}

				//sleep(10000);
				this->send_back_flags[ (*i).slave_index] = 1;

				while (this->tq.queue_lock.trylock() == 0) {};

				this->tq.queue.erase (i);

				this->tq.queue_lock.unlock();

				break;
			}
		}

		int printflag = 1;

		for (auto i : this->counter) {
			if (i <= threshold) {
				printflag = 0;
				break;
			}
		}

		if (printflag) {
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
		    std::cout << "|Milestone " << threshold <<" batches   |" << std::endl;
		    std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			std::cout << "---------------------------" << std::endl;
			threshold += 3 * this->ending_time / 12;
			sleep(2000);
		}

		int flag = 1;

		for (auto i : this->counter) {
			if (i < this->ending_time) {
				flag = 0;
				break;
			}
		}

		if (flag)
			break;

	}

	for (size_t i = 0; i < this->recievers.size(); i ++) {
		this->recievers[i]->join();
	}
};

}

#endif
