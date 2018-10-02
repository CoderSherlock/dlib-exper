#ifndef DLIB_DNn_SYNCER_WORKER_DEFAULT_H_
#define DLIB_DNn_SYNCER_WORKER_DEFAULT_H_

#include "syncer.h"

namespace dlib {

template<typename trainer_type>
void dnn_worker<trainer_type>::sn_sync() {

	std::vector<resizable_tensor> updated;

	this->send_gradients_to_master();

	this->recieve_updated_parameters (updated);

	std::vector<tensor *> temp (this->trainer->num_computational_layers);

	for (size_t i = 0; i < temp.size(); i++) {
		// TODO : Deal with 0
		temp[i] = &updated[i];

		if (this->num_debug) {
			if (temp[i]->size() != 0)
				print_tensor (temp[i], 10);
		}
	}

	this->update (temp);

	while (this->trainer->status_lock.trylock() == 0);

	this->trainer->synchronization_status = 3;
	this->trainer->status_lock.unlock();

	std::cout << "Sync finished" << std::endl;
	// sleep(1000);
}

template<typename trainer_type>
void dnn_worker<trainer_type>::send_gradients_to_master() {

	std::vector<tensor *> tensors;
	tensors.resize (this->trainer->num_computational_layers);

	visit_layer_parameter_gradients (this->trainer->devices[0]->net, [&] (size_t i, tensor & t) {
		tensors[i] = &t;
	});

	if (this->num_debug) {
		for (size_t i = 0; i < tensors.size(); i++) {
			if (tensors[i]->size() != 0)
				print_tensor (tensors[i], 10);
		}
	}

	for (size_t i = 0; i < tensors.size(); i++) {
		std::cout << i << " " << tensors[i]->size() << std::endl;

		if (tensors[i]->size() != 0) {
			network::send_compressed_tensor (this->master_conn, tensors[i]);
		}
	}

}


template<typename trainer_type>
void dnn_worker<trainer_type>::recieve_updated_parameters (std::vector<resizable_tensor> &updated) {
	// Initialize
	std::vector<tensor *> tensors;
	tensors.resize (this->trainer->num_computational_layers);
	visit_layer_parameters (this->trainer->devices[0]->net, [&] (size_t i, tensor & t) {
		tensors[i] = &t;
	});

	updated.resize (this->trainer->num_computational_layers);

	for (size_t i = 0; i < updated.size(); i++) {
		updated[i].copy_size (*tensors[i]);
	}


	for (size_t i = 0; i < updated.size(); i++) {
		if (updated[i].size() != 0)
			network::recieve_compressed_tensor (this->master_conn, &updated[i]);

		// this->print_tensor(&updated[i], 10);

	}
}



} // End of Namespace dlib

#endif
