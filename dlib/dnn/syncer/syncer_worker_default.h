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

	std::cout << "Sync finished" << std::endl;
	// sleep(1000);
}


} // End of Namespace dlib

#endif
