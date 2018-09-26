#ifndef DLIB_DNn_SYNCER_H_
#define DLIB_DNn_SYNCER_H_

#include "../trainer.h"
#include "../core.h"
#include "../solvers.h"
#include "../../statistics.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "../../serialize.h"

#include "../../pipe.h"
#include "../../threads.h"
#include "../../cuda/cuda_dlib.h"
#include "../../statistics/running_gradient.h"
#include <atomic>
#include <cstdio>
#include <set>
#include <future>
#include <exception>
#include <mutex>
#include "../../dir_nav.h"
#include "../../md5.h"

#include "../../sockets.h"
#include <bitset>
#include <thread>
#include <cstring>
#include "utils.h"
#include "utils_debug.h"


using std::chrono::system_clock;

namespace dlib {

template <typename trainer_type>
class dnn_syncer {
  public:
	trainer_type *trainer;

	int ismaster = 0;
	device me;
	device master;
	connection *master_conn = NULL;

	std::vector<device>			slaves_list;
	std::vector<connection *>	slaves_conns;
	std::vector<slaveStatus>	slaves_status;

	int verbose = 0;
	int num_debug = 1;
	int exper = 0;


	// Default sync initilization
	dnn_syncer() = default;
	dnn_syncer (const dnn_syncer &) = default;
	dnn_syncer &operator= (const dnn_syncer &) = default;

	dnn_syncer (int ism) {
		ismaster = ism;
	}

	dnn_syncer (trainer_type *trainer, int ism) {
		this->trainer = trainer;
		this->ismaster = ism;
	}

	~dnn_syncer() = default;

	void set_isMaster (int);

	void set_this_device (device);


	/*
	 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
	 *	void(*)
	 */
	void print_slaves_status();

	int get_running_slaves_num();

	int wait_for_master_init() {
		DLIB_CASSERT (ismaster == 0, "Master deivce doesn't need to wait for being initialized.");

		listener *lt;

		if (create_listener (lt, me.port, me.ip)) {
			std::cerr << "Unable to create a listener" << std::endl;
			return 0;
		}

		connection *master_conn;

		if (!lt->accept (master_conn)) {
			char master_msg[30];
			master_conn->read (master_msg, 30);
			char reply_msg[30];
			snprintf (reply_msg, sizeof (reply_msg), "%s:%d\n", &me.ip[0], me.port);
			master_conn->write (reply_msg, 30);

			std::cout << "Connected by " << master_msg << std::endl;
			this->master_conn = master_conn;
			return 1;
		}

		return 0;
	}

	int dispatch_jobs (int slave_index, int begin, int end) {
		DLIB_CASSERT (end > begin, "Job range is not valid.");

		if (this->slaves_status[slave_index] != slaveStatus::Running)
			return 0;

		return 1;
	}


	int notify_finish() {
		// TODO

		return 1;
	}


	void init_thread_pool() {

	}

	void send_gradients_to_master() {

		std::vector<tensor *> tensors;
		tensors.resize (this->trainer->num_computational_layers);

		visit_layer_parameter_gradients (trainer->devices[0]->net, [&] (size_t i, tensor & t) {
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
				network::send_compressed_tensor (master_conn, tensors[i]);
			}
		}

	}




	/********************************************************************************************************/

	int recieve_gradients_from_one (int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors) {

		for (size_t i = 0; i < cli_tensors[slave_index].size(); i++) {
			if (cli_tensors[slave_index][i].size() != 0) {
				// this->recieve_tensor(this->slave_conns[slave_index], &cli_tensors[slave_index][i]);
				network::recieve_compressed_tensor (this->slaves_conns[slave_index], &cli_tensors[slave_index][i]);

				// print_tensor(&cli_tensors[slave_index][i], cli_tensors[slave_index][i].size());
			}
		}

		return 1;
	}

	void init_before_recieving (std::vector<std::vector<resizable_tensor>> &all_tensors) {
		// Get the pointer of gradients from current device
		std::vector<tensor *> tensors;
		tensors.resize (this->trainer->num_computational_layers);
		visit_layer_parameters (trainer->devices[0]->net, [&] (size_t i, tensor & t) {
			tensors[i] = &t;
		});

		// Initialize temporary gradients contrainer from all other devices
		all_tensors.resize (slaves_status.size());

		for (size_t i = 0; i < all_tensors.size(); i++) {
			all_tensors[i].resize (this->trainer->num_computational_layers);
			std::cout << "layers:" << this->trainer->num_computational_layers << std::endl;

			for (size_t j = 0; j < all_tensors[i].size(); j++) {
				if (slaves_status[i] == slaveStatus::Running) {
					std::cout << "layer size:" << tensors[j]->size() << std::endl;
					all_tensors[i][j].copy_size (*tensors[j]);
				}
			}
		}
	}

	void recieve_gradients_serialism (std::vector<std::vector<resizable_tensor>> &all_tensors) {
		init_before_recieving (all_tensors);

		// Get gradients if there exists slave machine
		if (get_running_slaves_num() != 0) {
			for (int s_c = 0, s_c_max = slaves_status.size(); s_c < s_c_max ; s_c ++) {
				if (slaves_status[s_c] == slaveStatus::Running) {
					std::cout << "Reciveing from " << s_c << std::endl;
					recieve_gradients_from_one (s_c, all_tensors);
				}
			}
		}
	}


	void recieve_gradients_parallism (std::vector<std::vector<resizable_tensor>> &all_tensors) {
		init_before_recieving (all_tensors);
		std::vector<std::thread *> recievers;
		recievers.resize (all_tensors.size());

		for (size_t i = 0; i < recievers.size(); i++) {
			if (slaves_status[i] == slaveStatus::Running)
				recievers[i] = new std::thread (&dnn_syncer::recieve_gradients_from_one, this, i, std::ref (all_tensors));
		}

		for (size_t i = 0; i < recievers.size(); i++) {
			recievers[i]->join();
		}
	}


	void recieve_updated_parameters (std::vector<resizable_tensor> &updated) {
		// Initialize
		std::vector<tensor *> tensors;
		tensors.resize (this->trainer->num_computational_layers);
		visit_layer_parameters (trainer->devices[0]->net, [&] (size_t i, tensor & t) {
			tensors[i] = &t;
		});

		updated.resize (this->trainer->num_computational_layers);

		for (size_t i = 0; i < updated.size(); i++) {
			updated[i].copy_size (*tensors[i]);
		}


		for (size_t i = 0; i < updated.size(); i++) {
			if (updated[i].size() != 0)
				network::recieve_compressed_tensor (master_conn, &updated[i]);

			// this->print_tensor(&updated[i], 10);

		}
	}

	void average (std::vector<std::vector<resizable_tensor>> &all_tensors) {
		std::vector<std::vector<tensor *>> accessible_groups;
		float scale = 1.0 / all_tensors.size();

		for (size_t i = 0; i < this->trainer->num_computational_layers; i++) {
			std::vector<tensor *> group;

			for (size_t j = 0; j < all_tensors.size(); j++) {
				if (all_tensors[j][i].size() != 0 && this->slaves_status[j] == slaveStatus::Running) {
					group.push_back (&all_tensors[j][i]);
					// std::cout << &all_tensors[j][i] << std::endl;
				}
			}

			if (group.size() == 0)
				continue;

			if (group.size() == 1)
				tt::affine_transform (*group[0], *group[0], scale);
			else
				tt::affine_transform (*group[0], *group[0], *group[1], scale, scale);

			for (size_t i = 2; i < group.size(); ++i)
				tt::affine_transform (*group[0], *group[0], *group[i], 1, scale);
		}
	}

	void update (std::vector<tensor *> &updated) {
		std::vector<tensor *> old_tensors;
		old_tensors.resize (this->trainer->num_computational_layers);

		visit_layer_parameter_gradients (trainer->devices[0]->net, [&] (size_t i, tensor & t) {
			old_tensors[i] = &t;
		});
		// visit_layer_parameters(trainer->devices[0]->net, [&](size_t i, tensor& t){old_tensors[i] = &t;});

		for (size_t i = 0; i < old_tensors.size(); i++) {
			if (old_tensors[i]->size() != 0) {
				for (auto j = old_tensors[i]->begin(), k = updated[i]->begin(); j != old_tensors[i]->end(); j++, k++) {
					*j = *k;
				}
			}
		}
	}

	// TODO
	dnn_syncer &operator<< (std::ostream &out) {
		out << trainer << std::endl;
		out << ismaster << std::endl;

		if (ismaster)
			out << slaves_list.size() << std::endl;

	}

};

template<typename trainer_type>
class dnn_leader : public dnn_syncer<trainer_type> {

  public:
	dnn_leader() = default;
	dnn_leader (const dnn_leader &) = default;
	dnn_leader &operator= (const dnn_leader &) = default;

	dnn_leader (int ism) {
		this->ismaster = ism;
	}

	dnn_leader (trainer_type *trainer, int ism) {
		this->trainer = trainer;
		this->ismaster = ism;
	}

	~dnn_leader() {};

	void add_slave (device);
	void remove_slave (size_t);

	void init_slaves();

	void send_parameters (connection *slave, std::vector<tensor *> parameters);

	void send_parameters_to_slaves_serialised (std::vector<tensor *> parameters);

	void send_parameters_to_slaves_paralized (std::vector<tensor *> parameters);

	void sn_sync();
};

template<typename trainer_type>
class dnn_worker : public dnn_syncer<trainer_type> {
  public:
	dnn_worker() = default;
	dnn_worker (const dnn_worker &) = default;
	dnn_worker &operator= (const dnn_worker &) = default;

	dnn_worker (int ism) {
		this->ismaster = ism;
	}

	dnn_worker (trainer_type *trainer, int ism) {
		this->trainer = trainer;
		this->ismaster = ism;
	}

	void sn_sync();

};

} // End of Namespace dlib

#endif
