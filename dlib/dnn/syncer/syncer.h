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

namespace dlib
{

template <typename trainer_type>
class dnn_syncer
{
public:
	trainer_type *trainer;

	int role = device_role::undecided;
	device me;
	device master;
	connection *master_conn = NULL;

	int verbose = 1;
	int num_debug = 0;
	int exper = 0;

	// Default sync initilization
	dnn_syncer() = default;
	dnn_syncer(const dnn_syncer &) = default;
	dnn_syncer &operator=(const dnn_syncer &) = default;

	[[deprecated("Please use dnn_leader/dnn_async_leader or dnn_worker instead of dnn_syncer.")]] dnn_syncer(int role) {
		this->role = role;
	}

		[[deprecated("Please use dnn_leader/dnn_async_leader or dnn_worker instead of dnn_syncer.")]] dnn_syncer(trainer_type *trainer, int role)
	{
		this->trainer = trainer;
		this->role = role;
	}

	~dnn_syncer() = default;

	void set_role(int);

	void set_this_device(device);

	/*
	 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
	 *	void(*)
	 */
	void print_slaves_status();

	int get_running_slaves_num();

	int wait_for_master_init()
	{
		DLIB_CASSERT(this->role != device_role::supleader, "Super leader deivce doesn't need to wait for being initialized.");

		listener *lt;

		if (create_listener(lt, me.port, me.ip))
		{
			std::cerr << "Unable to create a listener" << std::endl;
			return 0;
		}

		connection *master_conn;

		if (!lt->accept(master_conn))
		{
			char master_msg[30];
			master_conn->read(master_msg, 30);
			char reply_msg[30];
			snprintf(reply_msg, sizeof(reply_msg), "%s:%d\n", &me.ip[0], me.port);
			master_conn->write(reply_msg, 30);

			std::cout << "Connected by " << master_msg << std::endl;
			this->master_conn = master_conn;
			return 1;
		}

		return 0;
	}

	int dispatch_jobs(int slave_index, task_op task)
	{
		if (this->slaves_status[slave_index] != slaveStatus::Running)
			return 0;

		if (sizeof(task) != 24)
		{
			std::cout << "The task is not 24 bytes, it is " << sizeof(task) << std::endl;
			return 0;
		}

		// char *task_message = new char[24];
		// char *dest = task_message;
		// std::memcpy(dest, (char *)&task.task_op, sizeof(task.opcode));
		// dest += sizeof(task.opcode);
		// std::memcpy(dest, (char *)&task.reserved, sizeof(task.reserved));
		// dest += sizeof(task.reserved);
		// std::memcpy(dest, (char *)&task.operand1, sizeof(task.operand1));
		// dest += sizeof(task.operand1);
		// std::memcpy(dest, (char *)&task.operand2, sizeof(task.operand2));

		this->slaves_conns[slave_index]->write((char*)&task.opcode, 24);

		return 1;
	}

	int notify_finish()
	{
		// TODO

		return 1;
	}

	void init_thread_pool()
	{
	}

	void average(std::vector<std::vector<resizable_tensor>> &all_tensors)
	{
		std::vector<std::vector<tensor *>> accessible_groups;
		float scale = 1.0 / all_tensors.size();

		for (size_t i = 0; i < this->trainer->num_computational_layers; i++)
		{
			std::vector<tensor *> group;

			for (size_t j = 0; j < all_tensors.size(); j++)
			{
				if (all_tensors[j][i].size() != 0 && this->slaves_status[j] == slaveStatus::Running)
				{
					group.push_back(&all_tensors[j][i]);
					// std::cout << &all_tensors[j][i] << std::endl;
				}
			}

			if (group.size() == 0)
				continue;

			if (group.size() == 1)
				tt::affine_transform(*group[0], *group[0], scale);
			else
				tt::affine_transform(*group[0], *group[0], *group[1], scale, scale);

			for (size_t i = 2; i < group.size(); ++i)
				tt::affine_transform(*group[0], *group[0], *group[i], 1, scale);
		}
	}

	void update(std::vector<tensor *> &updated);

	std::vector<device> slaves_list;
	std::vector<connection *> slaves_conns;
	std::vector<slaveStatus> slaves_status;
	std::vector<int> slaves_capacities;

	// TODO
	dnn_syncer &operator<<(std::ostream &out)
	{
		out << trainer << std::endl;
		out << role << std::endl;
	}
};

template <typename trainer_type>
class dnn_leader : public dnn_syncer<trainer_type>
{

public:
	dnn_leader() = default;
	dnn_leader(const dnn_leader &) = default;
	dnn_leader &operator=(const dnn_leader &) = default;

	dnn_leader(int role)
	{
		this->role = role;
	}

	dnn_leader(trainer_type *trainer, int role)
	{
		this->trainer = trainer;
		this->role = role;
	}

	~dnn_leader(){};

	void add_slave(device);
	void remove_slave(size_t);

	void init_slaves();

	void shut_slaves();

	void send_parameters(connection *slave);

	void send_parameters_to_slaves_serialised();

	void send_parameters_to_slaves_paralized();

	int receive_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors);

	void init_before_receiving(std::vector<std::vector<resizable_tensor>> &all_tensors);

	void receive_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors);

	void receive_gradients_parallism(std::vector<std::vector<resizable_tensor>> &all_tensors);

	void update_gradients(std::vector<tensor *> &gradients);

	void sn_sync();
};

template <typename trainer_type>
class dnn_async_leader : public dnn_leader<trainer_type>
{

public:
	dnn_async_leader() = default;
	dnn_async_leader(const dnn_async_leader &) = default;
	dnn_async_leader &operator=(const dnn_async_leader &) = default;

	dnn_async_leader(int role)
	{
		this->role = role;
	}

	dnn_async_leader(trainer_type *trainer, int role)
	{
		this->trainer = trainer;
		this->role = role;
	}

	void init_receiver_pool();

	int receive_gradients_from_one(int slave_index, std::vector<resizable_tensor> &cli_tensors);

	void send_parameters(int slave_index, std::vector<resizable_tensor> &parameters);

	void sync(unsigned long);

	int ending_time;

private:
	std::vector<int> counter;

	void async_thread(int);

	std::vector<std::thread *> receivers;

	std::vector<std::vector<resizable_tensor>> latest_paras;
	bool *idle_worker;

	signaler **job_signal;
	mutex **job_signal_mutex;
	bool *signal_status;

	task_queue tq;
};

template <typename trainer_type>
class dnn_worker : public dnn_syncer<trainer_type>
{
public:
	dnn_worker() = default;
	dnn_worker(const dnn_worker &) = default;
	dnn_worker &operator=(const dnn_worker &) = default;

	dnn_worker(trainer_type *trainer)
	{
		this->trainer = trainer;
	}

	task_op wait_for_task();

	void send_gradients_to_master();

	void receive_latest_parameters(std::vector<resizable_tensor> &updated);

	void pre_train(task_op operation);
};

} // End of Namespace dlib

#endif
