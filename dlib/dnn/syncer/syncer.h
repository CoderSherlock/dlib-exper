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

#include "../../../examples/dnn_dist_data.h"

using std::chrono::system_clock;

namespace dlib
{

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_syncer
	{
	public:
		trainer_type *trainer;
		dataset<data_type, label_type> *default_dataset;

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

		[[deprecated("Please use dnn_leader/dnn_async_leader or dnn_worker instead of dnn_syncer.")]] dnn_syncer(int role)
		{
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
					if (all_tensors[j][i].size() != 0)
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

		void average_ptr(std::vector<std::vector<tensor *>> &all_tensors)
		{
			if (all_tensors.size() < 1)
			{
				return;
			}
			std::vector<std::vector<tensor *>> accessible_groups;
			float scale = 1.0 / all_tensors.size();

			for (size_t i = 0; i < all_tensors[0].size(); i++)
			{
				std::vector<tensor *> group;

				for (size_t j = 0; j < all_tensors.size(); j++)
				{
					group.push_back(all_tensors[j][i]);
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


		

		task_op request_a_task(task_op req)
		{
			return request_a_task(NULL, req);
		}

		void request_a_task(connection *src_conn, task_op req)
		{
			if (src_conn == NULL)
				network::send_a_task(this->master_conn, req);
			else
				network::send_a_task(src_conn, req);
		}

		task_op wait_for_task()
		{
			return wait_for_task(NULL);
		}

		task_op wait_for_task(connection *src_conn)
		{
			task_op ret;
			if (src_conn == NULL)
				network::recv_a_task(this->master_conn, &ret);
			else
				network::recv_a_task(src_conn, &ret);

			return ret;
		}

		void receive_latest_parameters(std::vector<resizable_tensor> &updated)
		{
			// Initialize
			std::vector<tensor *> tensors;
			tensors.resize(this->trainer->num_computational_layers);
			visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
				tensors[i] = &t;
			});

			updated.resize(this->trainer->num_computational_layers);

			for (size_t i = 0; i < updated.size(); i++)
			{
				updated[i].copy_size(*tensors[i]);
			}

			for (size_t i = 0; i < updated.size(); i++)
			{
				if (updated[i].size() != 0)
					network::receive_compressed_tensor(this->master_conn, &updated[i]);
			}
		};

		void notify_train_finish()
		{
			char *msg = new char[1];
			this->master_conn->write(msg, 1);
		};

		void notify_send_begin(connection *conn)
		{
			char *msg = new char[1];
			conn->write(msg, 1);
		};

		void wait_finishing(connection *conn)
		{
			char *msg = new char[1];
			conn->read(msg, 1);
		};

		void wait_to_send()
		{
			char *msg = new char[1];
			this->master_conn->read(msg, 1);
		};

		void init_trainer(dataset<data_type, label_type> training) // Initial trainer before start to request tasks, runs once when at beginning of the bootup.
		{
			while (trainer->ready_status < 1) 											// Wait for trainer get ready.
			{
			};

			this->trainer->train_one_batch(training.getData(), training.getLabel()); 	// Give trainer one batch for allocate space of parameters and gradients

			this->trainer->distributed_signal.get_mutex().lock(); 						// Notify trainer start to train
			this->trainer->ready_status = 2;
			this->trainer->status_lock.unlock();
			this->trainer->distributed_signal.signal();

			while (trainer->ready_status < 4) 											// Wait until trainer finished training
			{
			};

			if (this->verbose)	std::cout << "Wait for leader to init" << std::endl;	// Print waiting for parent to init
			if (!this->wait_for_master_init())											// Wait for parent to init, TODO: Not necessary to be after this function finished.
			{
				std::cerr << "Error happens when master send init message" << std::endl;
				exit(0);
			}
		};

		// TODO
		dnn_syncer &operator<<(std::ostream &out)
		{
			out << trainer << std::endl;
			out << role << std::endl;
		}
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_leader : public dnn_syncer<trainer_type, data_type, label_type>
	{
		/* 	Class dnn_async_leader was written in 2019 mid-fall by PZH.
		 *	This class was designed to be used as middle-layer node and top-leader, but only remains top-leader's implementation. 
		 *	However, two functions receive_gradients_from_one & send_parameters are still usable in future's scheme.
		 *	Separate sync thread and task queue practice make a good overcome in concurrent performance. 
		 */
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

		std::vector<device> slaves_list;		// List of slaves' device meta
		std::vector<connection *> slaves_conns; // List of slaves' connection
		std::vector<slaveStatus> slaves_status; // List of slaves' stauts, such as RUNNING / NOTCONN / INITIALIZE
		std::vector<int> slaves_capacities;		// List of slaves' compute capabilities, this is reserved for future's heterogeneity archs

		void add_slave(device);		  // Add a device to the slave list
		void remove_slave(size_t);	  // Remove a slave by its index from slave list
		void init_slaves();			  // Initialize slaves_conns/ slaves_status accorading to slave list
		void shut_slaves();			  // Shutdown all children, never been used! @PZH
		void print_slaves_status();	  // Print all slaves' status
		int get_running_slaves_num(); // Get number of how many slaves are in running.

		void send_parameters(connection *slave);	 // Send leader's trainer's parameter (in-memory) directly to the connection
		void send_parameters_to_slaves_serialised(); // Send to all children in serialized using above function
		void send_parameters_to_slaves_paralized();	 // Send to all children in paralized, same as up

		void init_before_receiving(std::vector<std::vector<resizable_tensor>> &all_tensors);					  // Allocate space exactly size for each child it has before recv from them
		int receive_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors); // Receive gradients from the child given index
		void receive_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors);				  // Receive gradients in serialized using above function
		void receive_gradients_parallism(std::vector<std::vector<resizable_tensor>> &all_tensors);				  // Receive gradients in paralized, same as up
		void update_gradients(std::vector<tensor *> &gradients);												  // Update **gradients** to trainer, not perform update to parameter

		int dispatch_jobs(int slave_index, task_op task);		  // Dispatch a task to the slave specified by given index
		void sn_sync();											  //
		void sync();											  //
		void subdispatch(unsigned long start, unsigned long end); //

	protected:
		// queue_lock *send_lock = new queue_lock(100);
		// queue_lock *recv_lock = new queue_lock(100);
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_async_leader : public dnn_leader<trainer_type, data_type, label_type>
	{
		/* 	Class dnn_async_leader was written in 2019 mid-fall by PZH.
		 *	This class was designed to be used as middle-layer node and top-leader, but only remains top-leader's implementation. 
		 *	However, two functions receive_gradients_from_one & send_parameters are still usable in future's scheme.
		 *	Separate sync thread and task queue practice make a good overcome in concurrent performance. 
		 */
	public:
		int ending_time;

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

		void init_receiver_pool();																	 // Initialize all async working thread and all member variables defined below
		int receive_gradients_from_one(int slave_index, std::vector<resizable_tensor> &cli_tensors); // Recv parameters from a specified slave
		void send_parameters(int slave_index, std::vector<resizable_tensor> &parameters);			 // Send parameters to the specified slave, similar to the above function
		void subsync(unsigned long);																 // ONGOING: prototype of configurable middle-layer node
		void sync(unsigned long, dataset<matrix<unsigned char>, unsigned long> *);					 // Async-based top leader, endless loop until finishing amount of epochs or reaching an accuracy level of passed-in dataset

	private:
		void async_thread(int); 									// Working thread function for each child

		std::vector<int> counter;								 	// Amount of how many batches has been computed in current thread
		std::vector<std::thread *> receivers;					 	// Store working async_thread in here
		std::vector<std::vector<resizable_tensor>> latest_paras; 	// Temporarily store synced/updated parameter copy for each child
		bool *idle_worker;										 	// Indicate whether worker is idle or not, True = idle, False = busy
		signaler **job_signal;									 	// Signaler to sync status between main thread and async_threads
		mutex **job_signal_mutex;								 	// Mutex used by above signaler
		bool *signal_status;									 	// To indicate ready if main thread has a fast pace than receiver thread which has not set signal wait yet
		local_job_fffo_queue tq;									// A queue to organize all on-going tasks, not FIFO but First-IN-First-ready-OUT
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_worker : public dnn_syncer<trainer_type, data_type, label_type>
	{
		/* 	Class dnn_worker was written in 2019 spring by PZH.
		 *	This class has been used since then early test, only modification is pre_train after 3-level update.
		 */
	public:
		dnn_worker() = default;
		dnn_worker(const dnn_worker &) = default;
		dnn_worker &operator=(const dnn_worker &) = default;
		dnn_worker(trainer_type *trainer)
		{
			this->trainer = trainer;
		}

		void send_gradients_to_master();  	// Send gradients to its parent node
		void send_parameters_to_master(); 	// Send parameters to its parent node

		void pre_train(task_op operation); 	// Receive paramters from parent node and setup training prerequesties.

		int do_one_task_with_wait(void); 	// TODO
		int do_one_task_asap(void);	  		// TODO
	};

} // End of Namespace dlib

#endif
