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

		device_role role = device_role::undecided;
		device me;
		device master;

		
		connection *master_conn = NULL;

		std::thread *listener_thread_ptr = NULL;
		mutex *worker_thread_lock = NULL;
		std::list<std::thread *> worker_threads;
		local_msg_list *incoming_message = NULL;

		int verbose = 1;
		int num_debug = 0;
		int exper = 0;

		std::ofstream * logfile;

		// Default sync initilization
		dnn_syncer() = default;
		dnn_syncer(const dnn_syncer &) = default;
		dnn_syncer &operator=(const dnn_syncer &) = default;

		[[deprecated("Please use dnn_leader/dnn_full_leader or dnn_worker instead of dnn_syncer.")]] 
		dnn_syncer(device_role role);
		[[deprecated("Please use dnn_leader/dnn_full_leader or dnn_worker instead of dnn_syncer.")]] 
		dnn_syncer(trainer_type *trainer, device_role role);

		~dnn_syncer() = default;

		void set_role(device_role);
		void set_this_device(device);
		void set_master_device(device);

		[[deprecated("Temp removed and replaced by listen thread for server mode")]]
		int wait_for_master_init();

		void init_thread_pool();
		void listener_thread();
		void listener_worker_thread(connection *conn);

		void average(std::vector<std::vector<resizable_tensor>> &all_tensors);
		void average_ptr(std::vector<std::vector<tensor *>> &all_tensors);

		void update(std::vector<tensor *> &updated);

		task_op request_a_task(task_op req);
		void request_a_task(connection *src_conn, task_op req);
		task_op wait_for_task();
		task_op wait_for_task(connection *src_conn);

		void receive_latest_parameters(connection* src, std::vector<resizable_tensor> &updated);
		void notify_train_finish();
		void notify_send_begin(connection *conn);
		void wait_finishing(connection *conn);
		void wait_to_send();

		void init_trainer(); // Initial trainer before start to request tasks, runs once when at beginning of the bootup.
		void init_trainer(dataset<data_type, label_type> training); // Initial trainer before start to request tasks, runs once when at beginning of the bootup.


		// TODO
		std::ostream& operator<<(std::ostream &out);
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_leader : public dnn_syncer<trainer_type, data_type, label_type>
	{
		/* 	Class dnn_full_leader was written in 2019 mid-fall by PZH.
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

		dnn_leader(trainer_type *trainer, device_role role)
		{
			this->trainer = trainer;
			this->role = role;
		}

		~dnn_leader(){};

		std::vector<device> slaves_list;		// List of slaves' device meta
		std::vector<connection *> slaves_conns; // List of slaves' connection
		std::vector<slaveStatus> slaves_status; // List of slaves' stauts, such as RUNNING / NOTCONN / INITIALIZE
		std::vector<int> slaves_capacities;		// List of slaves' compute capabilities, this is reserved for future's heterogeneity archs

		int isSerialized = 1;
		mutex *serialized_upstream_lock;
		int childAmount = 0;
		volatile int sync_child_indicator = 0;
		mutex *sync_child_indicator_mutex;
		std::vector<std::vector<tensor *>> sync_global_paras;
		

		void add_slave(device);		  // Add a device to the slave list
		void remove_slave(size_t);	  // Remove a slave by its index from slave list
		void init_slaves();			  // Initialize slaves_conns/ slaves_status accorading to slave list
		void shut_slaves();			  // Shutdown all children, never been used! @PZH
		void print_slaves_status();	  // Print all slaves' status
		int get_running_slaves_num(); // Get number of how many slaves are in running.

		void send_parameters(connection *slave);	 // Send leader's trainer's parameter (in-memory) directly to the connection
		void send_parameters_to_slaves_serialized(); // Send to all children in serialized using above function
		void send_parameters_to_slaves_paralized();	 // Send to all children in paralized, same as up

		void init_before_receiving(std::vector<std::vector<resizable_tensor>> &all_tensors);					  // Allocate space exactly size for each child it has before recv from them
		int receive_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors); // Receive gradients from the child given index
		void receive_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors);				  // Receive gradients in serialized using above function
		void receive_gradients_parallism(std::vector<std::vector<resizable_tensor>> &all_tensors);				  // Receive gradients in paralized, same as up
		void update_gradients(std::vector<tensor *> &gradients);												  // Update **gradients** to trainer, not perform update to parameter

		void send_parameters_wp(connection *worker, std::vector<resizable_tensor> parameters);	 // Send leader's trainer's parameter (in-memory) directly to the connection

		void init_thread_pool();
		void listener_thread();
		void listener_worker_thread(connection *conn);

		int dispatch_jobs(int slave_index, task_op task);		  // Dispatch a task to the slave specified by given index
		void sn_sync();											  //
		void sync();											  //
		void subdispatch(unsigned long start, unsigned long end); //
		void endless_sync();

	protected:
		// queue_lock *send_lock = new queue_lock(100);
		// queue_lock *recv_lock = new queue_lock(100);
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	class dnn_full_leader : public dnn_leader<trainer_type, data_type, label_type>
	{
		/* 	Class dnn_full_leader was written in 2019 mid-fall by PZH.
		 *	This class was designed to be used as middle-layer node and top-leader, but only remains top-leader's implementation. 
		 *	However, two functions receive_gradients_from_one & send_parameters are still usable in future's scheme.
		 *	Separate sync thread and task queue practice make a good overcome in concurrent performance. 
		 */
	public:
		int ending_time;

		dnn_full_leader() = default;
		dnn_full_leader(const dnn_full_leader &) = default;
		dnn_full_leader &operator=(const dnn_full_leader &) = default;
		dnn_full_leader(device_role role)
		{
			this->role = role;
		}

		dnn_full_leader(trainer_type *trainer, device_role role)
		{
			this->trainer = trainer;
			this->role = role;
		}

		void init_receiver_pool();																	 	// Initialize all async working thread and all member variables defined below
		int receive_gradients_from_device(connection *src, std::vector<resizable_tensor> &cli_tensors); // Recv parameters from a specified slave
		void send_parameters(int slave_index, std::vector<resizable_tensor> &parameters);			 	// Send parameters to the specified slave, similar to the above function
		void subsync(unsigned long);																 	// ONGOING: prototype of configurable middle-layer node
		void sync(dataset<matrix<unsigned char>, unsigned long> *testing);							 	// Async-based top leader, endless loop until finishing amount of epochs or reaching an accuracy level of passed-in dataset

		void init_thread_pool();
		void listener_thread();
		void listener_worker_thread(connection *conn);

		volatile unsigned long epoch = 0;
		unsigned long current_start = 0;
		mutex *current_start_lock;

	private:
		void async_thread(int); // Working thread function for each child

		std::vector<int> counter;								 // Amount of how many batches has been computed in current thread
		std::vector<std::thread *> receivers;					 // Store working async_thread in here
		std::vector<std::vector<resizable_tensor>> latest_paras; // Temporarily store synced/updated parameter copy for each child
		bool *idle_worker;										 // Indicate whether worker is idle or not, True = idle, False = busy
		signaler **job_signal;									 // Signaler to sync status between main thread and async_threads
		mutex **job_signal_mutex;								 // Mutex used by above signaler
		bool *signal_status;									 // To indicate ready if main thread has a fast pace than receiver thread which has not set signal wait yet

		
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

		void send_gradients_to_master();  					// Send gradients to its parent node
		void send_parameters_to_master(); 					// Send parameters to its parent node
		void send_parameters_to_device(connection *dst); 	// Send parameters to customized destination device
		
		void recv_and_update_parameters(connection* src); 	// Receive paramters.
		void starting_pistol();								// Let trainer start.

		int do_one_task_with_wait(void);	// TODO
		int do_one_task_without_wait(void); // TODO
	};

} // End of Namespace dlib

#endif
