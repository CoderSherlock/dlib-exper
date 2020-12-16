#ifndef DLIB_DNn_SYNCER_LEADER_H_
#define DLIB_DNn_SYNCER_LEADER_H_

#include "syncer.h"

using std::chrono::system_clock;

namespace dlib
{
	/*
 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
 *	void(*)
 */
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::print_slaves_status()
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to get slaves' status.");

		for (int i = 0; i < this->slaves_list.size(); i++)
		{
			std::cout << "[" << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << "]\t";
			std::cout << this->slaves_list[i].master << " " << this->slaves_list[i].comp_ability << "\t";
			std::cout << this->slaves_conns[i] << "\t" << this->slaves_status[i] << std::endl;
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_leader<trainer_type, data_type, label_type>::get_running_slaves_num()
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to get running_slaves_num.");
		int ret = 0;

		for (int i = 0; i < this->slaves_list.size(); i++)
		{
			if (this->slaves_status[i] == 1)
				ret++;
		}

		return ret;
	}

	
	/*==================================================================================
 *	Leader Manage functions, by PZH
 *
 *  add_slave
 *  remove_slave
 *  init_slaves
 *  shut_slaves
 *  send_parameters: send parameters to specified slave(one)
 *  send_parameters_to_slaves_serialism: send parameters to all slaves in serial
 *  send_parameters_to_slaves_paralized: send parameters to all slaves in parali
 *  sn_sync: sync procedure/ call every round
 *  receive_gradients_from_one: receive gradients from specific slave
 ===================================================================================*/
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::add_slave(device slave)
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to add slaves.");

		this->slaves_list.push_back(slave);
		connection *c;
		this->slaves_conns.push_back(c);
		this->slaves_status.push_back(slaveStatus::Initialize);
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::remove_slave(size_t index)
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to remove slaves.");

		if (index < 0 || index >= this->slaves_list.size())
		{
			std::cerr << "Removing an invalid index" << std::endl;
		}

		this->slaves_list.erase(this->slaves_list.begin() + index);
		this->slaves_conns.erase(this->slaves_conns.begin() + index);
		this->slaves_status.erase(this->slaves_status.begin() + index);
	} // End of syncer::remove_slave

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::init_slaves()
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to init slaves.");

		for (int i = 0; i < this->slaves_list.size(); i++)
		{
			if (create_connection(this->slaves_conns[i], (unsigned short)this->slaves_list[i].port, this->slaves_list[i].ip, (unsigned short)0, this->me.ip))
			{
				std::cerr << "Create failed on " << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << std::endl;
				this->slaves_status[i] = slaveStatus::NotConn;
				continue;
			}

			connection *conn = this->slaves_conns[i];

			// Greeting messages
			char init_msg[30] = {0};
			snprintf(init_msg, sizeof(init_msg), "%s:%d\n", &this->me.ip[0], this->me.port);
			conn->write(init_msg, 30);
			char reply_msg[30];
			conn->read(reply_msg, 30);

			// Validating if slave ip and port is correct
			char *cptr = strchr(reply_msg, ':');
			char *eptr = strchr(reply_msg, '\n');

			if (!(std::string(reply_msg, cptr - reply_msg) == this->slaves_list[i].ip &&
				  atoi(std::string(cptr + 1, eptr - cptr + 1).c_str()) == this->slaves_list[i].port))
			{
				std::cerr << "Error in validating slaves" << std::endl;
				this->slaves_status[i] = slaveStatus::FailVal;
				continue;
			}

			this->slaves_status[i] = slaveStatus::Running;
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::shut_slaves()
	{
		DLIB_CASSERT(this->role != device_role::worker && this->role != device_role::undecided, "Worker/Undecided deivce doesn't have the right to init slaves.");

		for (int i = 0; i < this->slaves_list.size(); i++)
		{
			this->slaves_conns[i]->shutdown();
			delete this->slaves_conns[i];

			this->slaves_status[i] = slaveStatus::NotConn;
		}
	}

	// TODO:: slave is dependent on index
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::send_parameters(connection *slave)
	{

		std::vector<tensor *> tensors;
		tensors.resize(this->trainer->num_computational_layers);

		visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
			// std::cout << "SP get parameteres from" << &t << std::endl;
			tensors[i] = &t;
		});

		for (size_t i = 0; i < tensors.size(); i++)
		{
			if (tensors[i]->size() != 0)
			{
				print_tensor(tensors[i], 10);
				network::send_compressed_tensor(slave, tensors[i]);
			}
		}
	}

	/******************************************************
 *	Serialized send all tensors to all alive slaves
 *
 ******************************************************/
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::send_parameters_to_slaves_serialized()
	{
		if (this->get_running_slaves_num() != 0)
		{

			for (int s_c = 0, s_c_max = this->slaves_status.size(); s_c < s_c_max; s_c++)
			{
				if (this->slaves_status[s_c] == slaveStatus::Running)
				{
					send_parameters(this->slaves_conns[s_c]);
				}
			}
		}
	}

	/******************************************************
 *	Paralized send all tensors to all alive slaves
 *
 ******************************************************/
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::send_parameters_to_slaves_paralized()
	{
		std::vector<std::thread *> receivers;
		receivers.resize(this->slaves_list.size());

		for (size_t i = 0; i < receivers.size(); i++)
		{
			if (this->slaves_status[i] == slaveStatus::Running)
			{
				std::cout << "Send to worker " << i << std::endl;
				receivers[i] = new std::thread(&dnn_leader::send_parameters, this, this->slaves_conns[i]);
			}
		}

		for (size_t i = 0; i < receivers.size(); i++)
		{
			receivers[i]->join();
		}

		for (size_t i = 0; i < receivers.size(); i++)
		{
			delete receivers[i];
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_leader<trainer_type, data_type, label_type>::receive_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors)
	{
		this->wait_finishing(this->slaves_conns[slave_index]);

		this->notify_send_begin(this->slaves_conns[slave_index]);

		for (size_t i = 0; i < cli_tensors[slave_index].size(); i++)
		{
			if (cli_tensors[slave_index][i].size() != 0)
			{
				// this->receive_tensor(this->slave_conns[slave_index], &cli_tensors[slave_index][i]);
				network::receive_compressed_tensor(this->slaves_conns[slave_index], &cli_tensors[slave_index][i]);

				// print_tensor(&cli_tensors[slave_index][i], cli_tensors[slave_index][i].size());
			}
		}

		return 1;
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::init_before_receiving(std::vector<std::vector<resizable_tensor>> &all_tensors)
	{
		// Get the pointer of gradients from current device
		std::vector<tensor *> tensors;
		tensors.resize(this->trainer->num_computational_layers);
		visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
			tensors[i] = &t;
		});

		// Initialize temporary gradients contrainer from all other devices
		all_tensors.resize(this->slaves_status.size());

		for (size_t i = 0; i < all_tensors.size(); i++)
		{
			all_tensors[i].resize(this->trainer->num_computational_layers);

			for (size_t j = 0; j < all_tensors[i].size(); j++)
			{
				if (this->slaves_status[i] == slaveStatus::Running)
				{
					all_tensors[i][j].copy_size(*tensors[j]);
				}
			}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::receive_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors)
	{
		init_before_receiving(all_tensors);

		// Get gradients if there exists slave machine
		if (this->get_running_slaves_num() != 0)
		{
			for (int s_c = 0, s_c_max = this->slaves_status.size(); s_c < s_c_max; s_c++)
			{
				if (this->slaves_status[s_c] == slaveStatus::Running)
				{
					// std::cout << "Reciveing from " << s_c << std::endl;
					receive_gradients_from_one(s_c, all_tensors);
				}
			}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::receive_gradients_parallism(std::vector<std::vector<resizable_tensor>> &all_tensors)
	{
		init_before_receiving(all_tensors);
		std::vector<std::thread *> receivers;
		receivers.resize(all_tensors.size());

		for (size_t i = 0; i < receivers.size(); i++)
		{
			if (this->slaves_status[i] == slaveStatus::Running)
				receivers[i] = new std::thread(&dnn_leader::receive_gradients_from_one, this, i, std::ref(all_tensors));
		}

		for (size_t i = 0; i < receivers.size(); i++)
		{
			receivers[i]->join();
		}

		for (size_t i = 0; i < receivers.size(); i++)
		{
			delete receivers[i];
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::update_gradients(std::vector<tensor *> &gradients)
	{
		std::vector<tensor *> old_tensors;
		old_tensors.resize(this->trainer->num_computational_layers);

		visit_layer_parameter_gradients(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
			old_tensors[i] = &t;
		});

		for (size_t i = 0; i < old_tensors.size(); i++)
		{
			if (old_tensors[i]->size() != 0)
			{
				for (auto j = old_tensors[i]->begin(), k = gradients[i]->begin(); j != old_tensors[i]->end(); j++, k++)
				{
					*j = *k;
				}
			}
		}
	}

	
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::downstream_thread(int child_index, connection *src, bool *stop_flag)
	{
		task_op operation;
		while(*stop_flag)
		{
			memset(&operation, '\0', sizeof(task_op));
			// Read a control message as request
			operation = this->wait_for_task(src);
			// Receive the payload
			switch(operation.opcode)
			{
				case task_type::train_one_batch:
				{
					// This seems not possible to be triggered.
					break;
				}
				case task_type::send_trained_parameter:
				{
					break;

				} 
				case task_type::stop_train:
				{
					// This seems not possible to be triggered.
					break;
				}
				case task_type::request_one_batch:
				{
					// Add request into primary upstream_queue.
					// Wait for its task is ready.
					break;
				}
				default: 
				{

				}
			}
			// Send a control message as response
		}
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_leader<trainer_type, data_type, label_type>::dispatch_jobs(int slave_index, task_op task)
	{
		if (slave_index == -1)
		{
			if(!network::send_a_task(this->master_conn, task))
				return 0;
		}
		else
		{
			if (this->slaves_status[slave_index] != slaveStatus::Running)
				return 0;

			if(!network::send_a_task(this->slaves_conns[slave_index], task))
				return 0;
		}
		return 1;
	}

	// [[deprected]]
	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::sn_sync()
	{

		if (this->role == device_role::supleader)
		{
			std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);

			std::vector<std::vector<resizable_tensor>> all_tensors;

			////////////////////////////////////////////////////////////
			auto epoch_time = system_clock::now(); // HPZ: Counting
			////////////////////////////////////////////////////////////

			// receive_gradients_serialism(all_tensors);
			this->receive_gradients_parallism(all_tensors);

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (this->exper)																											   //
				std::cout << "(Time for receive_tensor) is "																			   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; //

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (this->num_debug)
			{
				for (size_t i = 0; i < all_tensors.size(); i++)
				{
					std::cout << "Gradient from slave " << i << std::endl;

					for (size_t j = 0; j < all_tensors[i].size(); j++)
					{
						if (all_tensors[i][j].size() != 0)
						{
							// std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
							// print_tensor(&all_tensors[i][j], 10);
						}
					}
				}
			}

			////////////////////////////////////////////////////
			epoch_time = system_clock::now(); // HPZ: Counting
			////////////////////////////////////////////////////

			this->average(all_tensors);

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (this->exper)																											   //
				std::cout << "(Time for average) is "																					   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; // HPZ: Counting   	//

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (this->num_debug)
			{
				std::cout << "Averaged gradient: " << std::endl;

				for (size_t j = 0; j < all_tensors[0].size(); j++)
				{
					if (all_tensors[0][j].size() != 0)
					{
						// std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
						// print_tensor(&all_tensors[0][j], 10);
					}
				}
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			epoch_time = system_clock::now(); // HPZ: Counting																							  //
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			std::vector<tensor *> temp(this->trainer->num_computational_layers);

			for (size_t i = 0; i < temp.size(); i++)
			{
				// TODO : Deal with 0
				temp[i] = &all_tensors[0][i];
			}

			this->update_gradients(temp);

			this->trainer->update_parameters();

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (this->exper)																											   //
				std::cout << "(Time for update) is "																					   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; //

			epoch_time = system_clock::now(); // HPZ: Counting																						  	//
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			send_parameters_to_slaves_paralized();

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (this->exper)																											   //
				std::cout << "(Time for syncback) is "																					   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; //

			epoch_time = system_clock::now(); // HPZ: Counting																							 //
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (this->num_debug)
			{
				for (size_t i = 0; i < all_tensors.size(); i++)
				{
					for (size_t j = 0; j < all_tensors[i].size(); j++)
					{
						// print_tensor(&all_tensors[i][j], 10);
					}
				}
			}

			std::cout << "Sync finished" << std::endl;
			//	sleep(1000);
		}
		else if (this->role == device_role::leader)
		{
			std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);
			std::vector<std::vector<resizable_tensor>> all_tensors;

			task_op operation = this->wait_for_task();
			switch (operation.opcode)
			{
			case task_type::train_one_batch:
			{
				unsigned long start = *(unsigned long *)&operation.operand1, end = *(unsigned long *)&operation.operand2;
				std::cout << start << "~" << end << std::endl;

				break;
			}
			default:
			{
				// HPZ: TODO
				std::cout << "Error op" << std::endl;
			}
			}

			auto epoch_time = system_clock::now();

			// receive_gradients_serialism(all_tensors);
			this->receive_gradients_parallism(all_tensors);

			if (this->exper)
				std::cout << "(Time for receive_tensor) is "
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;

			if (this->num_debug)
			{
				for (size_t i = 0; i < all_tensors.size(); i++)
				{
					std::cout << "Gradient from slave " << i << std::endl;

					for (size_t j = 0; j < all_tensors[i].size(); j++)
					{
						if (all_tensors[i][j].size() != 0)
						{
							// std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
							// print_tensor(&all_tensors[i][j], 10);
						}
					}
				}
			}

			epoch_time = system_clock::now(); // HPZ: Counting

			this->average(all_tensors);

			if (this->exper)
				std::cout << "(Time for average) is "
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;

			if (this->num_debug)
			{
				std::cout << "Averaged gradient: " << std::endl;

				for (size_t j = 0; j < all_tensors[0].size(); j++)
				{
					if (all_tensors[0][j].size() != 0)
					{
						// std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
						// print_tensor(&all_tensors[0][j], 10);
					}
				}
			}

			epoch_time = system_clock::now(); // HPZ: Counting																							  //

			std::vector<tensor *> temp(this->trainer->num_computational_layers);

			for (size_t i = 0; i < temp.size(); i++)
			{
				// TODO : Deal with 0
				temp[i] = &all_tensors[0][i];
			}

			for (size_t i = 0; i < temp.size(); i++)
			{
				std::cout << i << " " << temp[i]->size() << std::endl;

				if (temp[i]->size() != 0)
				{
					network::send_compressed_tensor(this->master_conn, temp[i]);
				}
			}

			if (this->exper)																											   //
				std::cout << "(Time for update) is "																					   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; //

			epoch_time = system_clock::now(); // HPZ: Counting																						  	//

			send_parameters_to_slaves_paralized();

			if (this->exper)																											   //
				std::cout << "(Time for syncback) is "																					   //
						  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl; //

			epoch_time = system_clock::now(); // HPZ: Counting																							 //

			if (this->num_debug)
			{
				for (size_t i = 0; i < all_tensors.size(); i++)
				{
					for (size_t j = 0; j < all_tensors[i].size(); j++)
					{
						// print_tensor(&all_tensors[i][j], 10);
					}
				}
			}

			std::cout << "Sync finished" << std::endl;
			//	sleep(1000);
		}
		else
		{
		}
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::sync()
	{
		std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);
		std::vector<std::vector<resizable_tensor>> all_tensors;
		auto breakdown = system_clock::now(); // *_*

		this->send_parameters_to_slaves_paralized();
		this->receive_gradients_parallism(all_tensors);

		std::cout << this->me.ip << ":" << this->me.port << " collected all data" << std::endl;

		std::cout << "(waitchild " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*
		breakdown = system_clock::now();

		this->average(all_tensors);

		std::vector<tensor *> temp(this->trainer->num_computational_layers);

		for (size_t i = 0; i < temp.size(); i++)
		{
			// TODO : Deal with 0
			temp[i] = &all_tensors[0][i];
		}

		this->notify_train_finish();
		this->wait_to_send();

		std::cout << "(average+wait " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*
		breakdown = system_clock::now();

		for (size_t i = 0; i < temp.size(); i++)
		{
			std::cout << i << " " << temp[i]->size() << std::endl;

			if (temp[i]->size() != 0)
			{
				network::send_compressed_tensor(this->master_conn, temp[i]);
			}
		}

		std::cout << "(send " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::subdispatch(unsigned long all_start, unsigned long all_end)
	{
		unsigned long diff = all_end - all_start;
		unsigned long share = std::ceil(diff / this->me.comp_ability);
		unsigned long current_start = all_start;

		for (int i = 0; i < this->slaves_status.size(); i++)
		{
			if (this->slaves_status[i] == slaveStatus::Running)
			{
				task_op worker_job;
				unsigned long start = current_start, end = (current_start + share * this->slaves_list[i].comp_ability >= all_end - 1 ? all_end : current_start + share * this->slaves_list[i].comp_ability);
				worker_job.opcode = 1;
				std::memcpy(&worker_job.operand1, &start, sizeof(worker_job.operand1));
				std::memcpy(&worker_job.operand2, &end, sizeof(worker_job.operand2));

				std::cout << worker_job.opcode << ":" << worker_job.operand1 << "~" << worker_job.operand2 << std::endl;
				this->dispatch_jobs(i, worker_job);

				current_start = ((current_start + share * this->slaves_list[i].comp_ability >= all_end - 1 ? 0 : current_start + share * this->slaves_list[i].comp_ability));
			}
		}
	};

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_leader<trainer_type, data_type, label_type>::endless_sync()
	{
		if (this->me.sync_type == device_sync_type::sync)
		{

			/* Sync */

			while (1)
			{

				// Wait for all requests
				task_op children_request_tasks[this->slaves_status.size()];
				for (int i = 0; i < this->slaves_status.size(); i++)
				{
					children_request_tasks[i] = this->wait_for_task(this->slaves_conns[i]);
				}

				// Aggregate requests
				int all_request_size = 0;
				for (int i = 0; i < sizeof(children_request_tasks)/sizeof(task_op); i++)
				{
					if (children_request_tasks[i].opcode == 4)
					{
						all_request_size += children_request_tasks[i].reserved;
					}
				}
				if (this->verbose)
				{
					std::cout << "[INFO] Node " << this->me.ip << ":" << this->me.port << " received " << all_request_size << " batchs request" << std::endl;
				}

				// Send combined requests
				task_op request_to_parent_node_task;
				request_to_parent_node_task.opcode = 4;
				request_to_parent_node_task.reserved = all_request_size;
				this->dispatch_jobs(-1, request_to_parent_node_task);

				// Wait for parent's response
				task_op response_from_parent_node_task = this->wait_for_task();
				switch (response_from_parent_node_task.opcode)
				{
				case task_type::train_one_batch:
				{
					// Divide task for all children
					unsigned long start = *(unsigned long *)&response_from_parent_node_task.operand1, end = *(unsigned long *)&response_from_parent_node_task.operand2;
					if (this->verbose)
					{
						std::cout << start << "~" << end << std::endl;
						std::cout << "diff:" << end - start << std::endl;
					}

					// Respond children
					this->subdispatch(start, end);

					// Recv stale_parameter
					std::vector<resizable_tensor> latest_parameters;
					this->receive_latest_parameters(this->master_conn, latest_parameters);
					std::vector<tensor *> temp(this->trainer->num_computational_layers);
					for (size_t i = 0; i < temp.size(); i++)
					{
						temp[i] = &latest_parameters[i];
					}
					this->update(temp);

					// Send stale_paramters
					// Wait for children finishing training
					// Notify children to send updated parameters
					// Recv parameters
					// Aggregate parameters
					// Send parameter to parent
					this->sync(); // This was supposed to point to sync_leader::sync which located at class dnn_leader, TODO
				}
				default:
				{
					// HPZ: TODO
					std::cout << "Error op" << std::endl;
					break;
				}
				}
			}
		}
		else
		{
			// Async

			while (1)
			{
				// Wait for requests

				// Aggregate requests if needed

				// Send combined requests

				// Wait for parent's response

				// Divide task for all children

				// Respond children

				// Send paramters

				// Wait for children finishing training

				// Notify children to send updated parameters

				// Recv parameters

				// Aggregate parameters

				// Send parameter to parent
			}
		}
	};

} // End of Namespace dlib

#endif
