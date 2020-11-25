#ifndef DLIB_DNn_SYNCER_WORKER_DEFAULT_H_
#define DLIB_DNn_SYNCER_WORKER_DEFAULT_H_

#include "syncer.h"

namespace dlib
{

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::pre_train(task_op operation)
	{
		switch (operation.opcode)
		{
		case task_type::train_one_batch:
		{
			auto breakdown = system_clock::now(); // *_*

			// HPZ: Sync lateset parameters
			std::vector<resizable_tensor> latest_parameters;
			this->receive_latest_parameters(latest_parameters);

			std::vector<tensor *> temp(this->trainer->num_computational_layers);
			for (size_t i = 0; i < temp.size(); i++)
			{
				// TODO : Deal with 0
				temp[i] = &latest_parameters[i];
				if (this->num_debug)
				{
					if (temp[i]->size() != 0)
						print_tensor(temp[i], 10);
				}
			}
			this->update(temp);

			std::cout << "(recv " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*

			this->trainer->distributed_signal.get_mutex().lock();
			this->trainer->ready_status = 2;
			this->trainer->status_lock.unlock();
			this->trainer->distributed_signal.signal();

			break;
		}
		default:
		{
			// HPZ: TODO
			std::cout << "Invalid task operation" << std::endl;
			break;
		}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::send_gradients_to_master()
	{

		std::vector<tensor *> tensors;
		tensors.resize(this->trainer->num_computational_layers);

		visit_layer_parameter_gradients(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
			tensors[i] = &t;
		});

		if (this->num_debug)
		{
			for (size_t i = 0; i < tensors.size(); i++)
			{
				if (tensors[i]->size() != 0)
					print_tensor(tensors[i], 10);
			}
		}

		for (size_t i = 0; i < tensors.size(); i++)
		{
			std::cout << i << " " << tensors[i]->size() << std::endl;

			if (tensors[i]->size() != 0)
			{
				network::send_compressed_tensor(this->master_conn, tensors[i]);
			}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::send_parameters_to_master()
	{

		std::vector<tensor *> tensors;
		tensors.resize(this->trainer->num_computational_layers);

		visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
			tensors[i] = &t;
		});

		if (this->num_debug)
		{
			for (size_t i = 0; i < tensors.size(); i++)
			{
				if (tensors[i]->size() != 0)
					print_tensor(tensors[i], 10);
			}
		}

		for (size_t i = 0; i < tensors.size(); i++)
		{
			std::cout << i << " " << tensors[i]->size() << std::endl;

			if (tensors[i]->size() != 0)
			{
				network::send_compressed_tensor(this->master_conn, tensors[i]);
			}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_worker<trainer_type, data_type, label_type>::do_one_task_with_wait()
	{
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_worker<trainer_type, data_type, label_type>::do_one_task_without_wait()
	{
		// TODO: check if trainer is busy, if then wait until it become idle.

		// Request a task fit its computability
		task_op request, response;
		dataset<matrix<unsigned char>, unsigned long> local_training;
		auto batch_time = system_clock::now(); // *_* Time for the training
		auto breakdown = system_clock::now();  // *_* Breakdown time for training procedure

		request.opcode = task_type::request_one_batch;	// Request batch opcode
		request.reserved = 1; 							// Request only one batch
		request.operand1 = 0; // Not used
		request.operand2 = 0; // Not used

		network::msgheader header;
		header.dev_index = this->me.number;
		inet_pton(AF_INET, this->me.ip.c_str(), &(header.ip));
		header.port = this->me.port;
		header.type = task_type::train_one_batch;
		header.length = 24;
		try {
			connection* session = network::create_message_session(this->master.ip, this->master.port, this->me.ip);
			network::send_header(session, &header); 			// Send a batch request
			network::send_a_task(session, request);
			network::halt_message_session(session);
		} catch (std::exception e) {
			
		}

		// Wait the response from its leader
		

		switch (response.opcode)
		{
		case task_type::train_one_batch:
		{

			unsigned long start = *(unsigned long *)&response.operand1, end = *(unsigned long *)&response.operand2;
			if (this->verbose)
			{
				std::cout << start << "~" << end << std::endl;
				std::cout << "diff:" << end - start << std::endl;
			}
			this->trainer->epoch_pos = 0;
			local_training = this->default_dataset->split(start, end);
			this->trainer->set_mini_batch_size(end - start);
			if (this->verbose)
			{
				std::cout << "mini_batch:" << this->trainer->get_mini_batch_size() << std::endl;
				std::cout << "data_size:" << local_training.getData().size() << std::endl;
			}
			std::cout << "(prepare "
					  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count()
					  << std::endl; // *_* Log time for preparation
			breakdown = system_clock::now();

			this->trainer->train_one_batch(local_training.getData(), local_training.getLabel());
			this->pre_train(response);

			this->trainer->distributed_signal.get_mutex().lock();
			if (this->trainer->ready_status != 3)
				this->trainer->distributed_signal.wait();

			this->trainer->status_lock.unlock();

			std::cout << "(train+recv "
					  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count()
					  << std::endl; // *_* Log time for recving parameter and training
			
			// if(me.number % 2 == 1) sleep((unsigned int)10);	// This line is used for simulate a slow worker.

			this->notify_train_finish();
			this->wait_to_send();
			breakdown = system_clock::now();
			this->send_parameters_to_master();

			std::cout << "(send "
					  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count()
					  << std::endl; // *_* Log time for send updated parameter

			std::cout << "Learning rate is " << this->trainer->learning_rate << std::endl;
			break;
		}
		default:
		{
			// HPZ: TODO
			std::cout << "Error op" << std::endl;
			return -1;
		}
		}
		std::cout << "Time for batch is " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - batch_time).count() << std::endl; // *_*
		return 1;
	}

} // End of Namespace dlib

#endif
