#ifndef DLIB_DNn_SYNCER_WORKER_H_
#define DLIB_DNn_SYNCER_WORKER_H_

#include "syncer.h"

namespace dlib
{

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::recv_and_update_parameters(connection *src)
	{
		std::vector<resizable_tensor> latest_parameters;
		this->receive_latest_parameters(src, latest_parameters);

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

		char tmp = ' ';
		src->write(&tmp, 1);
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::starting_pistol()
	{
		this->trainer->distributed_signal.get_mutex().lock();
		this->trainer->ready_status = 2;
		this->trainer->status_lock.unlock();
		this->trainer->distributed_signal.signal();
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
	void dnn_worker<trainer_type, data_type, label_type>::send_parameters_to_device(connection *dst)
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
				network::send_compressed_tensor(dst, tensors[i]);
			}
		}
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	void dnn_worker<trainer_type, data_type, label_type>::send_parameters_to_master()
	{
		this->send_parameters_to_device(this->master_conn);
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_worker<trainer_type, data_type, label_type>::do_one_task_with_wait()
	{
		return 1;
	}

	template <typename trainer_type,
			  typename data_type,
			  typename label_type>
	int dnn_worker<trainer_type, data_type, label_type>::do_one_task_without_wait()
	{
		// TODO: check if trainer is busy, if then wait until it become idle.

		// Request a task fit its computability
		network::msgheader req_header, res_header;
		task_op req_task, res_task;
		dataset<matrix<unsigned char>, unsigned long> local_training;
		auto batch_time = system_clock::now(); // *_* Time for the training
		auto breakdown = system_clock::now();  // *_* Breakdown time for training procedure

		req_header.dev_index = this->me.number;
		inet_pton(AF_INET, this->me.ip.c_str(), &(req_header.ip));
		req_header.port = this->me.port;
		req_header.type = task_type::request_one_batch;
		req_header.length = 24;

		req_task.opcode = task_type::request_one_batch; // Request batch opcode
		req_task.reserved = 1;							// Request only one batch
		req_task.operand1 = 0;							// Not used
		req_task.operand2 = 0;							// Not used

		// try
		// {
		logger(this->logfile, this->me.number, this->master.number, 0, "Request a batch from the worker" + std::to_string(this->me.number));
		connection *session = network::create_message_session(this->master.ip, this->master.port, this->me.ip);
		std::cout << __FILE__ << ":" << __LINE__ << " " << session->get_socket_descriptor() << std::endl;
		network::send_header(session, &req_header); // Send a batch request
		network::send_a_task(session, req_task);
		network::recv_header(session, &res_header);
		network::recv_a_task(session, &res_task);
		network::halt_message_session(session);
		logger(this->logfile, this->me.number, this->master.number, 1, "Request a batch from the worker" + std::to_string(this->me.number));
		// }
		// catch (...)
		// {
		// 	std::cerr << "Something went wrong when request a training job." << std::endl;
		// }

		if (res_task.opcode == task_type::train_one_batch)
		{
			// Expected response ==> Train a customized batch.
			unsigned long start = *(unsigned long *)&res_task.operand1, end = *(unsigned long *)&res_task.operand2;
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

			// Training procedure was successfully loaded, now wait for all prerequisite fulfilled.
			this->trainer->train_one_batch(local_training.getData(), local_training.getLabel());
		}
		else
		{
			// Unexpected response ==> Report and stop this session.
			std::cout << __FILE__ << ":" << __LINE__ << " Unexpected response payload with NO train_one_batch task." << std::endl;
		}

		// Request the most updated global parameter if needed
		// TODO: We can also request training data at this phase if idea were challenged.

		memset(&req_header, '\0', sizeof(req_header));
		memset(&res_header, '\0', sizeof(res_header));

		req_header.dev_index = this->me.number;
		inet_pton(AF_INET, this->me.ip.c_str(), &(req_header.ip));
		req_header.port = this->me.port;
		req_header.type = task_type::request_updated_parameter;
		req_header.length = 0;

		try
		{
			logger(this->logfile, this->me.number, this->master.number, 0, "Request updated parameter from the worker" + std::to_string(this->me.number));
			connection *session = network::create_message_session(this->master.ip, this->master.port, this->me.ip);
			std::cout << __FILE__ << ":" << __LINE__ << " " << session->get_socket_descriptor() << std::endl;
			network::send_header(session, &req_header); // Send a batch request
			network::recv_header(session, &res_header);
			this->recv_and_update_parameters(session);
			network::halt_message_session(session);
			logger(this->logfile, this->me.number, this->master.number, 1, "Request updated parameter from the worker" + std::to_string(this->me.number));
		}
		catch (...)
		{
			std::cerr << "Something went wrong when request the latest parameters." << std::endl;
		}

		logger(this->logfile, this->me.number, this->master.number, 0, "Training");
		this->starting_pistol();


		this->trainer->distributed_signal.get_mutex().lock();
		if (this->trainer->ready_status != 3)
			this->trainer->distributed_signal.wait();
		this->trainer->status_lock.unlock();
		logger(this->logfile, this->me.number, this->master.number, 1, "Training");

		memset(&req_header, '\0', sizeof(req_header));
		memset(&res_header, '\0', sizeof(res_header));

		req_header.dev_index = this->me.number;
		inet_pton(AF_INET, this->me.ip.c_str(), &(req_header.ip));
		req_header.port = this->me.port;
		req_header.type = task_type::send_trained_parameter;
		req_header.length = 0;

		try
		{
			logger(this->logfile, this->me.number, this->master.number, 0, "Send trained parameter from the worker" + std::to_string(this->me.number));
			connection *session = network::create_message_session(this->master.ip, this->master.port, this->me.ip);
			std::cout << __FILE__ << ":" << __LINE__ << " " << session->get_socket_descriptor() << std::endl;
			network::send_header(session, &req_header); // Send a batch request
			this->send_parameters_to_device(session);
			network::recv_header(session, &res_header);
			network::halt_message_session(session);
			logger(this->logfile, this->me.number, this->master.number, 1, "Send trained parameter from the worker" + std::to_string(this->me.number));
		}
		catch (...)
		{
			std::cerr << "Something went wrong when request the latest parameters." << std::endl;
		}

		std::cout << "Learning rate is " << this->trainer->learning_rate << std::endl;

		std::cout << "Time for batch is " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - batch_time).count() << std::endl; // *_*
		return 1;
	}

} // End of Namespace dlib

#endif
