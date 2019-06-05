#ifndef DLIB_DNn_SYNCER_WORKER_DEFAULT_H_
#define DLIB_DNn_SYNCER_WORKER_DEFAULT_H_

#include "syncer.h"

namespace dlib
{

template <typename trainer_type>
task_op dnn_worker<trainer_type>::wait_for_task()
{
	char *task_message = new char[24];
	this->master_conn->read(task_message, 24);

	task_op ret;
	ret.opcode = *((int *)task_message);
	ret.reserved = *((int *)(task_message + 4));
	ret.operand1 = *((double *)(task_message + 8));
	ret.operand2 = *((double *)(task_message + 16));

	delete[] task_message;

	return ret;
}

template <typename trainer_type>
void dnn_worker<trainer_type>::pre_train(task_op operation)
{
	switch (operation.opcode)
	{
	case task_type::train_one_batch:
	{

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

template <typename trainer_type>
void dnn_worker<trainer_type>::send_gradients_to_master()
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

template <typename trainer_type>
void dnn_worker<trainer_type>::receive_latest_parameters(std::vector<resizable_tensor> &updated)
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
}

} // End of Namespace dlib

#endif
