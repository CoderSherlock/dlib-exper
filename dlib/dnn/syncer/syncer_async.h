#ifndef DLIB_DNn_SYNCER_ASYNC_H_
#define DLIB_DNn_SYNCER_ASYNC_H_

#include "syncer.h"

namespace dlib
{

template <typename trainer_type>
void dnn_async_leader<trainer_type>::init_receiver_pool()
{

	// Initiliaze parameters storage for each thread
	std::vector<tensor *> tensors;
	tensors.resize(this->trainer->num_computational_layers);
	visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
		tensors[i] = &t;
	});

	this->latest_paras.resize(this->get_running_slaves_num());
	this->idle_worker = new bool[this->get_running_slaves_num()];

	this->job_signal_mutex = new mutex *[this->get_running_slaves_num()];
	this->job_signal = new signaler *[this->get_running_slaves_num()];
	this->signal_status = new bool[this->get_running_slaves_num()];

	for (size_t i = 0; i < this->latest_paras.size(); i++)
	{
		this->latest_paras[i].resize(this->trainer->num_computational_layers);

		for (size_t j = 0; j < this->latest_paras[i].size(); j++)
		{
			this->latest_paras[i][j].copy_size(*tensors[j]);
		}

		this->idle_worker[i] = true;
		this->job_signal_mutex[i] = new mutex();
		this->job_signal[i] = new signaler(*this->job_signal_mutex[i]);
		this->signal_status[i] = false;
	}

	// Just for exper
	this->counter.resize(this->get_running_slaves_num());

	for (auto i : this->counter)
	{
		i = 0;
	}

	// Initialize receiver threads
	this->receivers.resize(this->get_running_slaves_num());

	for (size_t i = 0; i < this->receivers.size(); i++)
	{
		this->receivers[i] = new std::thread(&dnn_async_leader::async_thread, this, i);
	}
};

template <typename trainer_type>
void dnn_async_leader<trainer_type>::async_thread(int slave_index)
{

	// Initialize the receiver structure
	std::vector<tensor *> tensors;
	tensors.resize(this->trainer->num_computational_layers);
	visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
		tensors[i] = &t;
	});

	std::vector<resizable_tensor> gradients;
	gradients.resize(this->trainer->num_computational_layers);

	for (size_t i = 0; i < gradients.size(); i++)
	{
		gradients[i].copy_size(*tensors[i]);
	}

	while (1)
	{
		this->job_signal_mutex[slave_index]->lock();
		if (this->signal_status[slave_index] != true)
			this->job_signal[slave_index]->wait();

		this->job_signal_mutex[slave_index]->unlock();

		if (this->slaves_status[slave_index] != slaveStatus::Running)
			break;

		this->signal_status[slave_index] = false;

		this->trainer->read_lock.lock();
		std::vector<tensor *> sys_tensors_ptrs;
		sys_tensors_ptrs.resize(this->trainer->num_computational_layers);
		visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t k, tensor &t) {
			sys_tensors_ptrs[k] = &t;
		});
		for (size_t layer = 0; layer < sys_tensors_ptrs.size(); layer++)
		{
			if (sys_tensors_ptrs[layer]->size() != 0)
			{
				for (auto l = this->latest_paras[slave_index][layer].begin(), m = sys_tensors_ptrs[layer]->begin(); m != sys_tensors_ptrs[layer]->end(); l++, m++)
				{
					*l = *m;
					// if (std::isnan(*m))
					// 	std::cout << "wtf" << std::endl;
				}
			}
		}
		this->trainer->read_lock.unlock();

		this->send_parameters(slave_index, this->latest_paras[slave_index]);

		this->receive_gradients_from_one(slave_index, gradients);

		std::cout << "Received from slave " << slave_index << std::endl;

		task t(slave_index, 1, gradients);
		this->tq.add_task(t);
		// this->trainer->train_noop(); // HPZ: Very important

		this->counter[slave_index]++;

		// if (this->counter[slave_index] >= this->ending_time)
		// 	break;

		this->idle_worker[slave_index] = true;
	}
};

template <typename trainer_type>
int dnn_async_leader<trainer_type>::receive_gradients_from_one(int slave_index, std::vector<resizable_tensor> &cli_tensors)
{
	// std::cout << slave_index << ":" << &this->slaves_conns << std::endl;

	try
	{
		for (size_t i = 0; i < cli_tensors.size(); i++)
		{
			if (cli_tensors[i].size() != 0)
			{
				network::receive_compressed_tensor(this->slaves_conns[slave_index], &cli_tensors[i]);
			}
		}
	}
	catch (...)
	{
		std::cout << "It seems that slave " << slave_index << " closed" << std::endl;
		this->slaves_status[slave_index] = slaveStatus::NotConn;
		close_gracefully(this->slaves_conns[slave_index], 1);
	}

	return 1;
};

template <typename trainer_type>
void dnn_async_leader<trainer_type>::send_parameters(int slave_index, std::vector<resizable_tensor> &tensors)
{

	for (size_t i = 0; i < tensors.size(); i++)
	{
		if (tensors[i].size() != 0)
		{
			// print_tensor (&tensors[i], 10);
			network::send_compressed_tensor(this->slaves_conns[slave_index], &tensors[i]);
		}
	}
}

#define BATCH_SIZE 128

template <typename trainer_type>
void dnn_async_leader<trainer_type>::sync(unsigned long training_size)
{
	int epoch = 1, batch_amount = 0, batch_pos = 0;
	unsigned long current_start = 0;

	while (1)
	{
		// Check idle workers & dispatch jobs
		for (int i = 0; i < this->slaves_status.size(); i++)
		{
			if (epoch > this->ending_time)
			{
				// std::cout << "Break!" << std::endl;
				break;
			}

			if (this->slaves_status[i] == slaveStatus::Running && this->idle_worker[i] == true)
			{
				task_op worker_job;
				unsigned long start = current_start, end = (current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? training_size - 1 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability);
				worker_job.opcode = 1;
				std::memcpy(&worker_job.operand1, &start, sizeof(worker_job.operand1));
				std::memcpy(&worker_job.operand2, &end, sizeof(worker_job.operand2));

				std::cout << worker_job.opcode << ":" << worker_job.operand1 << "~" << worker_job.operand2 << std::endl;
				this->dispatch_jobs(i, worker_job);
				this->job_signal_mutex[i]->lock();
				this->signal_status[i] = true;
				this->job_signal_mutex[i]->unlock();
				this->job_signal[i]->signal();
				this->idle_worker[i] = false;
				batch_amount += 1;

				// if (current_start + 128 >= training_size - 1)
				// {
				// 	epoch += 1;
				// 	std::cout << "-----" << std::endl;
				// 	std::cout << "|" << epoch << "|" << std::endl;
				// 	std::cout << "-----" << std::endl;
				// }

				current_start = ((current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? 0 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability));
				if (current_start == 0) {
					epoch += 1;
				}
			}
		}

		// Checking updated gradients.
		while (this->tq.queue_lock.trylock() == 0)
		{
		};
		auto i = this->tq.queue.begin();
		if (i == this->tq.queue.end())
		{

			if (batch_pos >= batch_amount)
			{
				std::cout << "Finished training task" << std::endl;
				break;
			}

			continue;
		}

		if ((*i).ready == 1)
		{

			// Let's rock it!
			(*i).ready = 0;
			this->tq.queue_lock.unlock();

			// Update to trainer
			std::vector<tensor *> temp(this->trainer->num_computational_layers);

			for (size_t j = 0; j < temp.size(); j++)
			{
				temp[j] = &((*i).tensors[j]);
			}

			this->update_gradients(temp);

			this->trainer->read_lock.lock();
			this->trainer->update_parameters();
			this->trainer->read_lock.unlock();
			//sleep(10000);

			while (this->tq.queue_lock.trylock() == 0)
			{
			};

			this->tq.queue.erase(i);
			batch_pos += 1;

			this->tq.queue_lock.unlock();
		}
	}

	for (size_t i = 0; i < this->receivers.size(); i++)
	{
		this->slaves_status[i] = slaveStatus::NotConn;
	}
	for (size_t i = 0; i < this->receivers.size(); i++)
	{
		this->job_signal[i]->signal();
		this->receivers[i]->join();
	}
};

} // namespace dlib

#endif
