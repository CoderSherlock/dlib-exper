#ifndef DLIB_DNn_SYNCER_FULL_LEADER_H_
#define DLIB_DNn_SYNCER_FULL_LEADER_H_

#include "syncer.h"
#include <chrono>

using std::chrono::system_clock;

#define BATCH_SIZE 128

namespace dlib
{
    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::init_receiver_pool()
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
            this->receivers[i] = new std::thread(&dnn_full_leader::async_thread, this, i);
        }
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::async_thread(int slave_index)
    {

        // Initialize the receiver structure
        // Make a pointer point to the global parameter
        std::vector<tensor *> tensors;
        tensors.resize(this->trainer->num_computational_layers);
        visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
            tensors[i] = &t;
        });

        // Create a local copy of parameters/gradients received from its child
        std::vector<resizable_tensor> gradients;
        gradients.resize(this->trainer->num_computational_layers);

        for (size_t i = 0; i < gradients.size(); i++)
        {
            gradients[i].copy_size(*tensors[i]);
        }

        // Endless async thread start.
        while (1)
        {
            this->job_signal_mutex[slave_index]->lock();
            if (this->signal_status[slave_index] != true)
            {
                std::cout << "[debug]Wait " << slave_index << std::endl;
                this->job_signal[slave_index]->wait();
                std::cout << "[debug]Resume " << slave_index << std::endl;
            }
            this->job_signal_mutex[slave_index]->unlock();

            if (this->slaves_status[slave_index] != slaveStatus::Running)
                break;

            this->signal_status[slave_index] = false;

            this->trainer->read_lock.lock();
            for (size_t layer = 0; layer < tensors.size(); layer++)
            {
                if (tensors[layer]->size() != 0)
                {
                    for (auto l = this->latest_paras[slave_index][layer].begin(), m = tensors[layer]->begin(); m != tensors[layer]->end(); l++, m++)
                    {
                        *l = *m;
                        // if (std::isnan(*m))
                        // 	std::cout << "wtf" << std::endl;
                    }
                }
            }
            this->trainer->read_lock.unlock();

            // this->send_lock->join_and_wait_till_my_turn(slave_index); // HPZ: New added send lock

            auto breakdown = system_clock::now();

            this->send_parameters(slave_index, this->latest_paras[slave_index]);

            // this->send_lock->release();

            std::cout << "(send from " << slave_index << " " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*

            this->wait_finishing(this->slaves_conns[slave_index]);

            // this->recv_lock->join_and_wait_till_my_turn(slave_index);
            std::cout << "(worker " << slave_index << " finished" << std::endl; // *_*

            this->notify_send_begin(this->slaves_conns[slave_index]);

            std::cout << "(worker " << slave_index << " start to send" << std::endl; // *_*

            breakdown = system_clock::now();

            this->receive_gradients_from_one(slave_index, gradients);

            // this->recv_lock->release();

            std::cout << "(recv from" << slave_index << " " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - breakdown).count() << std::endl; // *_*

            // std::cout << "Received from slave " << slave_index << std::endl;

            local_job_struct_to_store_tensor t(slave_index, 1, gradients);
            this->tq.add_task(t);
            // this->trainer->train_noop(); // HPZ: Very important

            this->counter[slave_index]++;

            // if (this->counter[slave_index] >= this->ending_time)
            // 	break;

            this->idle_worker[slave_index] = true;
        }
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    int dnn_full_leader<trainer_type, data_type, label_type>::receive_gradients_from_device(connection *src, std::vector<resizable_tensor> &cli_tensors)
    {
        try
        {
            for (size_t i = 0; i < cli_tensors.size(); i++)
            {
                if (cli_tensors[i].size() != 0)
                {
                    network::receive_compressed_tensor(src, &cli_tensors[i]);
                }
            }
        }
        catch (...)
        {
            std::cout << __FILE__ << ":" << __LINE__ << " Something went wrong here." << std::endl;
        }

        return 1;
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::send_parameters(int slave_index, std::vector<resizable_tensor> &tensors)
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

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::sync(dataset<matrix<unsigned char>, unsigned long> *testing)
    {
        unsigned long training_size = this->default_dataset->getData().size();
        if (this->me.sync_type == device_sync_type::sync)
        {

            /* Sync */

            while (1)
            {
                unsigned long current_start = 0;
                int epoch = 1, batch_amount = 0;

                // Wait for all requests
                task_op children_request_tasks[this->slaves_status.size()];
                for (int i = 0; i < this->slaves_status.size(); i++)
                {
                    children_request_tasks[i] = this->wait_for_task(this->slaves_conns[i]);

                    // Profile compuate ability for each child
                    if (children_request_tasks[i].reserved != this->slaves_list[i].comp_ability)
                    {
                        this->slaves_list[i].comp_ability = children_request_tasks[i].reserved;
                    }
                }

                // Reply requests
                int all_request_size = 0;
                for (int i = 0; i < sizeof(children_request_tasks) / sizeof(task_op); i++)
                {
                    task_op worker_job;
                    unsigned long start = current_start;
                    unsigned long end = (current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? training_size - 1 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability);
                    worker_job.opcode = 1;
                    std::memcpy(&worker_job.operand1, &start, sizeof(worker_job.operand1));
                    std::memcpy(&worker_job.operand2, &end, sizeof(worker_job.operand2));

                    if (this->verbose)
                        ::cout << worker_job.opcode << ":" << worker_job.operand1 << "~" << worker_job.operand2 << std::endl;
                    this->dispatch_jobs(i, worker_job);
                    this->job_signal_mutex[i]->lock();
                    this->signal_status[i] = true;
                    this->job_signal_mutex[i]->unlock();
                    this->job_signal[i]->signal();
                    this->idle_worker[i] = false;
                    batch_amount += 1;

                    current_start = ((current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? 0 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability));
                    if (current_start == 0)
                    {
                        epoch += 1;
                    }
                }

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
                    this->receive_latest_parameters(latest_parameters);
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
                    //this->dnn_leader::sync(); 	// This was supposed to point to sync_leader::sync which located at class dnn_leader, TODO
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
            int epoch = 1, old_epoch = 1, batch_amount = 0, batch_pos = 0;
            unsigned long current_start = 0;
            size_t last_updated_slave = -1;

            while (1)
            {
                if (epoch != old_epoch)
                {
                    // if (testing->accuracy(this->trainer->get_net()) >= 0.98)
                    if (epoch >= this->ending_time)
                    {
                        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!! epoch = " << epoch << std::endl;
                        std::cout << "Break!" << std::endl;
                        std::cout << batch_pos << "/" << batch_amount << std::endl;
                        break;
                    }
                    old_epoch = epoch;
                }

                // Check idle workers & dispatch jobs
                for (int i = 0; i < this->slaves_status.size(); i++)
                {
                    if (this->slaves_status[i] == slaveStatus::Running && this->idle_worker[i] == true)
                    {
                        task_op worker_job;
                        unsigned long start = current_start, end = (current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? training_size - 1 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability);
                        worker_job.opcode = 1;
                        std::memcpy(&worker_job.operand1, &start, sizeof(worker_job.operand1));
                        std::memcpy(&worker_job.operand2, &end, sizeof(worker_job.operand2));

                        // std::cout << worker_job.opcode << ":" << worker_job.operand1 << "~" << worker_job.operand2 << std::endl;
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
                        if (current_start == 0)
                        {
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
                    // std::cout << batch_pos << "/" << batch_amount << std::endl;
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
                    if ((*i).slave_index != last_updated_slave)
                    {

                        std::vector<std::vector<tensor *>> paras;
                        std::vector<tensor *> temp(this->trainer->num_computational_layers);

                        for (size_t j = 0; j < temp.size(); j++)
                        {
                            temp[j] = &((*i).tensors[j]);
                        }

                        std::vector<tensor *> old_tensors;
                        old_tensors.resize(this->trainer->num_computational_layers);

                        visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
                            old_tensors[i] = &t;
                        });

                        paras.push_back(old_tensors);
                        paras.push_back(temp);

                        this->trainer->read_lock.lock();

                        this->average_ptr(paras);

                        this->trainer->read_lock.unlock();
                        //sleep(10000);

                        while (this->tq.queue_lock.trylock() == 0)
                        {
                        };

                        this->tq.queue.erase(i);
                        batch_pos += 1;

                        this->tq.queue_lock.unlock();
                    }
                    else
                    {
                        std::cout << "Replaced !!!!!!" << std::endl;
                        std::vector<tensor *> temp(this->trainer->num_computational_layers);
                        for (size_t j = 0; j < temp.size(); j++)
                        {
                            temp[j] = &((*i).tensors[j]);
                        }
                        this->trainer->read_lock.lock();

                        this->update(temp);

                        this->trainer->read_lock.unlock();

                        while (this->tq.queue_lock.trylock() == 0)
                        {
                        };

                        this->tq.queue.erase(i);
                        batch_pos += 1;

                        this->tq.queue_lock.unlock();
                    }
                    last_updated_slave = (*i).slave_index;
                }
            }
        }

        for (size_t i = 0; i < this->receivers.size(); i++)
        {
            this->slaves_status[i] = slaveStatus::NotConn;
        }
        for (size_t i = 0; i < this->receivers.size(); i++)
        {
            std::cout << "[debug]Time to stop" << std::endl;
            this->job_signal_mutex[i]->lock();
            this->signal_status[i] = true;
            this->job_signal_mutex[i]->unlock();
            this->job_signal[i]->signal();
            std::cout << "[debug]Signaled" << std::endl;
            this->receivers[i]->join();
        }
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::init_thread_pool()
    {
        this->incoming_message = new local_msg_list();
        // Listener thread
        this->worker_thread_lock = new mutex();
        this->listener_thread_ptr = new std::thread(&dnn_full_leader::listener_thread, this);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::listener_thread()
    {
        std::cout << "Start a full leader listener thread ..." << std::endl;
        listener *lt;
        if (create_listener(lt, this->me.port, this->me.ip))
        {
            std::cerr << "Unable to create a listener" << std::endl;
        }

        while (true)
        {
            connection *src;

            lt->accept(src);
            std::thread *cur_thread = new std::thread(&dnn_full_leader::listener_worker_thread, this, src);
            this->worker_threads.push_back(cur_thread);

            this->worker_thread_lock->lock();
            for (auto t : this->worker_threads)
            {
                if (t->joinable())
                {
                    t->join();
                }
            }
            this->worker_thread_lock->unlock();
        }
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_full_leader<trainer_type, data_type, label_type>::listener_worker_thread(connection *conn)
    {
        std::cout << "Calling a full leader worker thread ..." << std::endl;
        network::msgheader req_header, res_header;
        task_op req_task, res_task;
        unsigned long start, end;

        res_header.dev_index = this->me.number;
        inet_pton(AF_INET, this->me.ip.c_str(), &(res_header.ip));
        res_header.port = this->me.port;

        network::recv_header(conn, &req_header);
        char src_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(req_header.ip), src_ip_str, INET_ADDRSTRLEN);
        std::cout << "Message from " << src_ip_str << ":" << req_header.port << std::endl;

        switch (req_header.type)
        {
        case task_type::train_one_batch:
            std::cout << "Dispatch a batch" << std::endl;
            network::recv_a_task(conn, &req_task);
            break;

        case task_type::request_one_batch:
            std::cout << "Request a batch" << std::endl;
            network::recv_a_task(conn, &req_task);

            res_header.length = sizeof(res_task);
            res_header.type = task_type::train_one_batch;
            // Get a task
            // unsigned long start = current_start, end = (current_start + BATCH_SIZE * this->slaves_list[i].comp_ability >= training_size - 1 ? training_size - 1 : current_start + BATCH_SIZE * this->slaves_list[i].comp_ability);
            res_task.opcode = task_type::train_one_batch;
            start = 0;
            end = 128;
            std::memcpy(&res_task.operand1, &start, sizeof(res_task.operand1));
            std::memcpy(&res_task.operand2, &end, sizeof(res_task.operand2));
            // Send task back
            network::send_header(conn, &res_header);
            network::send_a_task(conn, res_task);

            break;
        case task_type::request_updated_parameter:
            std::cout << "Request the updated parameter" << std::endl;

            res_header.type = task_type::response_most_updated_parameter;

            network::send_header(conn, &res_header);
            dnn_leader<trainer_type, data_type, label_type>::send_parameters(conn);

            break;
        case task_type::send_trained_parameter:
            std::cout << "Received a trained parameter" << std::endl;

            if (this->me.sync_type == device_sync_type::sync)
            {
                // SYNC
            }
            else
            {
                // ASYNC
                // Make a pointer point to the global parameter
                std::vector<tensor *> global_paras;
                global_paras.resize(this->trainer->num_computational_layers);
                visit_layer_parameters(this->trainer->devices[0]->net, [&](size_t i, tensor &t) {
                    global_paras[i] = &t;
                });

                // Create a data structure to store incoming parameters
                std::vector<resizable_tensor> incoming_paras;
                incoming_paras.resize(this->trainer->num_computational_layers);

                for (size_t i = 0; i < incoming_paras.size(); i++)
                {
                    incoming_paras[i].copy_size(*global_paras[i]);
                }

                // Receive after-trained parameter
                this->receive_gradients_from_device(conn, incoming_paras);

                std::vector<std::vector<tensor *>> paras;
                std::vector<tensor *> incoming_paras_ptr(this->trainer->num_computational_layers);

                for (size_t j = 0; j < incoming_paras_ptr.size(); j++)
                {
                    incoming_paras_ptr[j] = &incoming_paras[j];
                }


                paras.push_back(global_paras);
                paras.push_back(incoming_paras_ptr);

                this->trainer->read_lock.lock();

                this->average_ptr(paras);

                this->trainer->read_lock.unlock();
            }

            res_header.type = task_type::recv_trained_parameter;
            network::send_header(conn, &res_header);
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << " An unknown request header" << std::endl;
        }
    }

} // namespace dlib

#endif