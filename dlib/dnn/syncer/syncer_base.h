#ifndef DLIB_DNn_SYNCER_BASE_H_
#define DLIB_DNn_SYNCER_BASE_H_

#include "syncer.h"
namespace dlib
{
    template <typename trainer_type,
              typename data_type,
              typename label_type>
    dnn_syncer<trainer_type, data_type, label_type>::dnn_syncer(device_role role)
    {
        this->role = role;
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    dnn_syncer<trainer_type, data_type, label_type>::dnn_syncer(trainer_type *trainer, device_role role)
    {
        this->trainer = trainer;
        this->role = role;
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::set_role(device_role role)
    {
        this->role = role;
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::set_this_device(device me_)
    {
        this->me = me_;
        this->logfile = new ofstream(std::to_string(this->me.number) + ".log");
        this->logger = new logbot(this->logfile);
        // this->listener_thread_ptr = new std::thread(&dnn_syncer::listen_thread, this);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::set_master_device(device master_)
    {
        this->master = master_;
        // this->listener_thread_ptr = new std::thread(&dnn_syncer::listen_thread, this);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    int dnn_syncer<trainer_type, data_type, label_type>::wait_for_master_init()
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

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::init_thread_pool()
    {
        this->incoming_message = new local_msg_list();
        // Listener thread
        this->worker_thread_lock = new mutex();
        this->listener_thread_ptr = new std::thread(&dnn_syncer::listener_thread, this);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::listener_thread()
    {
        std::cout << "Start a syncer listener thread ..." << std::endl;
        listener *lt;
        if (create_listener(lt, me.port, me.ip))
        {
            std::cerr << "Unable to create a listener" << std::endl;
        }

        while (true)
        {
            connection *src;

            lt->accept(src);
            std::thread *cur_thread = new std::thread(&dnn_syncer::listener_worker_thread, this, src);
            this->worker_threads.push_back(cur_thread);
            for (auto t : this->worker_threads)
            {
                if (t->joinable())
                {
                    t->join();
                }
            }
        }
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::listener_worker_thread(connection *conn)
    {
        network::msgheader *header = new network::msgheader();
        network::recv_header(conn, header);

        char src_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(header->ip), src_ip_str, INET_ADDRSTRLEN);

        std::cout << "Message from " << src_ip_str << ":" << header->port << std::endl;
        task_op *task = new task_op();
        switch (header->type)
        {
        case task_type::train_one_batch:
            std::cout << "Dispatch a batch" << std::endl;
            network::recv_a_task(conn, task);
            break;
        case task_type::send_trained_parameter:
            std::cout << "Updated global parameters" << std::endl;
            break;
        case task_type::request_one_batch:
            std::cout << "Request a batch" << std::endl;
            network::recv_a_task(conn, task);
            break;
        }
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::average(std::vector<std::vector<resizable_tensor>> &all_tensors)
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

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::average_ptr(std::vector<std::vector<tensor *>> &all_tensors)
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

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::update(std::vector<tensor *> &updated)
    {
        std::vector<tensor *> old_tensors;
        old_tensors.resize(this->trainer->num_computational_layers);

        visit_layer_parameters(trainer->devices[0]->net, [&](size_t i, tensor &t) {
            old_tensors[i] = &t;
        });

        for (size_t i = 0; i < old_tensors.size(); i++)
        {
            if (old_tensors[i]->size() != 0)
            {
                for (auto j = old_tensors[i]->begin(), k = updated[i]->begin(); j != old_tensors[i]->end(); j++, k++)
                {
                    *j = *k;
                }
            }
        }
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    task_op dnn_syncer<trainer_type, data_type, label_type>::request_a_task(task_op req)
    {
        return request_a_task(NULL, req);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::request_a_task(connection *src_conn, task_op req)
    {
        if (src_conn == NULL)
            network::send_a_task(this->master_conn, req);
        else
            network::send_a_task(src_conn, req);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    task_op dnn_syncer<trainer_type, data_type, label_type>::wait_for_task()
    {
        return wait_for_task(NULL);
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    task_op dnn_syncer<trainer_type, data_type, label_type>::wait_for_task(connection *src_conn)
    {
        task_op ret;
        if (src_conn == NULL)
            network::recv_a_task(this->master_conn, &ret);
        else
            network::recv_a_task(src_conn, &ret);

        return ret;
    }

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::receive_latest_parameters(connection* src, std::vector<resizable_tensor> &updated)
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
            if (updated[i].size() != 0) {
                for(int tt = 0; tt < TRANS_TIME; tt++)
                    network::receive_compressed_tensor(src, &updated[i]);
            }
        }

        char tmp = ' ';
		src->write(&tmp, 1);
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::notify_train_finish()
    {
        char *msg = new char[1];
        this->master_conn->write(msg, 1);
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::notify_send_begin(connection *conn)
    {
        char *msg = new char[1];
        conn->write(msg, 1);
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::wait_finishing(connection *conn)
    {
        char *msg = new char[1];
        conn->read(msg, 1);
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::wait_to_send()
    {
        char *msg = new char[1];
        this->master_conn->read(msg, 1);
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::init_trainer() // Initial trainer before start to request tasks, runs once when at beginning of the bootup.
    {
        this->init_trainer(*(this->default_dataset));
    };

    template <typename trainer_type,
              typename data_type,
              typename label_type>
    void dnn_syncer<trainer_type, data_type, label_type>::init_trainer(dataset<data_type, label_type> training) // Initial trainer before start to request tasks, runs once when at beginning of the bootup.
    {
        while (trainer->ready_status < 1) // Wait for trainer get ready.
        {
        };

        this->trainer->train_one_batch(training.getData(), training.getLabel()); // Give trainer one batch for allocate space of parameters and gradients

        this->trainer->distributed_signal.get_mutex().lock(); // Notify trainer start to train
        this->trainer->ready_status = 2;
        this->trainer->status_lock.unlock();
        this->trainer->distributed_signal.signal();

        while (trainer->ready_status < 4) // Wait until trainer finished training
        {
        };
    };

    // TODO
    template <typename trainer_type,
              typename data_type,
              typename label_type>
    std::ostream& dnn_syncer<trainer_type, data_type, label_type>::operator<<(std::ostream &out)
    {
        out << trainer << std::endl;
        out << static_cast<int>(role) << std::endl;
        return out;
    }

} // namespace dlib

#endif