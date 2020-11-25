#ifndef _DLIB_DNn_SYNCER_UTILS_H_
#define _DLIB_DNn_SYNCER_UTILS_H_

#include "../../sockets.h"
#include "../trainer.h"
#include <iostream>
#include <list>
#include <mutex>
#include <arpa/inet.h>

#define COMP_BUFFER_SIZE 4096
#define SYNC_VERBOSE 0
#define NUM_DEBUG 0

namespace dlib
{
	struct task_op;
	struct msgheader;


	enum device_role
	{
		worker = 0,
		leader = 1,
		supleader = 2,
		undecided = -1
	};

	enum device_sync_type
	{
		sync = 0,
		async = 1
	};

	struct device
	{
		int number = -1;
		std::string ip;
		int port = 2333;
		int role = device_role::undecided;
		int master = -1;
		int comp_ability = 1;
		device_sync_type sync_type;

		device() {}

		device(int number_, std::string ip_, int port_)
			: number(number_), ip(ip_), port(port_)
		{
		}
		device(int number_, std::string ip_, int port_, int role_, int master_, int comp_, device_sync_type stype_)
			: number(number_), ip(ip_), port(port_), role(role_), master(master_), comp_ability(comp_), sync_type(stype_)
		{
		}
	}; // End of Structure device

	enum slaveStatus
	{
		FailVal = -2,
		NotConn = -1,
		Initialize = 0,
		Running = 1
	}; // End of Enum slaveStatus

	//////////////////////////
	//						//
	//	Local_job_queue		//
	//						//
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	struct local_job_struct_to_store_tensor
	{
		/*
		*	Local_job_struct_to_store_tensor is a the element to be stored in the following "queue" class.
		*	
		*	This structure is only being used for async-enabled node, stored only in local.
		*	Please differentiate between this structure and task_op (a operation message strucure for network
		*	transimission usage only)
		*/
	public:
		size_t slave_index = -1;
		volatile bool ready = 0;
		std::vector<resizable_tensor> tensors;

		local_job_struct_to_store_tensor() = default;
		local_job_struct_to_store_tensor &operator=(const local_job_struct_to_store_tensor &) = default;
		local_job_struct_to_store_tensor(size_t si_, bool ready_, std::vector<resizable_tensor> tensors_)
		{
			slave_index = si_;
			ready = ready_;
			tensors = tensors_;
		}

		~local_job_struct_to_store_tensor() = default;
	}; // End of class task

	class local_job_fffo_queue
	{
		/*
		*	Local_job_fffo_queue is a local first-finished-first-out queue for storing assigned jobs' expected response.
		*	
		*	Since async is runnable now, when async thread received a updated parameter or gradient, then it will store
		*	received data into the corresponded slot in this queue. Main thread will busy checking all the slot in this 
		*	queue. Once it find a finished job, it will do average or further operations.
		*	Honestly, I believe there are some problem within this queue's implementation, I will check this late for 
		*	avoiding undone job jam the whole training process.
		*/
	public:
		local_job_fffo_queue() = default;
		local_job_fffo_queue(const local_job_fffo_queue &) = delete;
		local_job_fffo_queue &operator=(const local_job_fffo_queue &) = delete;
		~local_job_fffo_queue() = default;

		void add_task(local_job_struct_to_store_tensor t)
		{
			while (queue_lock.trylock() == 0)
				;

			queue.push_back(t);
			queue_lock.unlock();
		}

		local_job_struct_to_store_tensor pop_task()
		{
			while (queue_lock.trylock() == 0)
				;

			local_job_struct_to_store_tensor ret;

			while (queue.empty())
			{
				while (!queue.front().ready)
				{
					ret = queue.front();
				}
			}

			queue.front().~local_job_struct_to_store_tensor();
			queue.pop_front();
			queue_lock.unlock();
			return ret;
		}

		bool empty()
		{
			while (queue_lock.trylock() == 0)
				;

			queue_lock.unlock();
			return false;
		}

		// private:
		std::list<local_job_struct_to_store_tensor> queue;
		mutex queue_lock;
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct message_elem
	{
		/*
		*	leaders_local_primary_job is a the element to be stored in the following "queue" class.
		*	
		*	This structure is only being used for async-enabled node, stored only in local.
		*	Please differentiate between this structure and task_op (a operation message strucure for network
		*	transimission usage only)
		*/
	public:
		msgheader *header; 
		task_op *request_task;
		std::vector<resizable_tensor> *tensors;

		message_elem() = default;
		message_elem &operator=(const message_elem &) = default;
		message_elem(msgheader *header_,  task_op *request_task_, std::vector<resizable_tensor> *tensors_)
		{
			header = header_;
			request_task = request_task_;
			tensors = tensors_;
		}

		~message_elem() = default;
	}; // End of class task

	class local_msg_list
	{
		/*
		*	Local_job_fifo_queue is a local FIFO queue for storing todo jobs.
		*	
		*
		*/
	public:
		local_msg_list() = default;
		local_msg_list(const local_msg_list &) = delete;
		local_msg_list &operator=(const local_msg_list &) = delete;
		~local_msg_list() = default;

		void push_back(message_elem t)
		{
			lock();
			queue.push_back(t);
			unlock();
		}

		void pop_front()
		{
			lock();
			if (!queue.empty())
			{
				queue.front().~message_elem();
				queue.pop_front();
			}
			unlock();
		}

		message_elem* front()
		{
			lock();

			if (!queue.empty())
			{
				unlock();
				return &(queue.front());
			}
			unlock();
			return NULL;
		}

		void lock()
		{
			while (queue_lock.trylock() == 0)
				;
		}

		void unlock()
		{
			queue_lock.unlock();
		}

	private:
		std::list<message_elem> queue;
		mutex queue_lock;
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class queue_lock
	{
	public:
		queue_lock(int _max_exec)
		{
			this->max_execution = _max_exec;
			this->current_execution = 0;
		}

		bool join_and_wait_till_my_turn(int number)
		{
			lock.lock();
			this->exec_queue.push(number);
			lock.unlock();

			while (1)
			{
				lock.lock();
				if (current_execution >= max_execution)
				{
					lock.unlock();
					continue;
				}
				if (this->exec_queue.front() != number)
				{
					lock.unlock();
				}
				else
					break;
			}

			this->current_execution += 1;
			this->exec_queue.pop();
			lock.unlock();

			return true;
		}

		bool release()
		{
			lock.lock();
			this->current_execution -= 1;
			lock.unlock();

			return true;
		}

	private:
		volatile int max_execution;
		volatile int current_execution;
		std::queue<int> exec_queue;
		const mutex lock;
	};

	

	//////////////////////////
	//						//
	//	Local_job_queue		//
	//						//
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum task_type
	{
		train_one_batch = 1,
		update_lr = 2,
		stop_train = 3,
		request_one_batch = 4
	};

	struct task_op
	{
		int opcode = 0;
		int reserved = 0;
		double operand1 = 0;
		double operand2 = 0;
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	namespace network
	{
		struct msgheader 
		{
			int dev_index;
			in_addr_t ip;
			int port;
			int type;
			int length;
			int reserve;
		};

		void send_header(connection *dst, msgheader* header)
		{
			if (sizeof(header) != 24)
			{
				std::cout << "The task is not 24 bytes, it is " << sizeof(header) << std::endl;
			}
			dst->write((char *)&header->dev_index, 24);
			dst->read(NULL, 1);
		}

		void recv_header(connection *src, msgheader* header)
		{
			char *headerbuffer = new char [24];
			src->read(headerbuffer, 24);
			header->dev_index = *((int *)headerbuffer);
			header->ip = *((in_addr_t *)(headerbuffer + 4));
			header->port = *((int *)(headerbuffer + 8));
			header->type = *((int *)(headerbuffer + 12));
			header->length = *((int *)(headerbuffer + 16));
			header->reserve = *((int *)(headerbuffer + 20));
			src->write(" ", 1);

			delete[] headerbuffer;
		}

		int send_a_task(connection *dst, task_op task)
		{
			if (sizeof(task) != 24)
			{
				std::cout << "The task is not 24 bytes, it is " << sizeof(task) << std::endl;
				return 0;
			}
			dst->write((char *)&task.opcode, 24);
			return 1;
		}

		void recv_a_task(connection *src, task_op *task)
		{
			char *task_message = new char[24];

			src->read(task_message, 24);
			task->opcode = *((int *)task_message);
			task->reserved = *((int *)(task_message + 4));
			task->operand1 = *((double *)(task_message + 8));
			task->operand2 = *((double *)(task_message + 16));

			delete[] task_message;
		}

		connection* create_message_session(std::string dst_ip, int port, std::string src_ip)
		{
			connection *dialog_session;
			if (create_connection(dialog_session, port, dst_ip, (unsigned short)0, src_ip))
			{
				std::cerr << "Create failed on " << dst_ip << ":" << port << std::endl;
				return NULL;
			}
			return dialog_session;
		}

		void halt_message_session(connection* conn)
		{
			close_gracefully(conn, 500);
		}

		// Wait for an acknowledgement from the connection
		void wait_ack(connection *src)
		{
			char tmpBuf[15] = {0};
			src->read(tmpBuf, 15);

			if (SYNC_VERBOSE)
				std::cout << "Ack:" << tmpBuf << std::endl;
		}

		// Send an acknowledgement from the connection
		void send_ack(connection *dest, char *content)
		{
			char tmpBuf[15] = {0};
			snprintf(tmpBuf, sizeof(tmpBuf), "%s", content);
			dest->write(tmpBuf, 15);

			if (SYNC_VERBOSE)
			{
				std::cout << "Send ack, content is:" << tmpBuf << std::endl;
			}
		}

		// Send a tensor to the connection
		void send_tensor(connection *dest, tensor *tensor)
		{
			char tBuf[30] = {0};
			snprintf(tBuf, sizeof(tBuf), "%lu",
					 tensor->size() * sizeof(*tensor->begin()));
			dest->write(tBuf, 30);

			char *tmpBuf = (char *)malloc(sizeof(float) * tensor->size());
			float *tmpPtr = (float *)tmpBuf;

			for (auto j = tensor->begin(); j != tensor->end(); j++)
			{
				*(tmpPtr++) = *j;
			}

			std::cout << dest->write(tmpBuf, sizeof(float) * tensor->size()) << std::endl;

			wait_ack(dest);
		}

		// Send a tensor to the connection in compression way
		// TODO: No compression implementation
		void send_compressed_tensor(connection *dest, tensor *tensor)
		{
			char tBuf[30] = {0};
			snprintf(tBuf, sizeof(tBuf), "%lu",
					 tensor->size() * sizeof(*tensor->begin()));
			dest->write(tBuf, 30);

			char *tmpBuf = (char *)malloc(sizeof(float) * tensor->size());
			std::memset(tmpBuf, '\0', sizeof(float) * tensor->size());
			float *tmpPtr = (float *)tmpBuf;

			for (auto j = tensor->begin(); j != tensor->end(); j++)
			{
				*(tmpPtr++) = *j;
				if (std::isnan(*j))
					std::cout << "wtf" << std::endl;
			}

			char *write_Ptr = tmpBuf;
			size_t write_length = 0;
			size_t write_max = sizeof(float) * tensor->size();

			while (write_length + COMP_BUFFER_SIZE <= write_max)
			{
				int size = dest->write(write_Ptr, COMP_BUFFER_SIZE);

				if (NUM_DEBUG)
				{
					unsigned char fuck_num[COMP_BUFFER_SIZE] = {0};
					std::memcpy(fuck_num, write_Ptr, COMP_BUFFER_SIZE);
					// std::cout << "send " << (++flag) << ": ";
					// for (auto i : fuck_num) {
					//     std::cout << (int) i << " ";
					// }
					// std::cout << "[" << size << "]" << std::endl;
				}

				write_length += size;
				write_Ptr += size;
			}

			if (write_length < write_max)
			{
				dest->write(write_Ptr, write_max - write_length);
			}

			free(tmpBuf);
			wait_ack(dest);
		}

		int receive_tensor(connection *src, tensor *container)
		{
			// auto epoch_time = system_clock::now();  // HPZ: Counting

			char sizeBuf[30] = {0};

			src->read(sizeBuf, 30);

			if (SYNC_VERBOSE)
				std::cout << sizeBuf << std::endl;

			size_t length = 0;

			try
			{
				length = atoi(sizeBuf);

				if (SYNC_VERBOSE)
					std::cout << "[!]Start receiving tensor, the size is " << length
							  << std::endl;
			}
			catch (...)
			{
				std::cerr << "incorrect with converting" << std::endl;
			}

			try
			{
				if (container->size() != (length / sizeof(*container->begin())))
				{
					std::cerr << "The buffer is " << sizeBuf << ", which supposed to be "
							  << container->size() << std::endl;
					std::cerr << "Receiving size is not same as container" << std::endl;
					sleep(100000);
				}
			}
			catch (...)
			{
			}

			float *tmpBuf = (float *)malloc(sizeof(float));
			*tmpBuf = 0;

			for (auto j = container->begin(); j != container->end(); j++)
			{
				src->read((char *)tmpBuf, sizeof(float));
				*j = *(tmpBuf);
			}

			network::send_ack(src, (char *)"got");

			// std::cout << "(Time for bbbbbbbb) is " <<
			// std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() -
			// epoch_time).count() << std::endl;   // HPZ: Counting //

			free(tmpBuf);
			return length;
		}

		int receive_compressed_tensor(connection *src, tensor *container)
		{
			char sizeBuf[30];
			src->read(sizeBuf, 30);
			//std::cout << sizeBuf << std::endl;
			size_t length = 0;

			try
			{
				length = atoi(sizeBuf);

				if (SYNC_VERBOSE)
					std::cout << "[!]Start receiving tensor, the size is " << length
							  << std::endl;
			}
			catch (...)
			{
				std::cerr << "incorrect with converting" << std::endl;
			}

			try
			{
				if (container->size() != (length / sizeof(*container->begin())))
				{
					std::cerr << "The buffer is " << sizeBuf << ", which supposed to be "
							  << container->size() << std::endl;
					std::cerr << "Receiving size is not same as container" << std::endl;
					// sleep(10000000);
				}

				if (length == 0)
				{
					std::cerr << "Length is invalid" << std::endl;
					return -1;
				}
			}
			catch (...)
			{
			}

			// Fix-size reading to the "deflated_buffer"
			char deflated_buffer[length];
			memset(deflated_buffer, '\0', length);
			char *deflated_ptr = &deflated_buffer[0];
			size_t read_length = length;

			int size = 0;

			while (read_length > 0)
			{
				size = src->read(deflated_ptr, (read_length < COMP_BUFFER_SIZE) ? read_length : COMP_BUFFER_SIZE);

				if (NUM_DEBUG)
				{
					unsigned char fuck_num[COMP_BUFFER_SIZE] = {0};
					std::memcpy(fuck_num, deflated_ptr, size);

					std::cout << "====>" << size << "<====";
					for (auto i : fuck_num)
					{
						std::cout << (int)i << " ";
					}
					std::cout << "\n\n\n";
				}

				deflated_ptr += size;
				read_length -= size;
			}

			// TODO: Add deflation process

			float *tmpPtr = (float *)&deflated_buffer[0];

			for (auto j = container->begin(); j != container->end(); j++)
			{
				*j = *tmpPtr;
				if (std::isnan(*j))
					std::cout << "wtf" << std::endl;
				tmpPtr++;
			}

			network::send_ack(src, (char *)"got_comp_2");
			return length;
		}

	} // End of Namespace network

} // namespace dlib
#endif
