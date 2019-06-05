#ifndef _DLIB_DNn_SYNCER_UTILS_H_
#define _DLIB_DNn_SYNCER_UTILS_H_

#include "../../sockets.h"
#include "../trainer.h"
#include <iostream>
#include <list>

#define COMP_BUFFER_SIZE 4096
#define SYNC_VERBOSE 0
#define NUM_DEBUG 0

namespace dlib
{

struct device
{
	int number = -1;
	std::string ip;
	int port = 2333;

	device() {}

	device(int number_, std::string ip_, int port_)
		: number(number_), ip(ip_), port(port_)
	{
		number = number_;
		ip = ip_;
		port = port_;
	}
}; // End of Structure device

enum slaveStatus
{
	FailVal = -2,
	NotConn = -1,
	Initlize = 0,
	Running = 1
}; // End of Enum slaveStatus

struct task
{
public:
	size_t slave_index = -1;
	volatile bool ready = 0;
	std::vector<resizable_tensor> tensors;

	task() = default;
	task &operator=(const task &) = default;
	task(size_t si_, bool ready_, std::vector<resizable_tensor> tensors_)
	{
		slave_index = si_;
		ready = ready_;
		tensors = tensors_;
	}

	~task() = default;
}; // End of class task

class task_queue
{
public:
	task_queue() = default;
	task_queue(const task_queue &) = delete;
	task_queue &operator=(const task_queue &) = delete;
	~task_queue() = default;

	void add_task(task t)
	{
		while (queue_lock.trylock() == 0)
			;

		queue.push_back(t);
		queue_lock.unlock();
	}

	task pop_task()
	{
		while (queue_lock.trylock() == 0)
			;

		task ret;

		while (queue.empty())
		{
			while (!queue.front().ready)
			{
				ret = queue.front();
			}
		}

		queue.front().~task();
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
	std::list<task> queue;
	mutex queue_lock;
};

enum task_type
{
	train_one_batch = 1,
	update_lr = 2,
	stop_train = 3
};

struct task_op
{
	int opcode = 0;
	int reserved = 0;
	double operand1 = 0;
	double operand2 = 0;
};

namespace network
{

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
		tmpPtr++;
	}

	network::send_ack(src, (char *)"got_comp_2");
	return length;
}

} // End of Namespace network

} // namespace dlib
#endif
