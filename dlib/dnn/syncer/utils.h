#ifndef _DLIB_DNn_SYNCER_UTILS_H_
#define _DLIB_DNn_SYNCER_UTILS_H_

#include <iostream>
#include "../trainer.h"
#include "../../sockets.h"

#define COMP_BUFFER_SIZE 4096
#define SYNC_VERBOSE 1
#define NUM_DEBUG 1

namespace dlib {

	struct device{
		int number = -1;
		std::string ip;
		int port = 2333;

		device() { }

		device(int number_, std::string ip_, int port_): number(number_), ip(ip_), port(port_) {
			number = number_;
			ip = ip_;
			port = port_;
		}
	}; // End of Structure device

	enum slaveStatus{
		FailVal		= -2,
		NotConn		= -1,
		Initlize	= 0,
		Running		= 1
	}; // End of Enum slaveStatus

	namespace network {

		// Wait for an acknowledgement from the connection
		void wait_ack(connection* src) {
			char tmpBuf[10] = {0};
			src->read(tmpBuf, 10);
			if (SYNC_VERBOSE)
				std::cout << "Ack:" << tmpBuf << std::endl;
		}

		// Send an acknowledgement from the connection
		void send_ack(connection* dest, char* content) {
			char tmpBuf[10] = {0};
			snprintf(tmpBuf, sizeof(tmpBuf), "%s", content);
			dest->write(tmpBuf, 10);
			if (SYNC_VERBOSE) {
				std::cout << "Send ack, content is:" << tmpBuf << std::endl;
			}
		}

		// Send a tensor to the connection 
		void send_tensor(connection* dest, tensor* tensor)
		{
			char tBuf[30] = {0};
			snprintf(tBuf, sizeof(tBuf), "%lu", tensor->size() * sizeof(*tensor->begin()) );
			std::cout << "tBuf:" << tBuf << std::endl;
			dest->write(tBuf, 30);

			char* tmpBuf = (char*) malloc(sizeof(float) * tensor->size());
			float* tmpPtr = (float*)tmpBuf;
			for(auto j = tensor->begin(); j != tensor->end(); j++)
			{
				*(tmpPtr++) = *j;
			}

			std::cout << dest->write(tmpBuf, sizeof(float) * tensor->size()) << std::endl;

			wait_ack(dest);
		}

		// Send a tensor to the connection in compression way
		// TODO: No compression implementation
		void send_compressed_tensor(connection* dest, tensor* tensor)
		{
			char tBuf[30] = {0};
			snprintf(tBuf, sizeof(tBuf), "%lu", tensor->size() * sizeof(*tensor->begin()) );
			std::cout << "tBuf:" << tBuf << std::endl;
			dest->write(tBuf, 30);

			char* tmpBuf = (char*) malloc(sizeof(float) * tensor->size());
			std::memset(tmpBuf, '\0', sizeof(float) * tensor->size());
			float* tmpPtr = (float*)tmpBuf;
			for(auto j = tensor->begin(); j != tensor->end(); j++)
			{
				*(tmpPtr++) = *j;
			}

			char* write_Ptr = tmpBuf;
			size_t write_length = 0;
			size_t write_max = sizeof(float) * tensor->size();
			size_t flag = 0;
			while(write_length + COMP_BUFFER_SIZE <= write_max) {
				int size = dest->write(write_Ptr, COMP_BUFFER_SIZE);

				if (NUM_DEBUG) {
					unsigned char fuck_num[COMP_BUFFER_SIZE] = {0};
					std::memcpy(fuck_num, write_Ptr, COMP_BUFFER_SIZE);
					std::cout << "send " << (++flag) << ": ";
					// for (auto i : fuck_num) {
					//     std::cout << (int) i << " ";
					// }
					std::cout << "[" << size << "]" << std::endl;
				}

				write_length += size;
				write_Ptr += size;
			}

			if (write_length < write_max) {
				dest->write(write_Ptr, write_max - write_length);
			}


			wait_ack(dest);

		}


	}

} // End of Namespace Dlib
#endif
