#ifndef DLIB_DNn_SYNCER_H_
#define DLIB_DNn_SYNCER_H_

#include "../trainer.h"
#include "../core.h"
#include "../solvers.h"
#include "../../statistics.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "../../serialize.h"

#include "../../pipe.h"
#include "../../threads.h"
#include "../../cuda/cuda_dlib.h"
#include "../../statistics/running_gradient.h"
#include <atomic>
#include <cstdio>
#include <set>
#include <future>
#include <exception>
#include <mutex>
#include "../../dir_nav.h"
#include "../../md5.h"

#include "../../sockets.h"
#include <bitset>
#include <thread>
#include <cstring>
#include "utils.h"
#include "utils_debug.h"


#define COMP_BUFFER_SIZE 4096
using std::chrono::system_clock;

namespace dlib{

	template <typename trainer_type>
	class dnn_syncer
	{
		public:
			trainer_type* trainer;

			int ismaster = 0;
			device me;
			device master;
			connection* master_conn = NULL;

			std::vector<device>			slaves_list;
			std::vector<connection*>	slaves_conns;
			std::vector<slaveStatus>	slaves_status;

			int verbose = 0;
			int num_debug = 1;
			int exper = 0;





			// Default sync initilization
			dnn_syncer();
			dnn_syncer(const dnn_syncer&);
			dnn_syncer& operator=(const dnn_syncer&);

			dnn_syncer(int ism){
				ismaster = ism;
			}

			dnn_syncer(trainer_type* trainer, int ism){
				this->trainer = trainer;
				this->ismaster = ism;
			}

			~dnn_syncer(){

			}

			void set_isMaster(int);
			void set_this_device(device);

			void add_slave(device);
			void remove_slave(size_t);

			void init_slaves();

			/*
			 *	Print out all slaves' status, including(ip, port, connection pointer and connection status)
			 *	void(*)
			 */
			void print_slaves_status(){
				DLIB_CASSERT(ismaster == 1, "Slave deivce doesn't have the right to get slaves' status.")

					for(int i = 0; i < this->slaves_list.size(); i++ ){
						std::cout << "[" << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << "]\t";
						std::cout << this->slaves_conns[i] << "\t" << this->slaves_status[i] << std::endl;
					}
			}


			int get_running_slaves_num(){
				int ret = 0;
				for(int i = 0; i < slaves_list.size(); i++){
					if(this->slaves_status[i] == 1)
						ret ++;
				}
				return ret;
			}

			int wait_for_master_init(){
				DLIB_CASSERT(ismaster == 0, "Master deivce doesn't need to wait for being initialized.");

				listener* lt;
				if(create_listener(lt, me.port, me.ip)){
					std::cerr << "Unable to create a listener" << std::endl;
					return 0;
				}
				connection* master_conn;
				if(!lt->accept(master_conn)){
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

			int dispatch_jobs(int slave_index, int begin, int end){
				DLIB_CASSERT(end > begin, "Job range is not valid.");

				if (this->slaves_status[slave_index] != slaveStatus::Running)
					return 0;

				return 1;
			}


			int notify_finish(){
				// TODO

				return 1;
			}


			void init_thread_pool()
			{

			}

			/************************************************************
			 * Do some statistics to tensosr (experiment used only)
			 *
			 ************************************************************/
			int stat_tensor(tensor* tensor)
			{
				int count = 0;
				for(auto k = tensor->begin(); k != tensor->end(); k ++){
					if (*k == 0 )
						count += 1;
				}
				return count;
			}

			/************************************************************
			 *	Print out the tensor abstract(size and first 10 number)
			 *
			 ************************************************************/
			void print_tensor(tensor* tensor, int size)
			{
				std::cout <<  "[" <<tensor->size() << "] ";
				for(auto k = tensor->begin(); k != tensor->end(); k ++){
					if(k == tensor->begin() + size)
						break;
					std::cout << *k << " ";
				}
				std::cout << std::endl;
			}

			/************************************************
			 *	Print a buffer of n size
			 ************************************************/
			void print_buffer(char* ptr, size_t n)
			{
				std::cout << "\n" << n << "---------------------------------\n";
				for(size_t i = 0; i < n; i++)
					std::cout << *(ptr + i);
				std::cout << "\n---------------------------------\n";
			}


			void wait_ack(connection* src)
			{
				char tmpBuf[10] = {0};
				src->read(tmpBuf, 10);
				if (this->verbose)
					std::cout << "Ack:" << tmpBuf << std::endl;
			}

			void send_ack(connection* dest, char* content)
			{
				char tmpBuf[10] = {0};
				snprintf(tmpBuf, sizeof(tmpBuf), "%s", content);
				dest->write(tmpBuf, 10);
				if (this->verbose)
					std::cout << "Send ack, content is:" << tmpBuf << std::endl;
			}

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

					if (this->num_debug) {
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


			void send_gradients_to_master(){

				std::vector<tensor*> tensors;
				tensors.resize(this->trainer->num_computational_layers);

				visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

				if (this->num_debug) {
					for (size_t i = 0; i < tensors.size(); i++) {
						if (tensors[i]->size() != 0)
							print_tensor(tensors[i], 10);
					}
				}

				for(size_t i = 0; i < tensors.size(); i++)
				{
					std::cout << i << " " << tensors[i]->size() << std::endl;
					if(tensors[i]->size() != 0)
					{
						send_compressed_tensor(master_conn, tensors[i]);
					}
				}

			}

			void send_parameters(connection* slave, std::vector<tensor*> parameters)
			{
				for(size_t i = 0; i < parameters.size(); i++)
				{
					if(parameters[i]->size() != 0)
					{
						// this->print_tensor(parameters[i], 10);
						send_compressed_tensor(slave, parameters[i]);
					}
				}

			}


			/******************************************************
			 *	Serialized send all tensors to all alive slaves
			 *
			 ******************************************************/
			void send_parameters_to_slaves_serialised(std::vector<tensor*> parameters)
			{
				if(get_running_slaves_num() != 0){

					for(int s_c = 0, s_c_max = slaves_status.size(); s_c < s_c_max ; s_c ++){
						if(slaves_status[s_c] == slaveStatus::Running){
							send_parameters(this->slaves_conns[s_c], parameters);
						}
					}
				}
			}


			/******************************************************
			 *	Paralized send all tensors to all alive slaves
			 *
			 ******************************************************/
			void send_parameters_to_slaves_paralized(std::vector<tensor*> parameters)
			{
				std::vector<std::thread *> recievers;
				recievers.resize(this->slaves_list.size());

				for(size_t i = 0; i< recievers.size(); i++)
				{
					if(slaves_status[i] == slaveStatus::Running)
						recievers[i] = new std::thread(&dnn_syncer::send_parameters, this, slaves_conns[i], parameters);
				}

				for(size_t i = 0; i < recievers.size(); i++)
				{
					recievers[i]->join();
				}
			}

			/********************************************************************************************************/

			int recieve_tensor(connection* src, tensor* container)
			{
				// auto epoch_time = system_clock::now();  // HPZ: Counting

				char sizeBuf[30] = {0};

				src->read(sizeBuf, 30);

				if (this->verbose)
					std::cout << sizeBuf << std::endl;
				size_t length = 0;
				try{
					length = atoi(sizeBuf);
					if(this->verbose)
						std::cout << "[!]Start recieving tensor, the size is " << length << std::endl;

				}catch(...){
					std::cerr << "incorrect with converting" << std::endl;
				}

				try {
					if (container->size() != (length / sizeof(*container->begin()))) {
						std::cerr << "The buffer is " << sizeBuf << ", which supposed to be " << container->size() << std::endl;
						std::cerr << "Recieving size is not same as container" << std::endl;
						sleep(100000);
					}
				} catch(...) {

				}

				float* tmpBuf = (float*) malloc( sizeof(float));
				*tmpBuf = 0;

				for(auto j = container->begin(); j != container->end(); j++){
					src->read((char*)tmpBuf, sizeof(float));
					*j = *(tmpBuf);
				}

				send_ack(src, (char*)"got");

				// std::cout << "(Time for bbbbbbbb) is " << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting //

				return length;
			}

			int recieve_compressed_tensor(connection* src, tensor* container)
			{
				char sizeBuf[30];
				src->read(sizeBuf, 30);
				std::cout << sizeBuf << std::endl;
				size_t length = 0;
				try
				{
					length = atoi(sizeBuf);
					if(this->verbose)
						std::cout << "[!]Start recieving tensor, the size is " << length << std::endl;
				} catch(...) {
					std::cerr << "incorrect with converting" << std::endl;
				}

				try {
					if (container->size() != (length / sizeof(*container->begin()))) {
						std::cerr << "The buffer is " << sizeBuf << ", which supposed to be " << container->size() << std::endl;
						std::cerr << "Recieving size is not same as container" << std::endl;
						// sleep(10000000);
					}
					if (length == 0) {
						std::cerr << "Length is invalid" << std::endl;
						return -1;
					}
				} catch(...) {

				}

				// Fix-size reading to the "deflated_buffer"
				char deflated_buffer[length];
				memset(deflated_buffer, '\0', length);
				char* deflated_ptr = &deflated_buffer[0];
				size_t read_length = length;
				size_t flag = 0;
				while (read_length > COMP_BUFFER_SIZE) {
					//					    std::cout << read_length << std::endl;
					int size = src->read(deflated_ptr, COMP_BUFFER_SIZE);

					if (this->num_debug) {
						if (size != COMP_BUFFER_SIZE) {
							unsigned char fuck_num[COMP_BUFFER_SIZE] = {0};
							std::memcpy(fuck_num, deflated_ptr, COMP_BUFFER_SIZE);
							std::cout << "Recv " << (++flag) << ": ";
							// for (auto i : fuck_num) {
							//     std::cout << (int) i << " ";
							// }
							std::cout << "[" << size << "]" << std::endl;
						}
					}

					deflated_ptr += size;
					read_length -= size;
				}
				if (read_length > 0) {
					src->read(deflated_ptr, read_length);
				}

				//TODO: Add deflation process

				float* tmpPtr = (float*)&deflated_buffer[0];
				for (auto j = container->begin(); j != container->end(); j++) {
					*j = *tmpPtr;
					tmpPtr ++;
				}

				send_ack(src, (char*)"got_comp_2");
				return length;
			}


			int recieve_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors){

				for(size_t i = 0; i < cli_tensors[slave_index].size(); i++)
				{
					if(cli_tensors[slave_index][i].size() != 0)
					{
						//							this->recieve_tensor(this->slave_conns[slave_index], &cli_tensors[slave_index][i]);
						this->recieve_compressed_tensor(this->slaves_conns[slave_index], &cli_tensors[slave_index][i]);

						// print_tensor(&cli_tensors[slave_index][i], cli_tensors[slave_index][i].size());
					}
				}
				return 1;
			}

			void init_before_recieving(std::vector<std::vector<resizable_tensor>> &all_tensors)
			{
				// Get the pointer of gradients from current device
				std::vector<tensor*> tensors;
				tensors.resize(this->trainer->num_computational_layers);
				visit_layer_parameters(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

				// Initialize temporary gradients contrainer from all other devices
				all_tensors.resize(slaves_status.size());
				for(size_t i = 0; i < all_tensors.size(); i++){
					all_tensors[i].resize(this->trainer->num_computational_layers);
					std::cout << "layers:" << this->trainer->num_computational_layers << std::endl;
					for(size_t j = 0; j < all_tensors[i].size(); j++){
						if(slaves_status[i] == slaveStatus::Running) {
							std::cout<<"layer size:" << tensors[j]->size() << std::endl;
							all_tensors[i][j].copy_size(*tensors[j]);
						}
					}
				}
			}

			void recieve_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors)
			{
				init_before_recieving(all_tensors);

				// Get gradients if there exists slave machine
				if(get_running_slaves_num() != 0)
				{
					for(int s_c = 0, s_c_max = slaves_status.size(); s_c < s_c_max ; s_c ++)
					{
						if(slaves_status[s_c] == slaveStatus::Running)
						{
							std::cout << "Reciveing from " << s_c << std::endl;
							recieve_gradients_from_one(s_c, all_tensors);
						}
					}
				}
			}


			void recieve_gradients_parallism(std::vector<std::vector<resizable_tensor>> &all_tensors)
			{
				init_before_recieving(all_tensors);
				std::vector<std::thread *> recievers;
				recievers.resize(all_tensors.size());

				for(size_t i = 0; i< recievers.size(); i++)
				{
					if(slaves_status[i] == slaveStatus::Running)
						recievers[i] = new std::thread(&dnn_syncer::recieve_gradients_from_one, this, i, std::ref(all_tensors));
				}

				for(size_t i = 0; i < recievers.size(); i++)
				{
					recievers[i]->join();
				}
			}


			void recieve_updated_parameters(std::vector<resizable_tensor> &updated)
			{
				// Initialize
				std::vector<tensor*> tensors;
				tensors.resize(this->trainer->num_computational_layers);
				visit_layer_parameters(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

				updated.resize(this->trainer->num_computational_layers);
				for(size_t i = 0; i < updated.size(); i++)
				{
					updated[i].copy_size(*tensors[i]);
				}


				for(size_t i = 0; i < updated.size(); i++)
				{
					if(updated[i].size() != 0)
						this->recieve_compressed_tensor(master_conn, &updated[i]);

					// this->print_tensor(&updated[i], 10);

				}
			}

			void average(std::vector<std::vector<resizable_tensor>> &all_tensors)
			{
				std::vector<std::vector<tensor*>> accessible_groups;
				float scale = 1.0 / all_tensors.size();
				for(size_t i = 0; i < this->trainer->num_computational_layers; i++)
				{
					std::vector<tensor*> group;
					for(size_t j = 0; j < all_tensors.size(); j++)
					{
						if(all_tensors[j][i].size() != 0 && this->slaves_status[j] == slaveStatus::Running)
						{
							group.push_back(&all_tensors[j][i]);
							// std::cout << &all_tensors[j][i] << std::endl;
						}
					}

					if(group.size() == 0)
						continue;

					if (group.size() == 1)
						tt::affine_transform(*group[0], *group[0], scale);
					else
						tt::affine_transform(*group[0], *group[0], *group[1], scale, scale);

					for (size_t i = 2; i < group.size(); ++i)
						tt::affine_transform(*group[0], *group[0], *group[i], 1, scale);
				}
			}

			void update(std::vector<tensor*>& updated){
				std::vector<tensor*> old_tensors;
				old_tensors.resize(this->trainer->num_computational_layers);

				visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){old_tensors[i] = &t;});
				// visit_layer_parameters(trainer->devices[0]->net, [&](size_t i, tensor& t){old_tensors[i] = &t;});

				for(size_t i = 0; i < old_tensors.size(); i++){
					if(old_tensors[i]->size() != 0){
						for(auto j = old_tensors[i]->begin(), k = updated[i]->begin(); j != old_tensors[i]->end(); j++, k++){
							*j = *k;
						}
					}
				}
			}

			void sn_sync_1(){

				// this->trainer->wait_for_thread_to_pause();

				if(ismaster){

					std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);

					std::vector<std::vector<resizable_tensor>> all_tensors;

					////////////////////////////////////////////////////////////
					auto epoch_time = system_clock::now();  // HPZ: Counting
					////////////////////////////////////////////////////////////

					// recieve_gradients_serialism(all_tensors);
					recieve_gradients_parallism(all_tensors);

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															//
						std::cout << "(Time for recieve_tensor) is "																							//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   				 	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					if (this->num_debug) {
						for(size_t i = 0; i < all_tensors.size(); i++){
							std::cout << "Gradient from slave " << i << std::endl;
							for(size_t j = 0; j < all_tensors[i].size(); j++){
								if(all_tensors[i][j].size() != 0){
									//									std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
									print_tensor(&all_tensors[i][j], 10);
								}
							}
						}
					}

					////////////////////////////////////////////////////
					epoch_time = system_clock::now();  // HPZ: Counting
					////////////////////////////////////////////////////

					average(all_tensors);

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																																//
						std::cout << "(Time for average) is "																										//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting   	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


					if (this->num_debug) {
						std::cout << "Averaged gradient: " << std::endl;
						for (size_t j = 0; j < all_tensors[0].size(); j++) {
							if (all_tensors[0][j].size() != 0) {
								//									std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
								print_tensor(&all_tensors[0][j], 10);
							}
						}
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					epoch_time = system_clock::now();  // HPZ: Counting																							  //
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					std::vector<tensor*> temp(this->trainer->num_computational_layers);
					for(size_t i = 0; i < temp.size(); i++)
					{
						// TODO : Deal with 0
						temp[i] = &all_tensors[0][i];
					}
					update(temp);



					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															//
						std::cout << "(Time for update) is "																								  	//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   					//
					epoch_time = system_clock::now();  // HPZ: Counting																						  	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					send_parameters_to_slaves_paralized(temp);

					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															 //
						std::cout << "(Time for syncback) is "																									 //
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   					 //
					epoch_time = system_clock::now();  // HPZ: Counting																							 //
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					if (this->num_debug) {
						for (size_t i = 0; i < all_tensors.size(); i++) {
							for (size_t j = 0; j < all_tensors[i].size(); j++) {
								//									 print_tensor(&all_tensors[i][j], 10);
							}
						}
					}
				}else{

					std::vector<resizable_tensor> updated;

					send_gradients_to_master();

					recieve_updated_parameters(updated);

					std::vector<tensor*> temp(this->trainer->num_computational_layers);
					for(size_t i = 0; i < temp.size(); i++)
					{
						// TODO : Deal with 0
						temp[i] = &updated[i];

						if (this->num_debug) {
							if (temp[i]->size() != 0)
								print_tensor(temp[i], 10);
						}
					}
					update(temp);
				}
				std::cout << "Sync finished" << std::endl;
				//					sleep(1000);
			}

			void sn_sync(){
				if(ismaster){

					std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);

					std::vector<std::vector<resizable_tensor>> all_tensors;

					////////////////////////////////////////////////////////////
					auto epoch_time = system_clock::now();  // HPZ: Counting
					////////////////////////////////////////////////////////////

					// recieve_gradients_serialism(all_tensors);
					recieve_gradients_parallism(all_tensors);

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															//
						std::cout << "(Time for recieve_tensor) is "																							//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   				 	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					if (this->num_debug) {
						for(size_t i = 0; i < all_tensors.size(); i++){
							std::cout << "Gradient from slave " << i << std::endl;
							for(size_t j = 0; j < all_tensors[i].size(); j++){
								if(all_tensors[i][j].size() != 0){
									//	std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
									print_tensor(&all_tensors[i][j], 10);
								}
							}
						}
					}

					////////////////////////////////////////////////////
					epoch_time = system_clock::now();  // HPZ: Counting
					////////////////////////////////////////////////////

					average(all_tensors);

					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																																//
						std::cout << "(Time for average) is "																										//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting   	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


					if (this->num_debug) {
						std::cout << "Averaged gradient: " << std::endl;
						for (size_t j = 0; j < all_tensors[0].size(); j++) {
							if (all_tensors[0][j].size() != 0) {
								//									std::cout << "---(" << i << "," << j << "):\t" << stat_tensor(&all_tensors[i][j]) << std::endl;
								print_tensor(&all_tensors[0][j], 10);
							}
						}
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					epoch_time = system_clock::now();  // HPZ: Counting																							  //
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					std::vector<tensor*> temp(this->trainer->num_computational_layers);
					for(size_t i = 0; i < temp.size(); i++)
					{
						// TODO : Deal with 0
						temp[i] = &all_tensors[0][i];
					}



					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															//
						std::cout << "(Time for update) is "																								  	//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   					//
					epoch_time = system_clock::now();  // HPZ: Counting																						  	//
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					send_parameters_to_slaves_paralized(temp);

					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (this->exper)																															 //
						std::cout << "(Time for syncback) is "																									 //
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   					 //
					epoch_time = system_clock::now();  // HPZ: Counting																							 //
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					if (this->num_debug) {
						for (size_t i = 0; i < all_tensors.size(); i++) {
							for (size_t j = 0; j < all_tensors[i].size(); j++) {
								//									 print_tensor(&all_tensors[i][j], 10);
							}
						}
					}
				}else{

					std::vector<resizable_tensor> updated;

					send_gradients_to_master();

					recieve_updated_parameters(updated);

					std::vector<tensor*> temp(this->trainer->num_computational_layers);
					for(size_t i = 0; i < temp.size(); i++)
					{
						// TODO : Deal with 0
						temp[i] = &updated[i];

						if (this->num_debug) {
							if (temp[i]->size() != 0)
								print_tensor(temp[i], 10);
						}
					}
					update(temp);
				}
				std::cout << "Sync finished" << std::endl;
				//					sleep(1000);
			}


			// TODO
			dnn_syncer& operator<< (std::ostream& out){
				out << trainer << std::endl;
				out << ismaster << std::endl;
				if(ismaster)
					out << slaves_list.size() << std::endl;

			}

	};

}







#endif
