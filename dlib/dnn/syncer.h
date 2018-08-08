#ifndef DLIB_DNn_SYNCER_H_
#define DLIB_DNn_SYNCER_H_

#include "trainer.h"
#include "core.h"
#include "solvers.h"
#include "../statistics.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "../serialize.h"

#include "../pipe.h"
#include "../threads.h"
#include "../cuda/cuda_dlib.h"
#include "../statistics/running_gradient.h"
#include <atomic>
#include <cstdio>
#include <set>
#include <future>
#include <exception>
#include <mutex>
#include "../dir_nav.h"
#include "../md5.h"

#include "../sockets.h"
#include <bitset>

using std::chrono::system_clock;
namespace dlib{

	struct device{
		int number = -1;
		std::string ip;
		int port = 2333;

		device(){
		}

		device(int number_, std::string ip_, int port_): number(number_), ip(ip_), port(port_){
			number = number_;
			ip = ip_;
			port = port_;			
		}
	};

	enum slaveStatus{
		FailVal		= -2,
		NotConn		= -1,
		Initlize	= 0,
		Running		= 1
	};

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
				std::vector<connection*>	slave_conns;
				std::vector<slaveStatus>	slave_status;
				
				int verbose = 1;





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

				void set_isMaster(int ism){
					this->ismaster = ism;
				}

				void set_this_device(device me_){
					this->me = me_;
				}

				void add_slave(device slave){
					DLIB_CASSERT(ismaster == 1, "Slave deivce doesn't have the right to add slaves.")

						this->slaves_list.push_back(slave);
					connection* c;
					this->slave_conns.push_back(c);
					this->slave_status.push_back(slaveStatus::Initlize);

				}

				void init_slaves(){
					DLIB_CASSERT(ismaster == 1, "Slave deivce doesn't have the right to init slaves.")

						for(int i = 0; i < this->slaves_list.size(); i++ ){
							connection* conn = NULL;
							if(create_connection(conn, 
										(unsigned short)slaves_list[i].port	, slaves_list[i].ip, 
										(unsigned short)me.port + i			, me.ip)
							  ){
								std::cerr << "Create failed on " << slaves_list[i].ip << ":" << slaves_list[i].port << std::endl;
								this->slave_status[i] = slaveStatus::NotConn;
								if(conn)
									conn->shutdown();
								continue;
							}

							this->slave_conns[i] = conn;

							// Greeting messages
							char init_msg[30];
							snprintf(init_msg, sizeof(init_msg), "%s:%d\n", &me.ip[0], me.port);
							conn->write(init_msg, 30);
							char reply_msg[30];
							conn->read(reply_msg, 30);

							// Validating if slave ip and port is correct
							char* cptr = strchr(reply_msg, ':');
							char* eptr = strchr(reply_msg, '\n');
							if(!(std::string(reply_msg, cptr - reply_msg)==slaves_list[i].ip &&
										atoi(std::string(cptr + 1, eptr - cptr + 1).c_str()) == slaves_list[i].port)){
								std::cerr << "Error in validating slaves" << std::endl;

								slave_status[i] = slaveStatus::FailVal;
								continue;
							}

							this->slave_status[i] = slaveStatus::Running;

						}
				}

				void get_slaves_status(){
					DLIB_CASSERT(ismaster == 1, "Slave deivce doesn't have the right to get slaves' status.")

						for(int i = 0; i < this->slaves_list.size(); i++ ){
							std::cout << "[" << this->slaves_list[i].ip << ":" << this->slaves_list[i].port << "]\t";
							std::cout << this->slave_conns[i] << "\t" << this->slave_status[i] << std::endl;
						}
				}


				int get_running_slaves_num(){
					int ret = 0;
					for(int i = 0; i < slaves_list.size(); i++){
						if(this->slave_status[i] == 1)
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

					if (this->slave_status[slave_index] != slaveStatus::Running)
						return 0;

					return 1;
				}


				int notify_finish(){
					// TODO

					return 1;		
				}

				/************************************************************
				 *	Print out the tensor abstract(size and first 10 number)
				 *
				 ************************************************************/
				void print_tensor(tensor* tensor, int size)
				{
					std::cout <<  "[" <<tensor->size() << "]";
					for(auto k = tensor->begin(); k != tensor->end(); k ++){
						if(k == tensor->begin() + size)
							break;
						std::cout << *k << " ";
					}
					std::cout << std::endl;
				}


				void wait_ack(connection* src)
				{
					char tmpBuf[10];
					src->read(tmpBuf, 10);
					std::cout << "Ack:" << tmpBuf << std::endl;
				}

				void send_ack(connection* dest, char* content)
				{
					char tmpBuf[10];
					snprintf(tmpBuf, sizeof(tmpBuf), "%s", content);
					dest->write(tmpBuf, 10);
					std::cout << "Send ack, content is:" << tmpBuf << std::endl;
				}

				void send_tensor(connection* dest, tensor* tensor)
				{
					char tBuf[30];
					snprintf(tBuf, sizeof(tBuf), "%lu", tensor->size());
					dest->write(tBuf, 30);

					char* tmpBuf = (char*) malloc(sizeof(float) * tensor->size());
					float* tmpPtr = (float*)tmpBuf;
					for(auto j = tensor->begin(); j != tensor->end(); j++)
					{
						*(tmpPtr++) = *j;
					}

					dest->write(tmpBuf, sizeof(float) * tensor->size());

					wait_ack(dest);
				}


				void send_gradients_to_master(){

					std::vector<tensor*> tensors;
					tensors.resize(this->trainer->num_computational_layers);

					visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

					for(size_t i = 0; i < tensors.size(); i++)
					{
						print_tensor(tensors[i], tensors[i]->size());
					}

					for(size_t i = 0; i < tensors.size(); i++)
					{
						if(tensors[i]->size() != 0)
						{
							send_tensor(master_conn, tensors[i]);
						}
					}

				}

				void send_parameters(connection* slave, std::vector<tensor*> parameters)
				{
					for(size_t i = 0; i < parameters.size(); i++)
					{
						if(parameters[i]->size() != 0)
						{
							send_tensor(slave, parameters[i]);
						}
					}

				}


				/******************************************************
				 *	Serialized send all tensors to all alive slaves
				 *
				 ******************************************************/
				void send_parameters_to_slaves(std::vector<tensor*> parameters)
				{
					if(get_running_slaves_num() != 0){

						for(int s_c = 0, s_c_max = slave_status.size(); s_c < s_c_max ; s_c ++){
							if(slave_status[s_c] == slaveStatus::Running){
								send_parameters(this->slave_conn[s_c], parameters);
							}
						}
					}
				}


				/********************************************************************************************************/
				
				int recieve_tensor(connection* src, tensor* contrainer)
				{
					char sizeBuf[30];
					src->read(sizeBuf, 30);
					std::cout << sizeBuf << std::endl;
					size_t length = 0;
					try{
						length = atoi(sizeBuf);
						if(this->verbose)
							std::cout << "[!]Start recieving tensor, the size is " << length << std::endl;

					}catch(...){
						std::cerr << "incorrect with converting" << std::endl;
					}

					float* tmpBuf = (float*) malloc( sizeof(float) * length);
					src->read((char*)tmpBuf, sizeof(float) * length);

					float* tmpPrt = tmpBuf;
					for(auto j = contrainer->begin(); j != contrainer->end(); j++){
						*j = *(tmpPrt++);
					}

					send_ack(src, "got");
					return length;
				}


				int recieve_gradients_from_one(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors){

					for(size_t i = 0; i < cli_tensors[slave_index].size(); i++)
					{
						if(cli_tensors[slave_index][i].size() != 0)
						{
							this->recieve_tensor(this->slave_conns[slave_index], &cli_tensors[slave_index][i]);

							print_tensor(&cli_tensors[slave_index][i], cli_tensors[slave_index][i].size());
						}
					}


					return 1;
				}

				void init_before_recieving(std::vector<std::vector<resizable_tensor>> &all_tensors)
				{
					// Get the pointer of gradients from current device
					std::vector<tensor*> tensors;
					tensors.resize(this->trainer->num_computational_layers);
					visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

					// Initialize temporary gradients contrainer from all other devices
					all_tensors.resize(slave_status.size() + 1);
					for(size_t i = 0; i < all_tensors.size(); i++){
						all_tensors[i].resize(this->trainer->num_computational_layers);
						for(size_t j = 0; j < all_tensors[i].size(); j++){

							if(i != (all_tensors.size()-1))
							{
								// Set size of tensors to slaves
								if(slave_status[i] == slaveStatus::Running)
									all_tensors[i][j].copy_size(*tensors[j]);
							}
							else
							{
								// Set master
								all_tensors[i][j].copy_size(*tensors[j]);
							}
						}
					}

					// Fill the container with currect device value
					for(size_t i = 0; i < tensors.size(); i ++){
						all_tensors[all_tensors.size() - 1][i] = *tensors[i];
					}

				}

				void recieve_gradients_serialism(std::vector<std::vector<resizable_tensor>> &all_tensors)
				{

					init_before_recieving(all_tensors);

					// Get gradients if there exists slave machine
					if(get_running_slaves_num() != 0)
					{
						for(int s_c = 0, s_c_max = slave_status.size(); s_c < s_c_max ; s_c ++)
						{
							if(slave_status[s_c] == slaveStatus::Running)
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
				}


				void recieve_updated_parameters()
				{

				}

				void average(std::vector<std::vector<resizable_tensor>> &all_tensors){
					std::vector<std::vector<tensor*>> accessible_groups;
					float scale = 1.0 / all_tensors.size();
					for(size_t i = 0; i < this->trainer->num_computational_layers; i++)
					{
						std::vector<tensor*> group;
						for(size_t j = 0; j < all_tensors.size(); j++)
						{
							if(all_tensors[j][i].size() != 0 && this->slave_status[j] == slaveStatus::Running)
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

				void sync(){

					// this->trainer->wait_for_thread_to_pause();

					if(ismaster){

						std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);


						std::vector<std::vector<resizable_tensor>> all_tensors;

						////////////////////////////////////////////////////////////
						auto epoch_time = system_clock::now();  // HPZ: Counting
						////////////////////////////////////////////////////////////

						recieve_gradients_serialism(all_tensors);

						//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						std::cout << "(Time for recieve_tensor) is "																								//
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting //
						//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

						for(size_t i = 0; i < all_tensors.size(); i++){
							for(size_t j = 0; j < all_tensors[i].size(); j++){
								print_tensor(&all_tensors[i][j], 10);
							}
						}

						////////////////////////////////////////////////////
						epoch_time = system_clock::now();  // HPZ: Counting
						////////////////////////////////////////////////////

						average(all_tensors);

						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						std::cout << "(Time for average) is "																										  //
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting   //
						epoch_time = system_clock::now();  // HPZ: Counting																							  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

						std::vector<tensor*> temp(this->trainer->num_computational_layers);
						for(size_t i = 0; i < temp.size(); i++)
						{
							// TODO : Deal with 0
							temp[i] = &all_tensors[0][i];
						}
						update(temp);

						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						std::cout << "(Time for update) is "																										  //
							<< std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting   //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

						// std::cout << "After" << std::endl;
						for(size_t i = 0; i < all_tensors.size(); i++){
							for(size_t j = 0; j < all_tensors[i].size(); j++){
								// std::cout <<  "[" <<all_tensors[i][j].size() << "]";
								for(auto k = all_tensors[i][j].begin(); k != all_tensors[i][j].end(); k ++){
									if(k == all_tensors[i][j].begin() + 10)
										break;
									// std::cout << "(" << k << ")";
									// std::cout << *k << " ";
								}
								// std::cout << std::endl;
							}
						}
					}else{

						std::vector<resizable_tensor> updated;

						send_gradients_to_master();
					}
					std::cout << "Sync finished" << std::endl;
					sleep(1000);
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
