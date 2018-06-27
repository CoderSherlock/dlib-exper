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

				int recv_tensor(int slave_index, std::vector<std::vector<resizable_tensor>> &cli_tensors){
					

					for(size_t i = 0; i < cli_tensors[slave_index].size(); i++){
						if(cli_tensors[i].size() != 0){
							char tBuf[30];
							slave_conns[slave_index]->read(tBuf, 30);
							size_t temp_length = 0;
							try{
								temp_length = atoi(tBuf);
							}catch(...){
								std::cerr << "incorrect with converting" << std::endl;
							}
							// while(temp_length--){
							for(auto j = cli_tensors[slave_index][i].begin(); j != cli_tensors[slave_index][i].end(); j++){
								char tbp[4];
								slave_conns[slave_index]->read(tbp, 4);
								float tmg = 0;
								std::memcpy((char*)&tmg, tbp, 4);
								*j = tmg;
							}
						}
					}

					return 1;
				}


				void send_tensor(){

					// std::cout << trainer->devices[0]->net << std::endl;
					std::vector<tensor*> tensors;
					tensors.resize(this->trainer->num_computational_layers);

					visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});

					for(size_t i = 0; i < tensors.size(); i++){
						std::cout << tensors[i] << " "<< tensors[i]->size() << std::endl;
					}

					for(size_t i = 0; i < tensors.size(); i++){
						if(tensors[i]->size() != 0){
							char tBuf[30];
							snprintf(tBuf, sizeof(tBuf), "%lu", tensors[i]->size());
							master_conn->write(tBuf, 30);

							for(auto j = tensors[i]->begin(); j != tensors[i]->end(); j++){
								float tmpf = *j;
								master_conn->write((char*)&tmpf, 4);
							}
						}
					}

				}

				void recieve_tensor(){
					std::vector<std::vector<resizable_tensor>> cli_tensors;
					
					// Get gradients from current device
					std::vector<tensor*> tensors;
					tensors.resize(this->trainer->num_computational_layers);
					visit_layer_parameter_gradients(trainer->devices[0]->net, [&](size_t i, tensor& t){tensors[i] = &t;});
					
					// Get gradients from all other devices
					cli_tensors.resize(slave_status.size());
					for(size_t i = 0; i < cli_tensors.size(); i++){
						cli_tensors[i].resize(this->trainer->num_computational_layers);
						for(size_t j = 0; j < cli_tensors[i].size(); j++){
							cli_tensors[i][j].copy_size(*tensors[j]);
							std::cout << cli_tensors[i][j].size() << " ";
						}
						std::cout << std::endl;
					}

					// Get gradients if there exists slave machine
					if(get_running_slaves_num() != 0){

						for(int s_c = 0, s_c_max = slave_status.size(); s_c < s_c_max ; s_c ++){
							if(slave_status[s_c] == slaveStatus::Running){
								std::cout << "Reciveing from " << s_c << std::endl;
								recv_tensor(s_c, cli_tensors);

								for(int j = 0; j < cli_tensors[s_c].size(); j++){
									std::cout << cli_tensors[s_c][j].size();
									// for(auto k = cli_tensors[s_c][j].begin(); k != cli_tensors[s_c][j].end(); k ++){
									//     std::cout << *k << " ";
									// }
									std::cout << std::endl;
								}
							}
							
						}					
					
					
					}
					
					

				
				}

				void sync(){

					this->trainer->wait_for_thread_to_pause();

					if(ismaster){

						std::vector<tt::multi_device_tensor_averager> averagers = std::vector<tt::multi_device_tensor_averager>(this->trainer->num_computational_layers);


						recieve_tensor();
						std::vector<std::vector<resizable_tensor>> all_tensors(this->get_running_slaves_num() + 1);
						for(size_t i = 0; i < all_tensors.size(); i++){
							all_tensors[i].resize(this->trainer->num_computational_layers);
						}


					}else{

						send_tensor();
						// Send length of the current matrix
						// char tBuf[30];
						// snprintf(tBuf, sizeof(tBuf), "%d", me.number);
						// master_conn->write(tBuf, 30);
					}
					std::cout << "Sync finished" << std::endl;
					sleep(10000);

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
