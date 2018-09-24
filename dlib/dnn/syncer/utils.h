#ifndef _DLIB_DNn_SYNCER_UTILS_H_
#define _DLIB_DNn_SYNCER_UTILS_H_

#include <iostream>

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
};

} // End of Namespace Dlib
#endif
