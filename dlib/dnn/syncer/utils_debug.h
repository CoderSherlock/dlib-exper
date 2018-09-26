#ifndef _DLIB_DNn_SYNCER_UTILS_DEBUG_H_
#define _DLIB_DNn_SYNCER_UTILS_DEBUG_H_

#include <iostream>
#include "../trainer.h"
#include "../../sockets.h"

namespace dlib {
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
	 *
	 ************************************************/
	void print_buffer(char* ptr, size_t n)
	{
		std::cout << "\n" << n << "---------------------------------\n";
		for(size_t i = 0; i < n; i++)
			std::cout << *(ptr + i);
		std::cout << "\n---------------------------------\n";
	}


}
#endif
