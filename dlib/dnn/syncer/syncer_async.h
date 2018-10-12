#ifndef DLIB_DNn_SYNCER_ASYNC_H_
#define DLIB_DNn_SYNCER_ASYNC_H_

#include "syncer.h"

namespace dlib {

template<typename trainer_type>
void dnn_async_leader<trainer_type>::init_reciever_pool() {
};

template<typename trainer_type>
void dnn_async_leader<trainer_type>::sync() {
	while (1) {
	}
};

}

#endif
