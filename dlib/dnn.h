// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_
#define DLIB_DNn_

// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases
#ifdef _MSC_VER
#   pragma warning( disable: 4503 )
#endif

#include "cuda/tensor.h"
#include "dnn/input.h"

// Problem:    Visual Studio's vcpkgsrv.exe constantly uses a single CPU core,
//             apparently never finishing whatever it's trying to do. Moreover,
//             this issue prevents some operations like switching from Debug to
//             Release (and vice versa) in the IDE. (Your mileage may vary.)
// Workaround: Keep manually killing the vcpkgsrv.exe process.
// Solution:   Disable IntelliSense for some files. Which files? Unfortunately
//             this seems to be a trial-and-error process.
#ifndef __INTELLISENSE__
#include "dnn/layers.h"
#endif // __INTELLISENSE__

#include "dnn/loss.h"
#include "dnn/core.h"
#include "dnn/solvers.h"
#include "dnn/trainer.h"
#include "dnn/syncer/syncer.h"						// HPZ: Add Syncer header
#include "dnn/syncer/syncer_base.h"
#include "dnn/syncer/syncer_leader.h"       // HPZ: Add Syncer header
#include "dnn/syncer/syncer_worker.h"
#include "dnn/syncer/syncer_full_leader.h"				// HPZ: Add Syncer header
#include "dnn/syncer/config.h"
#include "cuda/cpu_dlib.h"
#include "cuda/tensor_tools.h"
#include "dnn/utilities.h"
#include "dnn/validation.h"

#endif // DLIB_DNn_


