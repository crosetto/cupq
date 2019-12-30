/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <stdlib.h>
#include <vector>

namespace cupq {
enum class Backend { Cuda, Host };
#if (__cplusplus >= 201402L)

static constexpr std::size_t LIMIT = 256 - 16;
static constexpr std::size_t LOGLIMIT = 4;
#else
static const std::size_t LIMIT = 256 - 16;
static const std::size_t LOGLIMIT = 4;
#endif
} // namespace cupq

#ifdef __CUDACC__
#include "device/cuda_check_errors.h"
#include "device/cuda_check_memory.h"
#endif

#include "backend_specific.h"
#ifdef __CUDACC__
#include "device/attach_stream_fwd.h"
#include "device/backend_specific_cuda.h"
#endif

#ifdef UNIFMEM
#include "vector.h"
#else
#include "vector_legacy.h"
#endif

#include "array.h"
#include "device_pair.h"
#include "graph.h"
#include "matrix.h"

#ifdef __CUDACC__
#include "device/attach_stream.h"
#include "device/compute_dijkstra.h"
#include "device/kernel.h"
#endif

#include "dijkstra_solver.h"
#include "host/compute_dijkstra_host.h"
