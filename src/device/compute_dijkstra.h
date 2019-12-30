/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include "heap.h"
#include <cfloat>
#include <chrono>

namespace cupq {
template <Backend> struct BackendSpecific;

template <class TVector, typename TValue, typename TIndex>
void computeDijkstra(DevicePair<Matrix<TValue, Backend::Cuda>,
                                Matrix<TIndex, Backend::Cuda>> &d_out_,
                     Graph<TValue, TIndex, Backend::Cuda> const &graph_,
                     TVector const &in_sources_, TIndex no_path_,
                     TIndex source_id_,
                     BackendSpecific<Backend::Cuda> const &backend_) {

  auto const &stream_ = backend_.mStream;
  auto const &err_ = backend_.mErr;

#ifdef CUSP_VERBOSE
  // print device properties
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    printf("  Available global memory (MB): %f\n", prop.totalGlobalMem / 1.0e6);
    printf("  Shared memory per block (B): %zu\n", prop.sharedMemPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Clock rate (MHz): %f\n\n", prop.clockRate / 1.e3);
  }
#endif

  constexpr unsigned num_blocks = 512;
  constexpr unsigned warps_per_block = 4;
  // unsigned numBlocks = ((float)in_sources_.size())/(warpsPerBlock*32);
  // numBlocks = numBlocks>0 ? numBlocks : 1;
  constexpr int N = 32;

  dim3 threads_per_block(N, warps_per_block);

  using value_t = typename Graph<TValue, TIndex, Backend::Cuda>::value_t;
  using index_t = typename TVector::value_type;

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(Node<value_t, index_t,
  // N>)*LIMIT*N*warps_per_block*num_blocks);

#ifdef CUSP_TIMINGS
  cudaEvent_t estart, estop;
  cudaEventCreate(&estart);
  cudaEventCreate(&estop);
  cudaEventRecord(estart);
#endif

  using node_t = Node<value_t, index_t, N>;

  backend_.sync();
  cudaCheckMemory();

  launch_sssp<warps_per_block, LIMIT>
      <<<num_blocks, threads_per_block, 0, stream_>>>(
          graph_, d_out_.first, d_out_.second, in_sources_, source_id_,
          no_path_, err_, num_blocks);

#ifdef CUSP_TIMINGS
  cudaEventRecord(estop);
  cudaEventSynchronize(estop);
#endif
}
} // namespace cupq
