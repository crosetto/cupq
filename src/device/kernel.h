/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include "heap.h"
#include <cfloat>

#ifdef CUSP_PROFILE_HEAP
#include <cuda_profiler_api.h>
#endif

namespace cupq {
template <unsigned warpsPerBlock, int Limit, typename TValue, typename TIndex>
__device__ cudaError_t do_work(
    Array<TValue, Limit> &sTops, int &sPosition, TIndex &sTops2,
    Graph<TValue, TIndex, Backend::Cuda> const *graph_, TValue *out_distance_,
#ifdef SAVE_PATH
    TIndex *out_predecessor_,
#endif
    TIndex const &source_, int source_id_) {

  Heap<TValue, TIndex, 32, 32, warpsPerBlock, Limit> pq(sTops, sPosition,
                                                        sTops2);

  source_id_ = (source_id_ == (TIndex)-1) ? source_ : source_id_;
#ifdef SAVE_PATH
  out_predecessor_[source_] = source_id_;
#endif
  out_distance_[source_] = (TValue)0;
  pq.insert(make_DevicePair((TValue)0., source_), sTops, sPosition, sTops2);
  while (pq.size() > 0) {

    TIndex i = pq.pop(sTops, sPosition, sTops2);

    auto offseti = __ldg(&(*graph_).sourceOffsets(i));
    auto offsetip1 = __ldg(&(*graph_).sourceOffsets(i + 1));
    for (TIndex j = offseti; j < offsetip1; ++j) {
      auto idx = j;
      auto e_ = (*graph_).destinationIndices(idx);
      auto weight_ = (*graph_).weights(idx);
      TValue new_cost_ = __ldg(&out_distance_[i]) + weight_;

      if (new_cost_ < out_distance_[e_]) {
        out_distance_[e_] = new_cost_;
#ifdef SAVE_PATH
        out_predecessor_[e_] = i;
#endif
        if (pq.size() >= Limit)
          return cudaErrorMemoryAllocation;
        pq.insert(make_DevicePair(new_cost_, e_), sTops, sPosition, sTops2);
      }
    }
  }

  return cudaSuccess;
}

template <unsigned warpsPerBlock, int Limit, typename TValue, typename TIndex>
__global__ void launch_sssp(Graph<TValue, TIndex, Backend::Cuda> const graph_,
                            Matrix<TValue, Backend::Cuda> out_distance_,
                            Matrix<TIndex, Backend::Cuda> out_predecessor_,
                            Vector<TIndex, Backend::Cuda> const sources_,
                            TIndex source_id_, TIndex no_path_,
                            cudaError_t *err, unsigned const numBlocks) {

  __shared__ Array<TValue, Limit> sTops[warpsPerBlock];
  __shared__ int sPosition[warpsPerBlock];
  __shared__ TIndex sTops2[warpsPerBlock];

  unsigned totalWarps = numBlocks * warpsPerBlock;
  unsigned warpId = blockIdx.x * warpsPerBlock + threadIdx.y;

  for (unsigned ll = 0;
       warpId + totalWarps * ll < sources_.size() && *err == cudaSuccess;
       ++ll) {

    *err = do_work<warpsPerBlock, Limit>(
        sTops[threadIdx.y], sPosition[threadIdx.y], sTops2[threadIdx.y],
        &graph_, &out_distance_(warpId + ll * totalWarps, 0u),
#ifdef SAVE_PATH
        &out_predecessor_(warpId + ll * totalWarps, 0u),
#endif
        sources_[warpId + ll * totalWarps], source_id_);
  }
}

} // namespace cupq
