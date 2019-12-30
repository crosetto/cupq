/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include "dijkstra.h"
#include <cfloat>
#include <chrono>

namespace cupq {
template <Backend> struct BackendSpecific;

template <class Vector, typename TValue, typename TIndex>
void computeDijkstra(DevicePair<Matrix<TValue, Backend::Host>,
                                Matrix<TIndex, Backend::Host>> &d_out_,
                     Graph<TValue, TIndex, Backend::Host> const &graph_,
                     Vector const &in_sources_, TIndex no_path_,
                     TIndex source_id_,
                     BackendSpecific<Backend::Host> const & /*backend_*/) {

  using value_t = TValue;
  using index_t = TIndex;
  auto &d_out_distance_ = d_out_.first;
  auto &d_out_predecessor_ = d_out_.second;

  dijkstra(graph_, d_out_distance_, d_out_predecessor_, in_sources_,
           source_id_);
}
} // namespace cupq
