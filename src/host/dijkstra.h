/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <queue>

#ifdef __CUDACC__
#ifdef _MSC_VER
#define WIN_BUG_1 // VS compiler bug when including some boost and CUDA headers
#endif
#endif

#ifndef WIN_BUG_1
#include <boost/heap/d_ary_heap.hpp>
#endif

/** CPU version of the Dijkstra algorithm (for comparison) */

namespace cupq {
namespace impl_ {

/** compare predicate class */
template <typename TIndex> struct Cmp {
  Cmp() {}

  template <typename T> bool operator()(T const &a, T const &b) const {
    return a.first > b.first;
  }

  int min_value() const { return std::numeric_limits<TIndex>::min(); }
};
} // namespace impl_

template <typename Graph, typename VectorVal, typename VectorId,
          typename VecSources>
void static dijkstra(Graph &graph_, VectorVal &potential_, VectorId &parent_,
                     VecSources const &sources_,
                     typename VectorId::value_type source_id_) {
  static_assert(is_graph<typename std::decay<Graph>::type>::value,
                "wrong graph type");
  static_assert(is_Matrix<VectorVal>::value, "wrong vector type");
  static_assert(is_Matrix<VectorId>::value, "wrong vector type");

  int visited=0;

  for (int sourceIt = 0; sourceIt < sources_.size(); ++sourceIt) {

    using float_t = typename VectorVal::value_type;
    using index_t = typename VectorId::value_type;

#ifndef WIN_BUG_1
    ::boost::heap::d_ary_heap<
        std::pair<float_t, index_t>,
        ::boost::heap::compare<impl_::Cmp<std::pair<float_t, index_t>>>,
        ::boost::heap::arity<8>>
        waiting_;
#else
    std::priority_queue<std::pair<float_t, index_t>,
                        std::vector<std::pair<float_t, index_t>>,
                        std::greater<std::pair<float_t, index_t>>>
        waiting_;
#endif

    waiting_.reserve(potential_.nrows());
    potential_(sourceIt, sources_[sourceIt]) = 0;
#ifdef SAVE_PATH
    parent_(sourceIt, sources_[sourceIt]) = (source_id_ == (TIndex)-1) ? sources_[sourceIt] : source_id_;
#endif
    waiting_.push(std::make_pair(0, sources_[sourceIt]));
    // int k=1;
    while (!waiting_.empty()) {

      auto i = waiting_.top().second;

#ifdef CHECK_RESULTS
#ifdef CUSP_VERBOSE
      printf("host popping %d \n", i);
#endif
#endif

      waiting_.pop();

      for (int j = graph_.sourceOffsets(i); j < graph_.sourceOffsets(i + 1);
           ++j) {

        auto e_ = graph_.destinationIndices(j);
        if (potential_(sourceIt, i) + graph_.weights(j) <
            potential_(sourceIt, e_)) {
#ifdef SAVE_PATH
          parent_(sourceIt, e_) = i;
#endif
          waiting_.push(
              std::make_pair(potential_(sourceIt, i) + graph_.weights(j), e_));
		  ++visited;

          potential_(sourceIt, e_) =
              potential_(sourceIt, i) + graph_.weights(j);
#ifdef CHECK_RESULTS
#ifdef CUSP_VERBOSE
          printf("host inserting %d, ", e_);
          printf("host distance %f\n",
                 potential_(sourceIt, i) + graph_.weights(j));
#endif
#endif
        }
      }
    }

  std::cout<<"visited "<<visited <<" edges\n";

  }
}

} // namespace cupq
