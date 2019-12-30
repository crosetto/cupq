/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#ifdef CUSP_PROFILE_HEAP
#include <cuda_profiler_api.h>
#endif

namespace cupq {

template <typename TGraph> struct DijkstraSolver {

  using value_t = typename TGraph::value_t;
  using index_t = typename TGraph::index_t;

  DijkstraSolver() { mBackend.init(); }

  void setGraph(TGraph &graph_) {
    mGraph = &graph_;
    mGraphSize = graph_.size();
  }

  template <typename Vector>
  void setSources(Vector const &sources_,
                  DevicePair<unsigned, unsigned> const &chunk_) {
    if (mSources.size() != chunk_.second - chunk_.first)
      mSources.resize(chunk_.second - chunk_.first);
    for (unsigned i = 0; i < chunk_.second - chunk_.first; ++i)
      mSources[i] = sources_[i + chunk_.first];
  }

  template <typename Vector> void setSources(Vector const &sources_) {
    mSources.resize(sources_.size());
#ifndef UNIFMEM
    mSources.setOnHost();
#endif
    mBackend.attachStream(mSources);
    for (unsigned i = 0; i < sources_.size(); ++i)
      mSources[i] = sources_[i];
#ifndef UNIFMEM
    mSources.copyToDevice(mBackend);
    mSources.setOnDevice();
#endif
  }

#if (__cplusplus >= 201402L)
  auto &
#else
  DevicePair<Matrix<typename TGraph::value_t, TGraph::backend>,
             Matrix<typename TGraph::index_t, TGraph::backend>> &
#endif
  out() {
    return mOut;
  }

  void prepareData(typename TGraph::index_t no_path_) {

    if (mSources.size()) {
      mOut.first.resize(mSources.size(), mGraphSize);
#ifdef SAVE_PATH
      mOut.second.resize(mSources.size(), mGraphSize);
#endif
#ifndef UNIFMEM
      mOut.first.setOnHost();
#ifdef SAVE_PATH
      mOut.second.setOnHost();
#endif
#endif
      mBackend.attachStream(mOut);

      mBackend.attachStreamErr();
      for (int j = 0; j < mSources.size(); ++j) {
        for (int k = 0; k < mGraphSize; ++k) {
          mOut.first(j, k) = (value_t)FLT_MAX;
#ifdef SAVE_PATH
          mOut.second(j, k) = no_path_; //(TIndex)-1;
#endif
        }
      }

#ifndef UNIFMEM
      mOut.first.copyToDevice(mBackend);
      mOut.first.setOnDevice();
#ifdef SAVE_PATH
      mOut.second.copyToDevice(mBackend);
      mOut.second.setOnDevice();
#endif
#endif
    }
  }

  void computeDijkstra(typename TGraph::index_t no_path_,
                       typename TGraph::index_t source_id_) {

    prepareData(no_path_);
    cupq::computeDijkstra(mOut, *mGraph, mSources, no_path_, source_id_,
                          mBackend);
  }

  void finalize() {
    mBackend.fin();
#ifndef UNIFMEM
    mOut.first.copyToHost();
    mOut.first.setOnHost();
#ifdef SAVE_PATH
    mOut.second.copyToHost();
    mOut.second.setOnHost();
#endif
#endif
  }

  ~DijkstraSolver() {
    mSources.free();
    mOut.free();
    mBackend.free();
  }

private:
  TGraph *mGraph = nullptr;
  Vector<index_t, TGraph::backend> mSources;
  DevicePair<Matrix<value_t, TGraph::backend>, Matrix<index_t, TGraph::backend>>
      mOut;
  BackendSpecific<TGraph::backend> mBackend;
  index_t mGraphSize; // caching this, because the graph is shared among threads
};

} // namespace cupq
