#pragma once

namespace cupq {

template <typename TGraph, typename TValue, typename TIndex, typename TBackend>
void computeHeuristicEuclidean(
    TGraph const &graph_, Vector<TIndex, TBackend::backend> const &destinations2_,
    Vector<TValue, TBackend::backend> &heuristic_, Vector<unsigned, TBackend::backend> &n_destinations_,
    Vector<TIndex, TBackend::backend> const &sources_, Matrix<char, TBackend::backend> const &is_destination_,
    std::vector<std::pair<TValue, TValue>> const &coordinates_, TValue vel_max_,
    TBackend const& backend_);


template <typename TGraph> struct AStarSolver {

  using value_t = typename TGraph::value_t;
  using index_t = typename TGraph::index_t;

  AStarSolver() {
    mBackend.init();
  };

  void setGraph(TGraph &graph_) {
    mGraph = &graph_;
    mGraphSize = graph_.size();
  }

  template <typename TVector>
  void setSources(TVector const &sources_,
                  DevicePair<unsigned, unsigned> const &chunk_) {
    if (mSources.size() != chunk_.second - chunk_.first)
      mSources.resize(chunk_.second - chunk_.first);
    for (unsigned i = 0; i < chunk_.second - chunk_.first; ++i)
      mSources[i] = sources_[i + chunk_.first];
  }

  template <typename TVector> void setSources(TVector const &sources_) {
    // if(mSources.size() != sources_.size())
    // 	mSources.resize(sources_.size());
    mSources.free();
    mSources.setup();
    mSources.resize(sources_.size());
    mBackend.attachStream(mSources);
    for (unsigned i = 0; i < sources_.size(); ++i)
      mSources[i] = sources_[i];
  }

  template <typename TVector> void setDestinations(TVector const &destinations_) {
    // if(mDestinations.size() != destinations_.size())
    // 	mDestinations.resize(destinations_.size());
    mDestinations.free();
    mDestinations.setup();
    mDestinations.resize(destinations_.size());
    mBackend.attachStream(mDestinations);
    for (unsigned i = 0; i < destinations_.size(); ++i)
      mDestinations[i] = destinations_[i];
  }

#if (__cplusplus >= 201402L)
  auto &
#else
  DevicePair<
      Matrix<typename TGraph::value_t, TGraph::backend>, Matrix<typename TGraph::index_t, TGraph::backend>, >
      &
#endif
  out() {
    return mOut;
  }

  void computeHeuristicTight() {

    mNDestinations.resize(mGraphSize);

    mHeuristic.resize(mGraphSize);

    mDDestinations.resize(mSources.size(), mGraphSize);

    mBackend.attachStream(mOut);
	mBackend.attachStream(mDestinations);
    mBackend.attachStream(mDDestinations);
    mBackend.attachStream(mNDestinations);
    mBackend.attachStream(mHeuristic);

    for (unsigned i = 0; i < mSources.size(); ++i) {
      for (unsigned j = 0; j < mGraphSize; ++j)
        mDDestinations(i, j) = 0;
      for (unsigned j = 0u; j < mDestinations.size(); ++j) { // last nDest nodes
        mDDestinations(i, mDestinations[j]) = 1;
      }
    }

    mBackend.attachStreamErr();

    cupq::computeHeuristicTight(*mGraph, mDestinations, mHeuristic,
                                  mNDestinations, mSources, mDDestinations,
                                  mBackend);
  }

  void computeHeuristicEuclidean(
      std::vector<std::pair<value_t, value_t>> const &coordinates_,
      value_t vel_max_) {

    if (!mNDestinations.size()) {
      mBackend.init();
    }
    mHeuristic.resize(mGraphSize);
    mNDestinations.resize(mGraphSize);
    mDDestinations.resize(mSources.size(), mGraphSize);

    mBackend.attachStream(mSources);
    mBackend.attachStream(mDestinations);
    mBackend.attachStream(mDDestinations);
    mBackend.attachStream(mNDestinations);
    mBackend.attachStream(mHeuristic);
    mBackend.attachStreamErr();

    for (unsigned i = 0; i < mSources.size(); ++i) {
      for (unsigned j = 0; j < mGraphSize; ++j)
        mDDestinations(i, j) = 0;
      for (unsigned j = 0u; j < mDestinations.size(); ++j) { // last nDest nodes
        mDDestinations(i, mDestinations[j]) = 1;
      }
    }

    cupq::computeHeuristicEuclidean(*mGraph, mDestinations, mHeuristic,
                                      mNDestinations, mSources, mDDestinations,
                                      coordinates_, vel_max_, mBackend);
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

      mBackend.attachStream(mHeuristic);
      mBackend.attachStream(mDDestinations);
      mBackend.attachStream(mNDestinations);

    }
  }


  auto computeDijkstra(typename TGraph::index_t no_path_,
                       typename TGraph::index_t source_id_) {

    prepareData(no_path_);

    assert(mHeuristic.size());
    assert(mSources.size());
    assert(mNDestinations.size());
    assert(mDDestinations.ncols() && mDDestinations.nrows());
    assert(mOut.first.ncols() && mOut.first.nrows());
#ifdef SAVE_PATH
    assert(mOut.second.ncols() && mOut.second.nrows());
#endif
    // if(mOut.first.size() != mSources.size()){ //first pass
    // mOut.free();
    // mOut.first.setup();
    // mOut.second.setup();
    // mBackend.attachStream(mSources);

#ifdef CUSP_TIMINGS
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
    cudaEventRecord(mStart);
#endif

    cupq::computeAstar(mOut, *mGraph, mSources, mHeuristic, mNDestinations,
                        mDDestinations, source_id_, mBackend);

#ifdef CUSP_TIMINGS
    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
#endif

    // // cannot do it here, need to finalize first
    // for(int i=0; i<mSources.size(); ++i){
    //     for(int j=0; j<mDDestinations[i].size(); ++j)
    //         mDDestinations[i][j]=0;
    //     for( unsigned j = 0u; j < mDestinations.size(); ++j ) { //last nDest
    //     nodes
    //         mDDestinations[i][ mDestinations[j] ] = 1;
    //     }
    // }

    return mOut;
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

#ifdef CUSP_TIMINGS
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, mStart, mStop);
    std::cout << "elapsed time A*: " << ((float)milliseconds) / 1e3 << "s\n";
#endif

    for (unsigned i = 0; i < mSources.size(); ++i) {
      for (unsigned j = 0; j < mGraphSize; ++j)
        mDDestinations(i, j) = 0;
      for (unsigned j = 0u; j < mDestinations.size(); ++j) { // last nDest nodes
        mDDestinations(i, mDestinations[j]) = 1;
      }
    }

  }

  ~AStarSolver() {
    mSources.free();
    mDestinations.free();
    mHeuristic.free();
    mNDestinations.free();
    mDDestinations.free();
    mOut.free();
	mBackend.free();
  }

private:
  TGraph *mGraph = nullptr;
  Vector<index_t, TGraph::backend> mSources;
  Vector<index_t, TGraph::backend> mDestinations;

  // OVERHEAD w.r.t. Dijkstra:
  Vector<value_t, TGraph::backend> mHeuristic; // space complexity: #nodes
  Vector<unsigned, TGraph::backend> mNDestinations; // space complexity: #nodes. Reachable number of destinations for each source
  Matrix<char, TGraph::backend> mDDestinations; // space complexity (an overkill): #sources*#nodes. 
                                               // One map for each source, will contain "1" for each destination, "2" for an already visited node
                                               // One map per node because we write to the map concurrently. Should be 1 map per thread in reality.

  // pair<Vector<Vector<value_t, TGraph::backend>, TGraph::backend>,
  // Vector<Vector<index_t, TGraph::backend>, TGraph::backend>> mOut;
  DevicePair<Matrix<value_t, TGraph::backend>, Matrix<index_t, TGraph::backend>>
      mOut;
  index_t mGraphSize; // caching this, because the graph is shared among threads
  BackendSpecific<TGraph::backend> mBackend;
#ifdef CUSP_TIMINGS
  cudaEvent_t mStart, mStop;
#endif
  // (and thus accessing it after the first kernel launch we would need a call
  // to cudaDeviceSynchronize)
};

template <typename TGraph, typename TValue, typename TIndex, typename TBackend>
void computeHeuristicEuclidean(
    TGraph const &graph_, Vector<TIndex, TBackend::backend> const &destinations2_,
    Vector<TValue, TBackend::backend> &heuristic_, Vector<unsigned, TBackend::backend> &n_destinations_,
    Vector<TIndex, TBackend::backend> const &sources_, Matrix<char, TBackend::backend> const &is_destination_,
    std::vector<std::pair<TValue, TValue>> const &coordinates_, TValue vel_max_,
    TBackend const& backend_) {

  for (unsigned k = 0; k < graph_.size(); ++k) {
    n_destinations_[k] = 0;
  }

  for (unsigned k = 0; k < graph_.size(); ++k) {
    heuristic_[k] = FLT_MAX;
    for (unsigned j = 0; j < destinations2_.size(); ++j) {
      heuristic_[k] = std::min(
          heuristic_[k],
          std::sqrt((coordinates_[k].first - coordinates_[j].first) *
                        (coordinates_[k].first - coordinates_[j].first) +
                    (coordinates_[k].second - coordinates_[j].second) *
                        (coordinates_[k].second - coordinates_[j].second) /
                        vel_max_));
    }
  }

  for (unsigned k = 0; k < graph_.size(); ++k) {
    for (unsigned j = 0; j < sources_.size(); ++j) {
      if (is_destination_(0u, k))
        ++n_destinations_[sources_[j]];
    }
  }
}


} // namespace cupq
