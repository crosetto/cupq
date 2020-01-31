/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include "../../src/cupq.h"

// compile standalone with
// nvcc dijkstra_simple.cu -I.. -g -O3 -Xptxas -dlcm=ca -lineinfo
// -gencode=arch=compute_50,code=sm_50   -gencode=arch=compute_70,code=sm_70
// --ptxas-options=-v options:
// -DVERBOSE
// -DTIMINGS

// usage:

std::vector<float> astar(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_,
                                cupq::Graph<float, unsigned, cupq::Backend::Host>& graph_) {

  using graph_t = cupq::Graph<float, unsigned, cupq::Backend::Host>;
  using index_t = typename graph_t::index_t;
  using value_t = typename graph_t::value_t;

  // parameters hardcoded
  int nstreams = 1; // number of stream (= number of threads) for the GPU computation
  unsigned chunk = 1; // number of input points to consume per thread at once

  std::vector<float> finalCosts(origins_.size() * destinations_.size(), 0.);

  {

    // std::vector<cupq::DijkstraSolver<graph_t>> shortestPathSolver(nstreams);
    std::vector<cupq::AStarSolver<graph_t>> shortestPathSolver(nstreams);

#ifdef TIMINGS
    using time_t = std::chrono::duration<double>;
    std::vector<time_t> compute_time(nstreams, time_t(0.));
#endif

    std::cout << "starting A star computation on CPU \n";

	unsigned nruns_=3;
	for(unsigned run_=0; run_<nruns_; ++run_){

    std::vector<std::thread> t_;

    for (unsigned kk(0); kk < (float)origins_.size() / chunk; ++kk) {

      auto work = [&shortestPathSolver, &graph_, nstreams, kk, chunk, &origins_,
                   &destinations_, &finalCosts, run_
#ifdef TIMINGS
                   ,
                   &compute_time
#endif
      ]() {
        if (origins_.size() && destinations_.size()) {

          auto &solver_ = shortestPathSolver[kk % nstreams];

          std::vector<int> origins_chunk_(
              origins_.begin() + kk * chunk,
              origins_.begin() +
                  std::min((std::size_t)kk * chunk + chunk, origins_.size()));
          solver_.setSources(origins_chunk_);
          solver_.setGraph(graph_);
#ifdef TIMINGS
          auto dstart = std::chrono::system_clock::now();
#endif

          //AStar specific
          solver_.setDestinations(destinations_);
		  if(run_ == 0)// compute the heuristic only the first time
			  solver_.computeHeuristicTight();
          ////////////////

          solver_.computeDijkstra((index_t)-1, (index_t)-1);
          solver_.finalize();
          // result available on host in solver.out()
          for (int row = 0; row < origins_chunk_.size(); ++row)
            for (int col = 0; col < destinations_.size(); ++col)
              finalCosts[(row + kk * chunk) * destinations_.size() + col] =
                  solver_.out().first(row, destinations_[col]);

              // auto itCost=finalCosts.begin();
              // for(auto it=solver_.out().first.begin();
              // it!=solver_.out().first.end(); ++it) 	*itCost=*it;

#ifdef TIMINGS
          auto dend = std::chrono::system_clock::now();
          compute_time[kk % nstreams] += dend - dstart;
#endif
        }
      };

      if (kk < nstreams)
        t_.emplace_back(work);
      else { // blocking on a thread and reusing the resources
        t_[kk % nstreams].join();
        t_[kk % nstreams] = std::thread(work);
      }
    }

    for (auto &&i : t_)
      i.join();
	t_.resize(0);

    std::cout << "RUN " << run_ << " COMPLETED\n";

#ifdef TIMINGS
    std::cout << "computation time per tread (computing shortest path)\n";
    for (int i = 0; i < compute_time.size(); ++i)
      std::cout << "thread " << i << " computing for "
                << compute_time[i].count() << " s\n\n";
#endif
    for (auto &&i : compute_time)
      i*=0;
	if(run_ != nruns_-1)
		graph_.randomNoise();
  }

    // result available in solver[i].out()
  } // RAII

  return finalCosts;
}
