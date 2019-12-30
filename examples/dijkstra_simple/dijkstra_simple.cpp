/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include "../../src/cupq.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// compile standalone with
// nvcc dijkstra_simple.cu -I.. -g -O3 -Xptxas -dlcm=ca -lineinfo
// -gencode=arch=compute_50,code=sm_50   -gencode=arch=compute_70,code=sm_70
// --ptxas-options=-v options:
// -DVERBOSE
// -DTIMINGS

// usage:

std::vector<float> dijkstra_cpu(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_) {

  // parameters hardcoded
  int nstreams =
      1; // number of stream (= number of threads) for the GPU computation
  unsigned chunk = 1024; // number of input points to consume per thread at once

  std::vector<float> finalCosts(origins_.size() * destinations_.size(), 0.);

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "graph,g", boost::program_options::value<std::string>(),
      "graph filename without extensions");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);

  std::string graphname_;

  if (vm.count("graph")) {
    graphname_ = vm["graph"].as<std::string>();
  } else {
    std::cout << "specify a graph file";
    return finalCosts;
  }

  std::vector<std::thread> t_;

  // read graph
  using value_t = float;
  using index_t = unsigned;
  using graph_t = cupq::Graph<value_t, index_t, cupq::Backend::Host>;

  graph_t graph_;
  // read graph in DIMACS format
  graph_.readDIMACS((graphname_ + ".gr").c_str());

  {

    std::vector<cupq::DijkstraSolver<graph_t>> shortestPathSolver(nstreams);

#ifdef TIMINGS
    using time_t = std::chrono::duration<double>;
    std::vector<time_t> compute_time(nstreams, time_t(0.));
#endif

    std::cout << "starting computation on CPU \n";

    for (unsigned kk(0); kk < (float)origins_.size() / chunk; ++kk) {

      auto work = [&shortestPathSolver, &graph_, nstreams, kk, chunk, &origins_,
                   &destinations_, &finalCosts
#ifdef TIMINGS
                   ,
                   &compute_time
#endif
      ]() {
        if (origins_.size() && destinations_.size()) {

          auto &solver_ = shortestPathSolver[kk % nstreams];
          // std::range<int> origins_chunk_(origins_.begin()+kk,
          // std::max(origins_.begin()+kk+32, origins_.size_()));
          std::vector<int> origins_chunk_(
              origins_.begin() + kk * chunk,
              origins_.begin() +
                  std::min((std::size_t)kk * chunk + chunk, origins_.size()));
          solver_.setSources(origins_chunk_);
          solver_.setGraph(graph_);
#ifdef TIMINGS
          auto dstart = std::chrono::system_clock::now();
#endif
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

#ifdef TIMINGS
    std::cout << "computation time per tread (computing shortest path)\n";
    for (int i = 0; i < compute_time.size(); ++i)
      std::cout << "thread " << i << " computing for "
                << compute_time[i].count() << " s\n";
#endif
    // result available in solver[i].out()
  } // RAII

  graph_.free();

  return finalCosts;
}
