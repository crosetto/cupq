/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include "../../src/cupq.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/**
   Banchmark for the GPU heap implementation: a many-to-many shortest path
   problem solved by calling multiple instances of the Dijkstra algorithm in
   parallel.
*/
std::vector<float> dijkstra_gpu(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_) {

  // non blocking stream synchronization
  cupq::cudaCheckErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield));

  // parameters hardcoded
  int nstreams = 1; // number of stream (= number of threads) for the GPU computation
  unsigned chunk = 1024; // number of input points to consume per thread at once

  // the output vector
  std::vector<float> finalCosts;

  // command line options (i.e. the graph filename)
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "graph,g", boost::program_options::value<std::string>(),
      "graph filename without extensions");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc), vm);

  std::string graphname_;

  if (vm.count("help") || !vm.size()) {
    std::cout << "USAGE:\n\n"
                 "dijkstra_simple_exe -g <graph file> \n\n"
                 "printing this help:\n\n"
                 "dijkstra_exe -h \n"
                 "dijkstra_exe \n"
                 "\n";
    return finalCosts;
  }

  if (vm.count("graph")) {
    graphname_ = vm["graph"].as<std::string>();
  } else {
    std::cout << "specify a graph file";
    return finalCosts;
  }
  std::vector<std::thread> t_;

  // instantiating the graph type
  using value_t = float;
  using index_t = unsigned;
  using graph_t = cupq::Graph<value_t, index_t, cupq::Backend::Cuda>;

  graph_t graph_;
  // reading the graph in DIMACS format. Multiple files for this benchmark can
  // be downloadeed from
  // http://users.diag.uniroma1.it/challenge9/download.shtml
  graph_.readDIMACS((graphname_ + ".gr").c_str());

  // generating random origins and destinations
  // srand(time(0));
  for (auto &it : origins_) {
    it = (int)(rand() % graph_.size());
  }
  for (auto &it : destinations_) {
    it = (int)(rand() % graph_.size());
  }

  // //removing duplicates
  // std::sort( origins_.begin(), origins_.end() );
  // origins_.erase( std::unique( origins_.begin(), origins_.end() ),
  // origins_.end() ); std::sort( destinations_.begin(), destinations_.end() );
  // destinations_.erase( std::unique( destinations_.begin(),
  // destinations_.end() ), destinations_.end() );
  finalCosts.resize(origins_.size() * destinations_.size(), 0.);

  // older GPUs do not properly support unified memory. When the UNIFMEM
  // preprocessor flag is unset we need to explicitly manage the copies to/from
  // the device
#ifndef UNIFMEM
  graph_.copyToDevice();
  graph_.setOnDevice();
#endif

  {

    // instantiating one solver for each thread (stream)
    std::vector<cupq::DijkstraSolver<graph_t>> shortestPathSolver(nstreams);

#ifdef TIMINGS
    using time_t = std::chrono::duration<double>;
    std::vector<time_t> compute_time(nstreams, time_t(0.));
#endif

    std::cout << "starting computation on GPU\n";

    // partitioning the origins into chunks, so that the memory does not blow up
    // when we solve for a large number of origins/destinations
    for (unsigned kk(0); kk < (float)origins_.size() / chunk; ++kk) {

      // defining a lambda containing the work to be done concurrently
      auto work = [&shortestPathSolver, &graph_, nstreams, kk, chunk, &origins_,
                   &destinations_, &finalCosts
#ifdef TIMINGS
                   ,
                   &compute_time
#endif
      ]() {
        if (origins_.size() && destinations_.size()) {

          auto &solver_ = shortestPathSolver[kk % nstreams];
          // selecting one chunk of the origins vector
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

#ifndef UNIFMEM
          solver_.out().first.copyToHost();
          solver_.out().first.setOnHost();
#endif

          for (int row = 0; row < origins_chunk_.size(); ++row)
            for (int col = 0; col < destinations_.size(); ++col)
              finalCosts[(row + kk * chunk) * destinations_.size() + col] =
                  solver_.out().first(row, destinations_[col]);

#ifndef UNIFMEM
          solver_.out().first.setOnDevice(); // setting it to device again for
                                             // the next iteration
#endif

#ifdef TIMINGS
          auto dend = std::chrono::system_clock::now();
          compute_time[kk % nstreams] += dend - dstart;
#endif
        }
      };

      if (kk < nstreams)
        t_.emplace_back(work);
      else { // blocking when there are nstreams threads in flight
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

    cudaDeviceSynchronize(); // in case some stream is still executing

  } // RAII

  graph_.free();

  return finalCosts;
}
