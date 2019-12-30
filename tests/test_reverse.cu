/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <vector>

// boosts timers
#include <boost/timer/timer.hpp>

#include "../src/cupq.h"

// nvcc -gencode=arch=compute_50,code=sm_50 -std=c++14 test_reverse.cu -O3
// -DNDEBUG -lboost_timer -g

int main(int argc, char **argv) {
  using namespace cupq;

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
    return 0;
  }

  if (vm.count("graph")) {
    graphname_ = vm["graph"].as<std::string>();
  } else {
    std::cout << "specify a graph file";
  }

  size_t s;
  cudaDeviceGetLimit(&s, cudaLimitPrintfFifoSize);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, s * 100);

  using value_t = float;
  using index_t = int;
  using graph_t = Graph<value_t, index_t, cupq::Backend::Host>;

  graph_t graph_;
  graph_.setup();
  graph_.readDIMACS(graphname_.c_str());

  // computing heuristic
  {
    unsigned N = 92;
    std::vector<index_t> sources_(N);
    for (auto i = 0; i < N; ++i) {
      sources_[i] = i + 1;
    }

    std::vector<index_t> destinations_(N);
    for (auto i = 0; i < N; ++i) {
      destinations_[i] = graph_.size() - i - 1;
    }

    auto rgraph_ = graph_.reverse();

    cupq::DijkstraSolver<graph_t> solver_;
    solver_.setSources(sources_);
    solver_.setGraph(graph_);

    solver_.computeDijkstra(-1, -1);
    solver_.finalize();
    auto out = solver_.out();
    graph_.free();
    solver_.setGraph(rgraph_);
    solver_.setSources(destinations_);
    solver_.computeDijkstra(-1, -1);
    auto out3 = solver_.out();
    auto rrgraph_ = rgraph_.reverse();
    rgraph_.free();

    solver_.setGraph(rrgraph_);
    solver_.setSources(sources_);
    solver_.computeDijkstra(-1, -1);
    auto out2 = solver_.out();

    std::cout << "CPU:\n\n";
    for (unsigned i = 0; i < N; ++i) {
      Matrix<value_t, cupq::Backend::Host> d_potential_(graph_.nbNodes() - 1, 1,
                                                        (value_t)FLT_MAX);
      Matrix<index_t, cupq::Backend::Host> d_parent_(graph_.nbNodes() - 1, 1,
                                                     -1);
      std::vector<index_t> orig_{(index_t)(i + 1)};
      dijkstra(graph_, d_potential_, d_parent_, orig_, -1);

      for (unsigned k = 0; k < d_potential_.nrows(); ++k)
        if (out.first(i, k) != d_potential_(k, 0)) {
          if (d_parent_(k, 0) != -1) {
            std::cout << "error66\n";
            std::cout << i << " " << k << " " << out.second(i, k)
                      << " != " << d_parent_(k, 0) << "\n";
            std::cout << out.first(i, k) << " != " << d_potential_(k, 0)
                      << "\n";
          }
        }
    }

    for (auto i = 0; i < N; ++i)
      for (unsigned k = 0; k < graph_.size(); ++k)
        if (out.first(i, k) != out2.first(i, k)) {
          std::cout << "error2\n";
          std::cout << i << " " << k << " " << out.second(i, k)
                    << " != " << out2.second(i, k) << "\n";
          std::cout << out.first(i, k) << " != " << out2.first(i, k) << "\n";
          break;
        }

    auto eps = .1;
    for (auto i = 0; i < N; ++i)
      for (auto j = 0; j < N; ++j)
        if (out.first(i, destinations_[j]) > out3.first(j, sources_[i]) + eps ||
            out.first(i, destinations_[j]) < out3.first(j, sources_[i]) - eps) {
          std::cout << "error3\n";
          std::cout << i << " " << j << " " << out.second(i, destinations_[j])
                    << " != " << out3.second(j, sources_[i]) << "\n";
          std::cout << out.first(i, destinations_[j])
                    << " != " << out3.first(j, sources_[i]) << "\n";
          break;
        }
  }
}
