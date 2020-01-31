/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include <algorithm>
#include <vector>
#include <thread>
#include <boost/program_options.hpp>
#include "../../src/cupq.h"
#include "dijkstra.h"

int main(int argc, char **argv) {

  unsigned norigins = 1;
  unsigned ndestinations = 2;
  std::vector<int> origins(norigins, 0);
  std::vector<int> destinations(ndestinations, 0);

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
    return -1;
  }

  if (vm.count("graph")) {
    graphname_ = vm["graph"].as<std::string>();
  } else {
    std::cout << "specify a graph file";
    return -2;
  }
  std::vector<std::thread> t_;

  // instantiating the graph type
  using value_t = float;
  using index_t = unsigned;
  using graph_t = cupq::Graph<value_t, index_t, cupq::Backend::Host>;

  graph_t graph_;
  // reading the graph in DIMACS format. Multiple files for this benchmark can
  // be downloadeed from
  // http://users.diag.uniroma1.it/challenge9/download.shtml
  // graph_.readDIMACS((graphname_ + ".gr").c_str());

  // reading Aimsun graph
  graph_.readGraph((graphname_+".gr").c_str());
  graph_.readWeights( (graphname_+".w").c_str() );

  // std::cout<< "reading graph coordinates\n";
  // auto coordinates_ = graph_.readDIMACSCoordinates( (graphname_+".co").c_str() );
  // auto coordinates_ = graph_.readCoordinates( (graphname_+".co").c_str() );

  // generating random origins and destinations
  // srand(time(0));
  for (auto &it : origins) {
    it = (int)(rand() % graph_.size());
  }
  for (auto &it : destinations) {
    it = (int)(rand() % 10 /*graph_.size()*/);
  }

  //removing duplicates
  std::sort( origins.begin(), origins.end() );
  origins.erase( std::unique( origins.begin(), origins.end() ),
  origins.end() ); std::sort( destinations.begin(), destinations.end() );
  destinations.erase( std::unique( destinations.begin(),
  destinations.end() ), destinations.end() );

  auto vec1 = astar(argc, argv, origins, destinations, graph_);
  auto vec2 = dijkstra(argc, argv, origins, destinations, graph_);

  if (vec1.size()) {
    std::cout << "total origins: " << origins.size();
    std::cout << " total destinations: " << destinations.size() << "\n";

    std::cout << "Comparing results\n";
    bool OK = true;
    for (std::size_t k = 0; k < vec2.size(); ++k) {
      if (vec1[k] != vec2[k]) {
        std::cout << "Error in verification\n";
        std::cout << "[" << k << "]" << vec1[k] << " != " << vec2[k] << "\n";
        OK = false;
      } else {
        // std::cout<<"OK\n";
        // std::cout<<"[" << k << "]" <<vec1[k] <<" == " <<vec2[k] <<"\n";
      }
    }
    if (OK)
      std::cout << "OK\n";
  }
  graph_.free();

}
