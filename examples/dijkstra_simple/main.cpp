/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#include "../../src/cupq.h"
#include "dijkstra.h"
#include <algorithm>
#include <vector>
int main(int argc, char **argv) {

  unsigned norigins = 1024;
  unsigned ndestinations = 1;
  std::vector<int> origins(norigins, 0);
  std::vector<int> destinations(ndestinations, 0);

  auto vec1 = dijkstra_gpu(argc, argv, origins, destinations);
#ifdef CPU_COMPARISON
  if (vec1.size()) {
    std::cout << "total origins: " << origins.size();
    std::cout << " total destinations: " << destinations.size() << "\n";
    auto vec2 = dijkstra_cpu(argc, argv, origins, destinations);

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
#endif
}
