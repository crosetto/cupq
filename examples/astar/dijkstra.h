/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
std::vector<float> dijkstra(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_,
                                cupq::Graph<float, unsigned, cupq::Backend::Host>& graph_);
std::vector<float> astar(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_,
                                cupq::Graph<float, unsigned, cupq::Backend::Host>& graph_);
