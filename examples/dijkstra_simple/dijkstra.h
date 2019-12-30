/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
std::vector<float> dijkstra_cpu(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_);
std::vector<float> dijkstra_gpu(int argc, char **argv,
                                std::vector<int> &origins_,
                                std::vector<int> &destinations_);
