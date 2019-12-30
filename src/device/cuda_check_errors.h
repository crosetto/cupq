/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <iostream>
namespace cupq {
template <typename T>
void cudaCheckErrors(T err, const char *file = "no_file", const int line = -1) {
#ifdef __CUDACC__
#ifndef NDEBUG
  if (err != cudaSuccess)
    std::cout << "cuda error: " << cudaGetErrorString(err) << ", " << file
              << ", " << line << "\n";
#endif
#endif
}
} // namespace cupq
