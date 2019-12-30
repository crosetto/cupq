/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <iostream>
namespace cupq {
void cudaCheckMemory() {
#ifdef __CUDACC__
  size_t free_byte;
  size_t total_byte;
  cudaCheckErrors(cudaMemGetInfo(&free_byte, &total_byte), __FILE__, __LINE__);
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
#endif
}
} // namespace cupq
