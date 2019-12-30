/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

#ifdef __CUDACC__
#define H_F __host__
#define D_F __device__
#define H_D_F __host__ __device__
#define ALIGN //__align__(8)
#else
#define H_F
#define D_F
#define H_D_F
#define ALIGN //__align__(8)
#endif

namespace cupq {
template <typename T> struct is_Array : std::false_type {};

template <typename T, std::size_t N> struct ALIGN Array {
  T value[N];

  H_D_F
  Array() {}

  H_D_F
  auto &operator[](std::size_t id) { return value[id]; }

  H_D_F
  auto const &operator[](std::size_t id) const { return value[id]; }
};

template <typename T, std::size_t N>
struct is_Array<Array<T, N>> : std::true_type {};

} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
