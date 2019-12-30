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

/**@class simple DevicePair data structure

   simple replacememt of std::pair for the GPU
 */
template <typename T1, typename T2> struct ALIGN DevicePair {
  T1 first;
  T2 second;

  H_D_F
  DevicePair<T1, T2> &operator=(DevicePair<T1, T2> const &other_) {
    first = other_.first;
    second = other_.second;
    return *this;
  }

  H_D_F
  DevicePair<T1, T2> &operator=(DevicePair<T1, T2> &&other_) {
    first = std::move(other_.first);
    second = std::move(other_.second);
    return *this;
  }

  H_F DevicePair() : first(), second() {}

  H_D_F
  DevicePair(DevicePair<T1, T2> const &other_)
      : first(other_.first), second(other_.second) {}

  H_D_F
  DevicePair(DevicePair<T1, T2> &&other_)
      : first(std::move(other_.first)), second(std::move(other_.second)) {}

  H_D_F
  DevicePair(T1 const &t1, T2 const &t2) : first(t1), second(t2) {}

  template <
      typename = std::enable_if<is_Vector<T1>::value && is_Vector<T2>::value>>
  H_F void free() {
    first.free();
    second.free();
  }
};

template <typename T1, typename T2>
H_D_F DevicePair<T1, T2> make_DevicePair(T1 const &t1, T2 const &t2) {
  return DevicePair<T1, T2>(t1, t2);
}

template <typename T> struct is_DevicePair : std::false_type {};

template <typename T1, typename T2>
struct is_DevicePair<DevicePair<T1, T2>> : std::true_type {};

} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
