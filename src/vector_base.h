/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

#include <cstring>

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

template <typename T> struct is_Vector : std::false_type {};

/**
   @class  implementation of a very simple vector data structure, which
   allocates pinned memory. NOTE: the copy constructor does a shallow copy and
   the distructor does not free the memory. Freeing the memory is done by
   explicitly calling the free member function. The proper design of
   such container would be implementing a copy which does the real copy, and a
   destructor which really frees the memory, and pass this container around
   using the move semantic. This will probably be possible in future releases of
   CUDA.
 */
template <typename T, Backend TBackend> struct Vector;

template <typename T> struct VectorBase {
  using value_type = T;

  H_D_F
  VectorBase() {}

  H_F VectorBase(std::size_t size) {
    mSize = size;
    mCapacity = size;
    cudaCheckErrors(cudaMallocManaged(&mData, sizeof(T) * size), __FILE__,
                    __LINE__);
  }

  H_F VectorBase(std::size_t size, T const &init) {
    mSize = size;
    mCapacity = size;
    cudaCheckErrors(cudaMallocManaged(&mData, sizeof(T) * size), __FILE__,
                    __LINE__);
    for (unsigned i = 0; i < size; ++i)
      operator[](i) = init;
  }

  H_D_F
  VectorBase &operator=(VectorBase const &other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData; // shallow copy
    return *this;
  }

  H_D_F
  VectorBase(VectorBase const &other) {
    mData = other.mData;
    mSize = other.mSize;
    mCapacity = other.mCapacity;
  };

  H_D_F
  VectorBase &operator=(VectorBase &&other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData; // shallow copy
    other.setup();
    return *this;
  }

  H_D_F
  VectorBase(VectorBase &&other) {
    mData = other.mData;
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    other.setup();
  };

  H_D_F
  ~VectorBase() {}

  template <typename Vec> H_D_F Vec clone() {
    Vec ret(size());
    for (size_t i = 0; i < mSize; ++i) {
      ret[i] = mData[i];
    }
    return ret;
  }

  /** reset the data members */
  H_D_F
  void setup() {
    mData = nullptr;
    mSize = 0u;
    mCapacity = 0u;
  }

  H_D_F
  size_t const &size() const { return mSize; }

  /** returns a pointer to the first element of the Vector */
  H_D_F
  T *begin() const { return &mData[0]; }

  /** returns a pointer to the one past the last element of the Vector */
  H_D_F
  T *end() const { return begin() + size(); }

  /** returns a non constant reference to the id-th element of the Vector */
  template <typename TTInt> H_D_F T &operator[](TTInt id) { return mData[id]; }

  /** returns id-th element of the Vector as const reference*/
  template <typename TTInt> H_D_F T const &operator[](TTInt id) const {
    return mData[id];
  }

  H_D_F
  T *deviceData() { return mData; }

  // legacy methods
  bool isOnDevice() { return true; }
  bool isOnHost() { return true; }
  void setOnDevice() {}
  void setOnHost() {}
  template <typename TT> void copyToDevice(TT const &) {}
  void copyToHost() {}

protected:
  T *mData = nullptr;
  size_t mSize = 0u;
  size_t mCapacity = 1u;
};
} // namespace cupq
