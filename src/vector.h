/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

#include "vector_base.h"
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

#ifdef __CUDACC__
template <typename T> struct Vector<T, Backend::Cuda> : public VectorBase<T> {
  using super = VectorBase<T>;
  using value_type = typename super::value_type;
  using super::VectorBase;
  using super::operator=;

  /**frees the GPU memory*/
  H_F void free() {
    free(this);
    super::setup();
  }

  /** changes the size of the Vector, and allocates memory correspondingly */
  H_F void resize(size_t size) {
    if (this->mSize == size)
      return;
    reserve(size);
    this->mSize = size;
  }

  /** allocates space for elements, and sets the size of the Vector to zero*/
  H_F void reserve(size_t size) {
    free();
    this->mCapacity = size;
    this->mSize = 0;
    if (size != 0) {
      cudaCheckErrors(cudaMallocManaged(&this->mData, sizeof(T) * size),
                      __FILE__, __LINE__);
    }
  }

  /** puts an element at the end of the Vector, performing a memory
     reallocation (of the whole Vector) when the size exceeds the capacity

          the algorithm for memory reallocation doubles the capacity of the
     Vector each time the size and capacity values match. NOTE: using forwarding
     reference allows the same member function to deal with the rvalue and
     lvalue references at once
  */
  template <typename TT,
            typename = std::enable_if_t<std::is_same<
                T, typename std::remove_reference<TT>::type>::value>>
  H_F void push_back(TT &&val) {
    if (this->mSize < this->mCapacity) {
      this->mData[this->mSize] = std::forward<TT>(val);
      ++this->mSize;
    } else {
      this->mCapacity = 1 + 2 * this->mSize;
      T *oldData = this->mData;
      cudaCheckErrors(
          cudaMallocManaged(&this->mData, sizeof(T) * this->mCapacity),
          __FILE__, __LINE__);
      for (std::size_t i = 0; i < this->mSize; ++i)
        this->mData[i] = oldData[i];
      this->mData[this->mSize] = std::forward<TT>(val);
      ++this->mSize;
      if (oldData != nullptr)
        cudaCheckErrors(cudaFree(oldData), __FILE__, __LINE__);
    }
  }

private:
  /**frees a Vector of vectors recursively*/
  template <typename TT>
  H_F typename std::enable_if<
      is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Cuda> *t) {
    // std::cout<<"freeing Vector of vectors recursively\n";
    for (unsigned i = 0; i < this->mSize; ++i)
      this->mData[i].free();
    cudaCheckErrors(cudaFree(this->mData), __FILE__, __LINE__);
    this->mData = nullptr;
  }

  /**frees the memory*/
  template <typename TT>
  H_F typename std::enable_if<
      !is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Cuda> *t) {
    // std::cout<<"freeing Vector\n";
    if (this->mData)
      cudaCheckErrors(cudaFree(this->mData), __FILE__, __LINE__);
    this->mData = nullptr;
  }
};
#endif // __CUDACC__

/** Host implementation of the Vector data structure*/
template <typename T> struct Vector<T, Backend::Host> : public VectorBase<T> {
  using super = VectorBase<T>;
  using value_type = typename super::value_type;
  using super::VectorBase;
  using super::operator=;

  /**frees the host memory*/
  H_F void free() {
    free(this);
    super::setup();
  }

  /** chganges the size of the Vector, and allocates memory correspondingly */
  H_F void resize(size_t size) {
    if (this->mSize == size)
      return;
    reserve(size);
    this->mSize = size;
  }

  H_F void reserve(size_t size) {
    this->mCapacity = size;
    T *oldData = this->mData;
    this->mData = (T *)malloc(sizeof(T) * size);
    if (oldData != nullptr) {
      ::std::memcpy(this->mData, oldData, this->mSize);
      ::free(oldData);
    }
  }

  H_F void push_back(T &&val) {
    if (this->mSize < this->mCapacity) {
      this->mData[this->mSize] = std::forward<T>(val);
      ++this->mSize;
    } else {
      this->mCapacity = 1 + 2 * this->mSize;
      T *oldData = this->mData;
      this->mData = (T *)malloc(sizeof(T) * this->mCapacity);
      for (std::size_t i = 0; i < this->mSize; ++i)
        this->mData[i] = oldData[i];
      this->mData[this->mSize] = std::forward<T>(val);
      ++this->mSize;
      if (oldData != nullptr)
        ::free(oldData);
    }
  }

  H_F void push_back(T const &val) {
    if (this->mSize < this->mCapacity) {
      this->mData[this->mSize] = val;
      ++this->mSize;
    } else {
      this->mCapacity = 1 + 2 * this->mSize;
      T *oldData = this->mData;
      this->mData = (T *)malloc(sizeof(T) * this->mCapacity);
      for (std::size_t i = 0; i < this->mSize; ++i)
        this->mData[i] = oldData[i];
      this->mData[this->mSize] = val;
      ++this->mSize;
      if (oldData != nullptr) {
        ::free(oldData);
      }
    }
  }

private:
  /**frees a Vector of vectors recursively*/
  template <typename TT>
  H_F typename std::enable_if<
      is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Host> *t) {
    for (unsigned i = 0; i < this->mSize; ++i)
      this->mData[i].free();
    ::free(this->mData);
    this->mData = nullptr;
  }

  /**frees the GPU memory*/
  template <typename TT>
  H_F typename std::enable_if<
      !is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Host> *t) {
    ::free(this->mData);
    this->mData = nullptr;
  }
};

template <typename T, Backend B>
struct is_Vector<Vector<T, B>> : std::true_type {};

} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
