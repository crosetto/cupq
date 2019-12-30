/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <cassert>

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
   @class  implementation of a very simple Vector data structure, which
   allocates memory on the CPU and exposes an API to manage to copies to/from
   the device. This class is meant to replace the implementation in @see
   Vector.h when CUDA unified memory is not available or has performance
   limitations.

   The instances are on the host by default, the device memory is allocated on
   demand the first time
   @ref Vector::copyToDevice is called.

   NOTE: the copy constructor does a shallow copy and the distructor does not
   free the memory. Freeing the memory is done by explicitly calling the @ref
   Vector::free member function. A more elegant (i.e. safe) approach for such
   container would be to miplement a copy which does the real copy, and a
   destructor which really frees the memory, and pass this container around
   using the move semantic.
 */
template <typename T, Backend TBackend> struct Vector;

#ifdef __CUDACC__
template <typename T> struct Vector<T, Backend::Cuda> {
  using value_type = T;

  H_F Vector() {}

  H_F Vector(std::size_t size) {
    mSize = size;
    mCapacity = size;
    mDataCPU = (T *)malloc(sizeof(T) * size);
    setOnHost();
  }

  H_F Vector(std::size_t size, T const &init) {
    mSize = size;
    mCapacity = size;
    mDataCPU = (T *)malloc(sizeof(T) * size);
    for (unsigned i = 0; i < size; ++i)
      operator[](i) = init;
    setOnHost();
  }

  Vector &operator=(Vector const &other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData;       // shallow copy
    mDataCPU = other.mDataCPU; // shallow copy
    mDataGPU = other.mDataGPU; // shallow copy
    return *this;
  }

  Vector &operator=(Vector &&other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData;       // shallow copy
    mDataCPU = other.mDataCPU; // shallow copy
    mDataGPU = other.mDataGPU; // shallow copy
    other.setup();
    return *this;
  }

  Vector(Vector &&other) {
    mData = other.mData;
    mDataCPU = other.mDataCPU;
    mDataGPU = other.mDataGPU;
    mSize = other.mSize;
    mCapacity = mSize;
    other.setup();
  };

  /** copy constructor does a shallow copy */
  H_D_F
  Vector(Vector const &other)
      : mSize{other.mSize}, mCapacity{other.mCapacity}, mData{other.mData},
        mDataGPU{other.mDataGPU}, mDataCPU{other.mDataCPU} {}

  template <typename Vec> Vec clone() {
    Vec ret(size());
    for (size_t i = 0; i < mSize; ++i) {
      ret[i] = mData[i];
    }
    return ret;
  }

  bool isOnDevice() { return mData == mDataGPU; }

  bool isOnHost() { return mData == mDataCPU; }

  void setOnDevice() { mData = mDataGPU; }

  void setOnHost() { mData = mDataCPU; }

  void copyToDevice(BackendSpecific<Backend::Cuda> const &backend_) {

    if (!mDataGPU)
      cudaCheckErrors(cudaMalloc(&mDataGPU, sizeof(T) * mSize), __FILE__,
                      __LINE__);

    cudaCheckErrors(cudaMemcpyAsync(mDataGPU, mDataCPU, mSize * sizeof(T),
                                    cudaMemcpyHostToDevice, backend_.mStream),
                    __FILE__, __LINE__);
  }

  // without giving a stream
  void copyToDevice() {

    if (!mDataGPU)
      cudaCheckErrors(cudaMalloc(&mDataGPU, sizeof(T) * mSize), __FILE__,
                      __LINE__);

    cudaCheckErrors(cudaMemcpyAsync(mDataGPU, mDataCPU, mSize * sizeof(T),
                                    cudaMemcpyHostToDevice),
                    __FILE__, __LINE__);
  }

  void copyToHost() {
    cudaCheckErrors(cudaMemcpyAsync(mDataCPU, mDataGPU, mSize * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                    __FILE__, __LINE__);
  }

  /**frees the GPU memory*/
  H_F void free() {
    free(this);
    setup();
  }

  H_D_F
  ~Vector() {}

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
  template <typename TTInt> H_D_F T &operator[](TTInt id) {
#ifndef NDEBUG
    assert(id < size());
#endif
    return mData[id];
  }

  /** returns id-th element of the Vector as const reference*/
  template <typename TTInt> H_D_F T const &operator[](TTInt id) const {
#ifndef NDEBUG
    assert(id < size());
#endif
    return mData[id];
  }

  /** allocates space for elements without changing the size of the Vector*/
  H_F void reserve(size_t size) {
    free();
    mCapacity = size;
    mSize = 0;
    if (size != 0) {
      mDataCPU = (T *)malloc(sizeof(T) * size);
    }
    setOnHost();
  }

  /** changes the size of the Vector, and allocates memory correspondingly */
  H_F void resize(size_t size) {
    if (mSize == size)
      return;
    reserve(size);
    mSize = size;
  }

  /** emplace an element at the end of the Vector, performing a memory
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
    assert(isOnHost());
    if (mSize < mCapacity) {
      mData[mSize] = std::forward<TT>(val);
      ++mSize;
    } else {
      mCapacity = 1 + 2 * mSize;
      T *oldData2 = mDataCPU;
      mDataCPU = (T *)malloc(sizeof(T) * mCapacity);
      for (std::size_t i = 0; i < mSize; ++i)
        mDataCPU[i] = oldData2[i];
      mDataCPU[mSize] = std::forward<TT>(val);
      ++mSize;
      if (oldData2 != nullptr)
        ::free(oldData2);
    }
    setOnHost();
  }

  H_D_F
  T *deviceData() { return mDataGPU; }

private:
  /**frees a Vector of vectors recursively*/
  template <typename TT>
  H_F typename std::enable_if<
      is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Cuda> *t) {
    // freeing Vector of vectors recursively
    for (unsigned i = 0; i < mSize; ++i)
      mData[i].free();
  }

  /**frees the GPU memory*/
  template <typename TT>
  H_F typename std::enable_if<
      !is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Cuda> *t) {
    // std::cout<<"freeing Vector\n";
    if (mDataGPU)
      cudaCheckErrors(cudaFree(mDataGPU), __FILE__, __LINE__);
    if (mDataCPU)
      ::free(mDataCPU);

    mData = nullptr;
    mDataGPU = nullptr;
    mDataCPU = nullptr;
  }

  T *mDataCPU = nullptr;
  T *mDataGPU = nullptr;
  T *mData = nullptr;
  size_t mSize = 0u;
  size_t mCapacity = 1u;
};
#endif // __CUDACC__

/** The same implementation for the host*/
template <typename T> struct Vector<T, Backend::Host> {
  using value_type = T;

  H_F Vector() { setup(); }

  H_F Vector(std::size_t size) {
    mSize = size;
    mCapacity = size;
    mData = (T *)malloc(sizeof(T) * size);
  }

  H_F Vector(std::size_t size, T const &init) {
    mSize = size;
    mCapacity = size;
    mData = (T *)malloc(sizeof(T) * size);
    for (unsigned i = 0; i < size; ++i)
      operator[](i) = init;
  }

  Vector &operator=(Vector const &other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData; // shallow copy
    return *this;
  }

  Vector &operator=(Vector &&other) {
    mSize = other.mSize;
    mCapacity = other.mCapacity;
    mData = other.mData; // shallow copy
    other.setup();
    return *this;
  }

  Vector(Vector &&other) {
    mData = other.mData;
    mSize = other.mSize;
    mCapacity = mSize;
    other.setup();
  };

  /**frees the GPU memory*/
  H_F void free() {
    free(this);
    setup();
  }

  H_D_F
  ~Vector() {}

  /** copy constructor does a shallow copy */
  H_D_F
  Vector(Vector const &other)
      : mSize{other.mSize}, mCapacity{other.mCapacity}, mData{other.mData} {}

  /** reset the data members */
  H_D_F
  void setup() {
    mData = nullptr;
    mSize = 0u;
    mCapacity = 0u;
  }

  H_D_F
  size_t const &size() const { return mSize; }

  H_D_F
  T *begin() { return &operator[](0); }

  H_D_F
  T *end() { return begin() + size(); }

  template <typename TTInt> H_D_F T &operator[](TTInt id) {
    assert(id < size());
    return mData[id];
  }

  template <typename TTInt> H_D_F T const &operator[](TTInt id) const {
    assert(id < size());
    return mData[id];
  }

  H_F void reserve(size_t size) {
    free();
    mCapacity = size;
    mSize = 0;
    if (size != 0) {
      mData = (T *)malloc(sizeof(T) * size);
    }
  }

  H_F void resize(size_t size) {
    if (mSize == size)
      return;
    reserve(size);
    mSize = size;
  }

  H_F void push_back(T &&val) {
    if (mSize < mCapacity) {
      mData[mSize] = std::forward<T>(val);
      ++mSize;
    } else {
      mCapacity = 1 + 2 * mSize;
      T *oldData = mData;
      mData = (T *)malloc(sizeof(T) * mCapacity);
      for (std::size_t i = 0; i < mSize; ++i)
        mData[i] = oldData[i];
      mData[mSize] = std::forward<T>(val);
      ++mSize;
      if (oldData != nullptr)
        ::free(oldData);
    }
  }

  H_F void push_back(T const &val) {
    if (mSize < mCapacity) {
      mData[mSize] = val;
      ++mSize;
    } else {
      mCapacity = 1 + 2 * mSize;
      T *oldData = mData;
      mData = (T *)malloc(sizeof(T) * mCapacity);
      for (std::size_t i = 0; i < mSize; ++i)
        mData[i] = oldData[i];
      mData[mSize] = val;
      ++mSize;
      if (oldData != nullptr) {
        ::free(oldData);
      }
    }
  }

  H_F void setOnHost() {}
  H_F void setOnDevice() {}
  H_F void copyToDevice(BackendSpecific<Backend::Host> const &) {}
  H_F void copyToDevice() {}
  H_F void copyToHost() {}

private:
  /**frees a Vector of vectors recursively*/
  template <typename TT>
  H_F typename std::enable_if<
      is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Host> *t) {
    for (unsigned i = 0; i < mSize; ++i)
      mData[i].free();
    ::free(mData);
    mData = nullptr;
  }

  /**frees the GPU memory*/
  template <typename TT>
  H_F typename std::enable_if<
      !is_Vector<typename std::decay<TT>::type>::value>::type
  free(Vector<TT, Backend::Host> *t) {
    ::free(mData);
    mData = nullptr;
  }

private:
  T *mData = nullptr;
  size_t mSize = 0u;
  size_t mCapacity = 1u;
};

template <typename T, Backend TBackend>
struct is_Vector<Vector<T, TBackend>> : std::true_type {};

} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
