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

template <typename T> struct is_Matrix : std::false_type {};

/**
   @class implementation of a very simple 2D array data structure.
 */
template <typename T, Backend TBackend> struct Matrix {
  using value_type = T;

  H_F Matrix() {}

  H_F Matrix(std::size_t size_, std::size_t size2_)
      : mCols(size2_), mVec(size_ * size2_) {}

  H_F Matrix(std::size_t size_, std::size_t size2_, T const &init_)
      : mCols(size2_), mVec(size_ * size2_, init_) {}

  Matrix &operator=(Matrix const &other) {
    mCols = other.mCols;
    mVec = other.mCVec;
    return *this;
  }

  Matrix &operator=(Matrix &&other) {
    mCols = std::move(other.mCols);
    mVec = std::move(other.mVec);
    return *this;
  }

  /** concept: Mat has an operator()(index_t, index_t) */
  template <typename Mat> Mat clone() {
    Mat ret(mVec.size() / mCols, mCols);
    for (size_t i = 0; i < mVec.size() / mCols; ++i) {
      for (size_t j = 0; j < mCols; ++j) {
        ret(i, j) = operator()(i, j);
      }
    }
    return ret;
  }

  H_F void free() {
    mVec.free();
    setup();
  }

  H_D_F
  ~Matrix() {}

  /** copy constructor does a shallow copy */
  H_D_F
  Matrix(Matrix const &other) : mCols{other.mCols}, mVec{other.mVec} {}

  /** reset the data members */
  H_D_F
  void setup() {
    mCols = 0u;
    mVec.setup();
  }

  H_D_F
  T *begin() { return mVec.begin(); }

  H_D_F
  T *end() { return mVec.end(); }

  template <typename TTInt1, typename TTInt2,
            typename = std::enable_if_t<std::is_integral<TTInt1>::value>,
            typename = std::enable_if_t<std::is_integral<TTInt2>::value>>
  H_D_F T &operator()(TTInt1 id, TTInt2 id2) {
    return mVec[id * mCols + id2];
  }

  template <typename TTInt1, typename TTInt2,
            typename = std::enable_if_t<std::is_integral<TTInt1>::value>,
            typename = std::enable_if_t<std::is_integral<TTInt2>::value>>
  H_D_F T const &operator()(TTInt1 id, TTInt2 id2) const {
    return mVec[id * mCols + id2];
  }

  H_F void reserve(size_t size, size_t size2) {
    mVec.reserve(size * size2);
    mCols = 0;
  }

  H_F void resize(size_t size, size_t size2) {
    mVec.resize(size * size2);
    mCols = size2;
  }

  H_D_F
  auto ncols() const { return mCols; }

  H_D_F
  auto nrows() const { return mCols ? mVec.size() / mCols : 0; }

  // legacy methods
  bool isOnDevice() { return mVec.isOnDevice(); }
  bool isOnHost() { return mVec.isOnHost(); }
  void setOnDevice() { mVec.setOnDevice(); }
  void setOnHost() { mVec.setOnHost(); }
  void copyToDevice(BackendSpecific<TBackend> const &backend_) {
    mVec.copyToDevice(backend_);
  }
  void copyToHost() { mVec.copyToHost(); }

public:
  Vector<T, TBackend> mVec;
  unsigned mCols = 0u;
};

template <typename TT, Backend B>
struct is_Matrix<Matrix<TT, B>> : std::true_type {};

} // namespace cupq

#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
