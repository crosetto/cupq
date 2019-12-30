/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

namespace cupq {
template <typename, Backend> struct Vector;
template <typename, Backend> struct Matrix;
template <typename> struct is_Vector;
template <typename, typename> struct DevicePair;

template <typename TT>
typename std::enable_if<is_Vector<typename std::decay<TT>::type>::value>::type
attachStream(Vector<TT, Backend::Cuda> *t, cudaStream_t &stream_);
template <typename TT>
typename std::enable_if<!is_Vector<typename std::decay<TT>::type>::value>::type
attachStream(Vector<TT, Backend::Cuda> *t, cudaStream_t &stream_);
template <typename T1, typename T2>
void attachStream(DevicePair<T1, T2> *t, cudaStream_t &stream_);
template <typename TT>
void attachStream(Matrix<TT, Backend::Cuda> *t, cudaStream_t &stream_);
template <typename TT> void attachStream(TT *t, cudaStream_t &stream_);
} // namespace cupq
