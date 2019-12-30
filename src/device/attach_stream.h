/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

namespace cupq {
/**attach to a cuda stream*/
template <typename TT>
typename std::enable_if<is_Vector<typename std::decay<TT>::type>::value>::type
attachStream(Vector<TT, Backend::Cuda> *t, cudaStream_t &stream_) {

#ifdef UNIFMEM
  cudaCheckErrors(cudaStreamAttachMemAsync(stream_, t->deviceData(),
                                           t->size() * sizeof(TT),
                                           cudaMemAttachSingle),
                  __FILE__, __LINE__);
#endif
  // cudaCheckErrors(cudaStreamAttachMemAsync(stream_, t,
  // sizeof(Vector<TT,Backend::Cuda>), cudaMemAttachSingle), __FILE__,
  // __LINE__);
  for (int i = 0; i < t->size(); ++i) {
    attachStream(&(*t)[i], stream_);
#ifdef UNIFMEM
    cudaCheckErrors(cudaStreamAttachMemAsync(stream_, &(*t)[i], sizeof(TT),
                                             cudaMemAttachSingle),
                    __FILE__, __LINE__);
#endif
  }
}

template <typename TT>
typename std::enable_if<!is_Vector<typename std::decay<TT>::type>::value>::type
attachStream(Vector<TT, Backend::Cuda> *t, cudaStream_t &stream_) {
#ifdef UNIFMEM
  cudaCheckErrors(cudaStreamAttachMemAsync(stream_, t->deviceData(),
                                           t->size() * sizeof(TT),
                                           cudaMemAttachSingle),
                  __FILE__, __LINE__);
#endif
  // cudaCheckErrors(cudaStreamAttachMemAsync(stream_, (void*)t,
  // sizeof(Vector<TT,Backend::Cuda>), cudaMemAttachSingle), __FILE__,
  // __LINE__);
}

/**attach to a cuda stream*/
template <typename T1, typename T2>
void attachStream(DevicePair<T1, T2> *t, cudaStream_t &stream_) {
  attachStream(&t->first, stream_);
  attachStream(&t->second, stream_);
}

template <typename TT>
void attachStream(Matrix<TT, Backend::Cuda> *t, cudaStream_t &stream_) {
  attachStream(&t->mVec, stream_);
  // cudaCheckErrors(cudaStreamAttachMemAsync(stream_, t,
  // sizeof(Matrix<TT,Backend::Cuda>), cudaMemAttachSingle), __FILE__,
  // __LINE__);
}

template <typename TT> void attachStream(TT *t, cudaStream_t &stream_) {
  cudaCheckErrors(
      cudaStreamAttachMemAsync(stream_, t, sizeof(TT), cudaMemAttachSingle),
      __FILE__, __LINE__);
}

} // namespace cupq
