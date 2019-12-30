/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/

namespace cupq {

template <> struct BackendSpecific<Backend::Cuda> {
  template <typename TT> void attachStream(TT &sources_) {
    return ::cupq::attachStream(&sources_, mStream);
  }

  void attachStreamErr() { return ::cupq::attachStream(mErr, mStream); }

  void init() {
    cudaSetDevice(0);

    cudaCheckErrors(cudaStreamCreate(&mStream));
    cudaCheckErrors(cudaMallocManaged(&mErr, sizeof(cudaError_t)));
#ifdef CUSP_PROFILE_HEAP
    cudaCheckErrors(cudaProfilerInitialize(
        "profile_heap.conf", "profile_heap.out", cudaKeyValuePair));
#endif
  }

  void sync() const { cudaStreamSynchronize(mStream); }

  void fin() {
    sync();

    if (*mErr != cudaSuccess) {
      if (*mErr == cudaErrorMemoryAllocation)
        std::cout << "You seem to be running a pretty big network, there's a "
                     "memory allocation problem in the solution of the "
                     "shortest path algorithm. Contact with the support to get "
                     "the default memory limit extended\n";
      else
        std::cout << "Error in CUDA kernel launch (shortest path):\n";
      std::cout << cudaGetErrorString(*mErr) << "\n";
    }
  }

  void free() {
    if (mStream)
      cudaStreamDestroy(mStream);
    if (mErr)
      cudaFree(mErr);
  }

  cudaStream_t mStream;
  cudaError_t *mErr = nullptr;
};
} // namespace cupq
