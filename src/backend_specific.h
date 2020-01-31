/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
namespace cupq {
template <Backend TBackend> struct BackendSpecific;

template <> struct BackendSpecific<Backend::Host> {
  auto static constexpr const backend_t=Backend::Host;
  template <typename TT> void attachStream(TT &sources_) {}
  void attachStreamErr() {}
  void init() {}
  void fin() {}
  void free() {}
  void sync() const {}
};
} // namespace cupq
