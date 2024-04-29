 /*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CnpapiActivity.h"

#include <fmt/format.h>

#include "MluDeviceProperties.h"
#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

template<>
inline const std::string MluActivity<cnpapiActivityKernel>::name() const {
  return demangle(raw().name);
}

template<>
inline ActivityType MluActivity<cnpapiActivityKernel>::type() const {
  return ActivityType::MLU_CONCURRENT_KERNEL;
}

template<class T>
inline void MluActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

template<>
inline const std::string MluActivity<cnpapiActivityKernel>::metadataJson() const {
  const cnpapiActivityKernel& kernel = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "kernel type": "{}",
      "dimx": {}, "dimy": {}, "dimz": {},
      "tasktopo": {},
      "tasktopo_node": {})JSON",
      us(kernel.queued), kernel.device_id, kernel.context_id,
      kernel.queue_id, kernel.correlation_id,
      kernelTypeString(kernel.kernel_type),
      kernel.dimx, kernel.dimy, kernel.dimz,
      kernel.tasktopo_id,
      kernel.tasktopo_node_id);
  // clang-format on
}


inline std::string memcpyName(uint64_t kind) {
  return fmt::format(
      "Memcpy {}",
      memcpyKindString((cnpapiActivityMemcpyType)kind));
}

template<>
inline ActivityType MluActivity<cnpapiActivityMemcpy>::type() const {
  return ActivityType::MLU_MEMCPY;
}

template<>
inline const std::string MluActivity<cnpapiActivityMemcpy>::name() const {
  return memcpyName(raw().copy_type);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template<>
inline const std::string MluActivity<cnpapiActivityMemcpy>::metadataJson() const {
  const cnpapiActivityMemcpy& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.device_id, memcpy.context_id,
      memcpy.queue_id, memcpy.correlation_id,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}


template<>
inline ActivityType MluActivity<cnpapiActivityMemcpyPtoP>::type() const {
  return ActivityType::MLU_MEMCPY;
}

template<>
inline const std::string MluActivity<cnpapiActivityMemcpyPtoP>::name() const {
  return memcpyName(raw().copy_type);
}

template<>
inline const std::string MluActivity<cnpapiActivityMemcpyPtoP>::metadataJson() const {
  const cnpapiActivityMemcpyPtoP& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "Context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.src_device_id, memcpy.device_id, memcpy.dst_device_id,
      memcpy.context_id,
      memcpy.queue_id, memcpy.correlation_id,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}

template<>
inline const std::string MluActivity<cnpapiActivityMemset>::name() const {
  return fmt::format("Memset");
}

template<>
inline ActivityType MluActivity<cnpapiActivityMemset>::type() const {
  return ActivityType::MLU_MEMSET;
}

template<>
inline const std::string MluActivity<cnpapiActivityMemset>::metadataJson() const {
  const cnpapiActivityMemset& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memset.device_id, memset.context_id,
      memset.queue_id, memset.correlation_id,
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
  // clang-format on
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void OverheadActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline bool OverheadActivity::flowStart() const {
  return false;
}

inline const std::string OverheadActivity::metadataJson() const {
  return "";
}

inline bool RuntimeActivity::flowStart() const {
  return activity_.cbid == CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel ||
      (activity_.cbid >= CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync &&
       activity_.cbid <= CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoDAsync);
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(R"JSON("cbid": {}, "correlation": {})JSON",
      activity_.cbid, activity_.correlation_id);
}

template<class T>
inline const std::string MluActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
