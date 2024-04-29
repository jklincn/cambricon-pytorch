// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CnpapiActivityApi.h"

#include <assert.h>
#include <chrono>

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

// TODO: do we want this to be configurable?
// Set to 2MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
// Consider putting this on huge pages?
constexpr size_t kBufSize(2 * 1024 * 1024);

CnpapiActivityApi& CnpapiActivityApi::singleton() {
  static CnpapiActivityApi instance;
  return instance;
}

void CnpapiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_CNPAPI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  VLOG(2) << "pushCorrelationID(" << id << ")";
  switch(type) {
    case Default:
      CNPAPI_CALL(cnpapiActivityPushExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, id));
        break;
    case User:
      CNPAPI_CALL(cnpapiActivityPushExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, id));
  }
#endif
}

void CnpapiActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_CNPAPI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch(type) {
    case Default:
      CNPAPI_CALL(cnpapiActivityPopExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, nullptr));
        break;
    case User:
      CNPAPI_CALL(cnpapiActivityPopExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, nullptr));
  }
#endif
}

static bool nextActivityRecord(
    uint64_t* buffer,
    size_t valid_size,
    cnpapiActivity*& record) {
#ifdef HAS_CNPAPI
  cnpapiResult status = CNPAPI_CALL_NOWARN(
      cnpapiActivityGetNextRecord(buffer, valid_size, &record));
  if (status != CNPAPI_SUCCESS) {
    if (status != CNPAPI_ERROR_MAX_LIMIT_REACHED) {
      CNPAPI_CALL(status);
    }
    record = nullptr;
  }
#endif
  return record != nullptr;
}

void CnpapiActivityApi::setMaxBufferSize(int size) {
  maxMluBufferCount_ = 1 + size / kBufSize;
}

#ifdef HAS_CNPAPI
void CnpapiActivityApi::bufferRequestedTrampoline(
    uint64_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void CnpapiActivityApi::bufferRequested(
    uint64_t** buffer, size_t* size, size_t* maxNumRecords) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedMluTraceBuffers_.size() >= maxMluBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max MLU buffer count ("
                 << allocatedMluTraceBuffers_.size()
                 << " > " << maxMluBufferCount_
                 << ") - terminating tracing";
  }

  auto buf = std::make_unique<CnpapiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedMluTraceBuffers_[*buffer] = std::move(buf);

  *maxNumRecords = 0;
}
#endif

std::unique_ptr<CnpapiActivityBufferMap>
CnpapiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedMluTraceBuffers_.empty()) {
      return nullptr;
    }
  }

#ifdef HAS_CNPAPI
  VLOG(1) << "Flushing MLU activity buffers";
  time_point<system_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = system_clock::now();
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcnpapi and mutex_ is acquired there.
  CNPAPI_CALL(cnpapiActivityFlushAll());
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(system_clock::now() - t1).count();
  }
#endif
  std::lock_guard<std::mutex> guard(mutex_);
  // Transfer ownership of buffers to caller. A new map is created on-demand.
  return std::move(readyMluTraceBuffers_);
}

#ifdef HAS_CNPAPI
int CnpapiActivityApi::processActivitiesForBuffer(
    uint64_t* buf,
    size_t validSize,
    std::function<void(const cnpapiActivity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    cnpapiActivity* record{nullptr};
    while ((nextActivityRecord(buf, validSize, record))) {
      handler(record);
      ++count;
    }
  }
  return count;
}
#endif

const std::pair<int, int> CnpapiActivityApi::processActivities(
    CnpapiActivityBufferMap& buffers,
    std::function<void(const cnpapiActivity*)> handler) {
  std::pair<int, int> res{0, 0};
#ifdef HAS_CNPAPI
  for (auto& pair : buffers) {
    // No lock needed - only accessed from this thread
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
#endif
  return res;
}

void CnpapiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedMluTraceBuffers_.empty()) {
      return;
    }
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcnpapi and mutex_ is acquired there.
#ifdef HAS_CNPAPI
  CNPAPI_CALL(cnpapiActivityFlushAll());
#endif
  // FIXME: We might want to make sure we reuse
  // the same memory during warmup and tracing.
  // Also, try to use the amount of memory required
  // for active tracing during warmup.
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  readyMluTraceBuffers_ = nullptr;
}

#ifdef HAS_CNPAPI
void CnpapiActivityApi::bufferCompletedTrampoline(
    uint64_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void CnpapiActivityApi::bufferCompleted(
    uint64_t* buffer,
    size_t size,
    size_t validSize) {

  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedMluTraceBuffers_.find(buffer);
  if (it == allocatedMluTraceBuffers_.end()) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << (void*) buffer;
    return;
  }

  if (!readyMluTraceBuffers_) {
    readyMluTraceBuffers_ = std::make_unique<CnpapiActivityBufferMap>();
  }
  // Set valid size of buffer before moving to ready map
  it->second->setSize(validSize);
  (*readyMluTraceBuffers_)[it->first] = std::move(it->second);
  allocatedMluTraceBuffers_.erase(it);
}

namespace {

const cnpapi_CallbackIdCNRT enabledCnrtCbidList[] = {
  CNPAPI_CNRT_TRACE_CBID_cnrtMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocHost,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemset,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeer,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtSyncDevice,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeerAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtHostMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueCreate,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueDestroy,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueQuery,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueSync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy2D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy3D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync_V2
};

const cnpapi_CallbackIdCNDRV enabledCndrvCbidList[] = {
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeer,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoH,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeerAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32Async,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD3D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy3D,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeHostFunc,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync_V2
};

const cnpapi_CallbackIdCNNL disabledCnnlCbidList[] = {
  CNPAPI_CNNL_TRACE_CBID_cnnlSetQueue,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetQueue,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateActivationDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyActivationDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetActivationDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorPosition,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorPositionAndScale,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetTensorDescriptorPosition,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetTensorDescriptorPositionAndScale,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetConvolutionDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateConvolutionDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyConvolutionDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateOpTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyOpTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetOpTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetOpTensorDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetOpTensorWorkspaceSize,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateReduceDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetReduceDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyReduceDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateTransposeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyTransposeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTransposeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorEx,
  CNPAPI_CNNL_TRACE_CBID_cnnlCreateNormalizeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetNormalizeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlDestroyNormalizeDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetActivationDescriptor_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetActivationDescriptor_v5,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetActivationDescriptor_v6,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetActivationDescriptor,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptor_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetSeqDataDescriptor_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorEx_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorDim_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetTensorDescriptorEx_v2,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetSizeOfDataType,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetConvolutionDescriptorAllowTF32,
  CNPAPI_CNNL_TRACE_CBID_cnnlSetTensorDescriptorOnchipDataType,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetBatchNormForwardWorkspaceSize,
  CNPAPI_CNNL_TRACE_CBID_cnnlGetBatchNormBackwardWorkspaceSize,
};

}
#endif

void CnpapiActivityApi::enableCnpapiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_CNPAPI
  static bool registered = false;
  if (!registered) {
    CNPAPI_CALL(
        cnpapiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline));
    registered = true;
  }

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::MLU_MEMCPY) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    }
    if (activity == ActivityType::MLU_MEMSET) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == ActivityType::MLU_CONCURRENT_KERNEL) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::MLU_RUNTIME) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CALLBACK_API));
      CNPAPI_CALL(cnpapiSubscribe(&subscriber_, (cnpapi_CallbackFunc)emptyCallback, nullptr));
      CNPAPI_CALL(cnpapiEnableDomain(1, subscriber_, CNPAPI_CB_DOMAIN_CNNL_API));
      CNPAPI_CALL(cnpapiEnableDomain(1, subscriber_, CNPAPI_CB_DOMAIN_CNNL_EXTRA_API));
      CNPAPI_CALL(cnpapiEnableDomain(1, subscriber_, CNPAPI_CB_DOMAIN_CNCL_API));
      for (const auto& cbid : enabledCnrtCbidList) {
        CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, CNPAPI_CB_DOMAIN_CNRT_API, cbid));
      }
      for (const auto& cbid : enabledCndrvCbidList) {
        CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, CNPAPI_CB_DOMAIN_CNDRV_API, cbid));
      }
      for (const auto& cbid : disabledCnnlCbidList) {
        CNPAPI_CALL(cnpapiEnableCallback(0, subscriber_, CNPAPI_CB_DOMAIN_CNNL_API, cbid));
      }
    }
    if (activity == ActivityType::OVERHEAD) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
#endif

  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void CnpapiActivityApi::disableCnpapiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_CNPAPI
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::MLU_MEMCPY) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    }
    if (activity == ActivityType::MLU_MEMSET) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == ActivityType::MLU_CONCURRENT_KERNEL) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::MLU_RUNTIME) {
      CNPAPI_CALL(cnpapiUnsubscribe(subscriber_));
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CALLBACK_API));
    }
    if (activity == ActivityType::OVERHEAD) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

} // namespace KINETO_NAMESPACE
