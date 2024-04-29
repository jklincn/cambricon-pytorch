// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>

#ifdef HAS_CNPAPI
#include <cnpapi.h>
#endif

#include "cnpapi_call.h"
#include "ActivityType.h"
#include "CnpapiActivityBuffer.h"


namespace KINETO_NAMESPACE {

using namespace libkineto;

#ifndef HAS_CNPAPI
using cnpapiActivity = void;
#endif

class CnpapiActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  CnpapiActivityApi() {
#ifdef HAS_CNPAPI
    CNPAPI_CALL(cnpapiInit());
#endif
  }
  CnpapiActivityApi(const CnpapiActivityApi&) = delete;
  CnpapiActivityApi& operator=(const CnpapiActivityApi&) = delete;

  virtual ~CnpapiActivityApi() {
#ifdef HAS_CNPAPI
    CNPAPI_CALL(cnpapiRelease());
#endif
  }

  static CnpapiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCnpapiActivities(
    const std::set<ActivityType>& selected_activities);
  void disableCnpapiActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<CnpapiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      CnpapiActivityBufferMap&,
      std::function<void(const cnpapiActivity*)> handler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
#ifdef HAS_CNPAPI
  int processActivitiesForBuffer(
      uint64_t* buf,
      size_t validSize,
      std::function<void(const cnpapiActivity*)> handler);
  static void
  bufferRequestedTrampoline(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  static void bufferCompletedTrampoline(
      uint64_t* buffer,
      size_t size,
      size_t validSize);
  cnpapi_SubscriberHandle subscriber_;
  static void emptyCallback(void *userdata, cnpapi_CallbackDomain domain,
                     cnpapi_CallbackId cbid, const cnpapi_CallbackData *cbdata) {}
#endif // HAS_CNPAPI

  int maxMluBufferCount_{0};
  CnpapiActivityBufferMap allocatedMluTraceBuffers_;
  std::unique_ptr<CnpapiActivityBufferMap> readyMluTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};

 protected:
#ifdef HAS_CNPAPI
  void bufferRequested(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(
      uint64_t* buffer,
      size_t size,
      size_t validSize);
#endif
};

} // namespace KINETO_NAMESPACE
