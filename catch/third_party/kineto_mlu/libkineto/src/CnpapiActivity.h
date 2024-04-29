// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cnpapi.h>

#include <chrono>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ITraceActivity.h"
#include "GenericTraceActivity.h"
#include "CnpapiActivityPlatform.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "cnpapi_strings.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

template<class T>
static const T* removeTimeGap(const T* activity) {
  static uint64_t time_gap = [] {
    uint64_t cnpapi_time = cnpapiGetTimestamp();
    int64_t time_cpu = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
    return (time_cpu * 1000 - cnpapi_time);
  }();
  T* activity_noconst = const_cast<T*>(activity);
  activity_noconst->start = activity->start + time_gap;
  activity_noconst->end = activity->end + time_gap;
  return activity;
}

// These classes wrap the various CNPAPI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Cnpapi activity type
template<class T>
struct CnpapiActivity : public ITraceActivity {
  explicit CnpapiActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*removeTimeGap(activity)), linked_(linked) {}
  int64_t timestamp() const override {
    return nsToUs(unixEpochTimestamp(activity_.start));
  }
  int64_t duration() const override {
    return nsToUs(activity_.end - activity_.start);
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override {return 0;}
  int32_t getThreadId() const override {return 0;}
  const ITraceActivity* linkedActivity() const override {return linked_;}
  int flowType() const override {return kLinkAsyncCpuMlu;}
  int flowId() const override {return correlationId();}
  const T& raw() const {return activity_;}
  const TraceSpan* traceSpan() const override {return nullptr;}

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// cnpapiActivityAPI - MLU runtime activities
struct RuntimeActivity : public CnpapiActivity<cnpapiActivityAPI> {
  explicit RuntimeActivity(
      const cnpapiActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CnpapiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {return activity_.correlation_id;}
  int64_t deviceId() const override {return processId();}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::MLU_RUNTIME;}
  bool flowStart() const override;
  const std::string name() const override {return runtimeCbidName(activity_.type, activity_.cbid);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// cnpapiActivityAPI - MLU runtime activities
struct OverheadActivity : public CnpapiActivity<cnpapiActivityOverhead> {
  explicit OverheadActivity(
      const cnpapiActivityOverhead* activity,
      const ITraceActivity* linked,
      int32_t threadId=0)
      : CnpapiActivity(activity, linked), threadId_(threadId) {}

  int64_t timestamp() const override {
    return nsToUs(unixEpochTimestamp(activity_.start));
  }
  int64_t duration() const override {
    return nsToUs(activity_.end - activity_.start);
  }
  // TODO: Update this with PID ordering
  int64_t deviceId() const override {return -1;}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::OVERHEAD;}
  bool flowStart() const override;
  const std::string name() const override {return overheadKindString(activity_.overhead_type);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Base class for MLU activities.
// Can also be instantiated directly.
template<class T>
struct MluActivity : public CnpapiActivity<T> {
  explicit MluActivity(const T* activity, const ITraceActivity* linked)
      : CnpapiActivity<T>(activity, linked) {}
  int64_t correlationId() const override {return raw().correlation_id;}
  int64_t deviceId() const override {return raw().device_id;}
  int64_t resourceId() const override {return raw().queue_id;}
  ActivityType type() const override;
  bool flowStart() const override {return false;}
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {return CnpapiActivity<T>::raw();}
};

} // namespace KINETO_NAMESPACE
