#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <caffe2/core/logging.h>

#include "framework/core/device.h"
#include "framework/core/queue.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

TEST(QueuePoolTest, getCurrentQueue) {
  const int cid = 0;
  setDevice(cid);
  for (int dev = 0; dev < device_count(); ++dev) {
    auto queue = getQueueFromPool(false, dev);
    TORCH_CHECK_EQ(queue.device_index(), dev);
  }
  auto queue = getQueueFromPool(false, -1);
  TORCH_CHECK_EQ(queue.device_index(), cid);
}

TEST(QueuePoolPressureTest, getQueueFromPool) {
  const int cid = 0;
  setDevice(cid);
  for (int i = 0; i < 40000; ++i) {
    auto queue = getQueueFromPool();
    TORCH_CHECK_EQ(queue.device_index(), cid);
  }
}
}  // namespace torch_mlu
