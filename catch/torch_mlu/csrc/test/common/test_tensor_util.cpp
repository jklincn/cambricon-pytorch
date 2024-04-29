#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <caffe2/core/logging.h>
#include <c10/util/Optional.h>

#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "framework/core/queue.h"
#include "framework/core/queue_guard.h"
#include "utils/assert_tensor.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

TEST(TensorUtilTest, getTensorUtilTest) {
  at::Tensor self = at::ones({1}).to(at::Device(at::Device::Type::MLU));
  auto self_impl = getMluTensorImpl(self);
  TORCH_CHECK_EQ(self_impl->device_type(), c10::DeviceType::MLU);
}

// TODO(kongweiguang): temporarily sheild this code for testing.
//TEST(TensorUtilTest, getTensorDeviceTest) {
//  at::Tensor self = at::ones({1}).to(at::Device(at::Device::Type::MLU, 1));
//  auto device_index = getTensorDevice({self});
//  TORCH_CHECK_EQ(device_index, 1);
//}

TEST(TensorUtilTest, copy_to_cpu_cnnlTest) {
  at::Tensor self = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto self_cpu = at::empty_like(self, self.options().device(at::kCPU)).zero_();
  copy_to_cpu(self_cpu, self, true);
  TORCH_CHECK_EQ(self_cpu.device(), c10::DeviceType::CPU);
  assertTensorsEqual(self_cpu, self.cpu(), 0.0, true, false, false);
}

TEST(TensorUtilTest, ismluTest) {
  at::Tensor self = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto self_impl = getMluTensorImpl(self);
  bool self_ismlu = self.is_mlu();
  bool self_impl_ismlu = self_impl->is_mlu();
  TORCH_CHECK_EQ(self_ismlu, true);
  TORCH_CHECK_EQ(self_impl_ismlu, true);
}

TEST(TensorUtilTest, ischannellastTest) {
  at::Tensor self = at::ones({2, 4, 3, 5}).to(at::Device(at::Device::Type::MLU));
  at::Tensor self_cl = self.to(at::MemoryFormat::ChannelsLast);
  bool self_iscl = is_channels_last(self);
  bool self_cl_iscl = is_channels_last(self_cl);
  TORCH_CHECK_EQ(self_iscl, false);
  TORCH_CHECK_EQ(self_cl_iscl, true);
}

}  // namespace torch_mlu
