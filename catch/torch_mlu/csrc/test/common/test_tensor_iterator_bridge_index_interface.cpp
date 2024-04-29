#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ATen/native/TensorIterator.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

TEST(index_op, index_op_test_dtype_and_stride) {
  at::Tensor input = at::ones({2, 2, 2, 8})
                     .to(at::Device(at::Device::Type::MLU))
                     .slice(3, 0, 8, 2);
  at::Tensor index_1 = at::ones({2, 8})
                       .to(at::Device(at::Device::Type::MLU))
                       .to(at::kBool)
                       .slice(1, 0, 8, 2);
  at::Tensor index_2 = at::ones({4}).to(at::Device(at::Device::Type::MLU)).to(at::kLong);
  at::Tensor output = at::ones({2, 2, 2, 8})
                      .to(at::Device(at::Device::Type::MLU))
                      .slice(3, 0, 8, 2);
  at::TensorIteratorConfig config;
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(output);
  config.add_input(input);
  config.add_input(index_1);
  config.add_input(index_2);
  at::TensorIterator iter = config.build();
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "index");
  ASSERT_TRUE(iter.input(0).scalar_type() == at::ScalarType::Float);
  ASSERT_TRUE(iter.input(1).scalar_type() == at::ScalarType::Bool);
  ASSERT_TRUE(iter.input(2).scalar_type() == at::ScalarType::Long);
  ASSERT_TRUE(iter.output(0).scalar_type() == at::ScalarType::Float);
  ASSERT_TRUE(iter.input(0).is_contiguous() && iter.input(1).is_contiguous()
              && iter.input(2).is_contiguous() && iter.output(0).is_contiguous());
  std::vector<int64_t> output_stride = {16, 8, 4, 1};
  std::vector<int64_t> index_1_stride = {4, 1};
  std::vector<int64_t> index_2_stride = {1};
  ASSERT_TRUE(iter.input(0).sizes() == input.sizes());
  ASSERT_TRUE(iter.input(0).strides() == output_stride);
  ASSERT_TRUE(iter.input(1).sizes() == index_1.sizes());
  ASSERT_TRUE(iter.input(1).strides() == index_1_stride);
  ASSERT_TRUE(iter.input(2).sizes() == index_2.sizes());
  ASSERT_TRUE(iter.input(2).strides() == index_2_stride);
  ASSERT_TRUE(iter.output(0).sizes() == output.sizes());
  ASSERT_TRUE(iter.output(0).strides() == output_stride);
}

}  // namespace torch_mlu
