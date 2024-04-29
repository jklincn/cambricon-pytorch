#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

// nullary_input is protected in TensorIterator, so we need to use
// to_build() to get it.
TEST(nullary_op, nullary_input_with_float_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::nullary_op(input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
}

TEST(nullary_op, nullary_input_with_int_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU)).to(at::kInt);
  auto iter = at::TensorIterator::nullary_op(input);
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  try {
    iter_bridge.to_build(iter, "default");
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

// Using fixed dtype to build op.
TEST(nullary_op, nullary_input_with_fixed_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIteratorConfig().set_check_mem_overlap(true)
                                        .check_all_same_dtype(false)
                                        .resize_outputs(false)
                                        .add_owned_output(input)
                                        .declare_static_dtype(at::kInt)
                                        .build();
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::kInt);
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kInt);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kFloat);
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type() == at::kInt);
}

TEST(nullary_op, nullary_stride_test_without_cast) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU)).slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto input_dtype = input.scalar_type();
  auto iter = at::TensorIterator::nullary_op(input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  ASSERT_TRUE(input_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(input_dtype == iter_bridge.output(iter, 0).scalar_type());
}

// CNNL cast op is not support stride, so get a contiguous tensor.
TEST(nullary_op, nullary_stride_test_with_cast) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
    .to(at::kInt).slice(1, 0, 4, 2);
  auto input_dtype = input.scalar_type();
  auto iter = at::TensorIterator::nullary_op(input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  // get a contiguous stride.
  std::vector<int64_t> input_stride = {2, 1};
  ASSERT_TRUE(input_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kInt);
}

}  // namespace torch_mlu
