#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

TEST(unary_op, unary_input_with_float_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::unary_op(output, input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
}

TEST(unary_op, unary_inplace_input) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::unary_op(input, input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(iter_bridge.output(iter, 0).is_same(iter_bridge.input(iter, 0)));
  ASSERT_TRUE(iter.operand(0).original_tensor()
    .is_same(iter.operand(1).original_tensor()));
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(input_stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(1).original_tensor_base().strides());
}

TEST(unary_op, unary_input_with_float_dtype_and_stride) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                      .slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto output_stride = output.strides();
  auto iter = at::TensorIterator::unary_op(output, input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  // TODO(shangang): output tensor will resize flag is False when size is same with
  // common shape. This will cause output original tensor will be a strided tensor,
  // and using copy with stride to copy data from output tensor to output original
  // tensor.
  // ASSERT_TRUE(stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(1).original_tensor_base().strides());
}

TEST(unary_op, unary_input_with_int_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::kInt).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::kInt).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::unary_op(output, input);
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  try {
    iter_bridge.to_build(iter, "default");
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

// Using fixed dtype to build op.
TEST(unary_op, unary_input_with_fixed_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIteratorConfig().set_check_mem_overlap(true)
                                        .check_all_same_dtype(false)
                                        .add_owned_output(output)
                                        .add_owned_input(input)
                                        .declare_static_dtype(at::kInt)
                                        .build();
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::kFloat);
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kInt);
  ASSERT_TRUE(iter_bridge.input(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kFloat);
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type() == at::kInt);
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(iter.operand(1).tensor_base().scalar_type() == at::kFloat);
}

TEST(unary_op, unary_stride_test_without_cast) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                      .slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto input_dtype = input.scalar_type();
  auto output_stride = output.strides();
  auto output_dtype = output.scalar_type();
  auto iter = at::TensorIterator::unary_op(output, input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  ASSERT_TRUE(input_stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(input_dtype == iter_bridge.input(iter, 0).scalar_type());
  ASSERT_TRUE(output_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(output_dtype == iter_bridge.output(iter, 0).scalar_type());
}

// CNNL cast op is not support stride, so get a contiguous tensor.
TEST(unary_op, nullary_stride_test_with_cast) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
    .to(at::kInt).slice(1, 0, 4, 2);
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
    .to(at::kInt).slice(1, 0, 4, 2);
  auto input_dtype = input.scalar_type();
  auto iter = at::TensorIterator::unary_op(output, input);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  // get a contiguous stride.
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter_bridge.input(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kInt);
}

}  // namespace torch_mlu
