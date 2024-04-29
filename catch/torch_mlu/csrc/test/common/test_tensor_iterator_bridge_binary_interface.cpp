#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

TEST(binary_op, binary_input_with_float_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(1).tensor_base().scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(2).tensor_base().scalar_type() == at::kFloat);
}

TEST(binary_op, binary_input_with_different_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 4}).to(at::kHalf).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(iter.operand(2).original_tensor_base().scalar_type() == at::kHalf);
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(1).tensor_base().scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(2).tensor_base().scalar_type() == at::kFloat);
}

TEST(binary_op, binary_inplace_input_with_left_half_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor other = at::ones({1, 2}).to(at::kHalf)
                                     .to(at::Device(at::Device::Type::MLU));
  auto input_stride = input.strides();
  auto other_stride = other.strides();
  auto iter = at::TensorIterator::binary_op(input, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(iter_bridge.output(iter, 0).is_same(iter_bridge.input(iter, 0)));
  ASSERT_TRUE(iter.operand(0).original_tensor()
    .is_same(iter.operand(1).original_tensor()));
  ASSERT_TRUE(at::kHalf == iter.operand(2).original_tensor_base().scalar_type());
  ASSERT_TRUE(at::kFloat == iter.operand(2).tensor_base().scalar_type());
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(input_stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(1).original_tensor_base().strides());
  ASSERT_TRUE(other_stride == iter.operand(2).tensor_base().strides());
  ASSERT_TRUE(other_stride == iter.operand(2).original_tensor_base().strides());
}

TEST(binary_op, binary_inplace_input_half_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::kHalf)
                                     .to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor other = at::ones({1, 2}).to(at::Device(at::Device::Type::MLU));
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(input, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(iter_bridge.output(iter, 0).is_same(iter_bridge.input(iter, 0)));
  ASSERT_TRUE(iter.operand(0).original_tensor()
    .is_same(iter.operand(1).original_tensor()));
  ASSERT_TRUE(at::kFloat == iter.operand(2).tensor_base().scalar_type());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(at::kFloat == iter.operand(2).tensor_base().scalar_type());
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(input_stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(1).original_tensor_base().strides());
}

TEST(binary_op, binary_input_with_float_dtype_and_stride) {
  at::Tensor input = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor other = at::ones({1}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
                                      .slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto other_stride = other.strides();
  auto output_stride = output.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> other_size = {1, 1};
  std::vector<int64_t> stride = {2, 1};
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::ScalarType::Float);
  ASSERT_TRUE(stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(stride == iter_bridge.output(iter, 0).strides());
  // TODO(shangang): output tensor will resize flag is False when size is same with
  // common shape. This will cause output original tensor will be a strided tensor,
  // and using copy with stride to copy data from output tensor to output original
  // tensor.
  // ASSERT_TRUE(stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(output.sizes() == iter.operand(0).original_tensor_base().sizes());
  ASSERT_TRUE(input_stride == iter.operand(0).original_tensor_base().strides());
  ASSERT_TRUE(input_stride == iter.operand(1).original_tensor_base().strides());
  ASSERT_TRUE(input.sizes() == iter.operand(1).original_tensor_base().sizes());
  ASSERT_TRUE(other.sizes() == iter.operand(2).original_tensor_base().sizes());
  ASSERT_TRUE(other_size == iter.operand(2).tensor_base().sizes());
  ASSERT_TRUE(other.strides() == iter.operand(2).original_tensor_base().strides());
  ASSERT_TRUE(other_size == iter.operand(2).tensor_base().strides());
}

TEST(binary_op, binary_input_with_int_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::kInt).to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 4}).to(at::kByte).to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::kInt).to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  try {
    iter_bridge.to_build(iter, "default");
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

// Using fixed dtype to build op.
TEST(binary_op, binary_input_with_fixed_dtype) {
  at::Tensor input = at::ones({2, 4}).to(at::kHalf)
                                     .to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 4}).to(at::kHalf)
                                     .to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::kHalf)
                                      .to(at::Device(at::Device::Type::MLU));
  auto iter = at::TensorIteratorConfig().set_check_mem_overlap(true)
                                        .check_all_same_dtype(false)
                                        .add_owned_output(output)
                                        .add_owned_input(input)
                                        .add_owned_input(other)
                                        .declare_static_dtype(at::kInt)
                                        .build();
  TensorIteratorBridge iter_bridge;
  // Using default op, and don't support implicit type cast.
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(iter_bridge.compute_dtype() == at::kHalf);
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kInt);
  ASSERT_TRUE(iter_bridge.input(iter, 0).scalar_type() == at::kHalf);
  ASSERT_TRUE(iter_bridge.input(iter, 0).scalar_type() == at::kHalf);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kHalf);
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type() == at::kInt);
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(iter.operand(1).tensor_base().scalar_type() == at::kHalf);
}

TEST(binary_op, binary_non_overlapping_and_dense_test1) {
  at::Tensor input = at::ones({2, 4, 3, 4}).permute({0, 2, 3, 1})
                       .to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 4, 3, 4}).permute({0, 2, 3, 1})
                       .to(at::Device(at::Device::Type::MLU));
  at::Tensor output;
  auto input_size = input.sizes();
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(input_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(input_size == iter_bridge.output(iter, 0).sizes());
}

TEST(binary_op, binary_non_overlapping_and_dense_test2) {
  at::Tensor input = at::ones({2, 4, 3, 4}).permute({0, 2, 3, 1})
                       .to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({3, 4, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output;
  auto input_size = input.sizes();
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> c_input_size = {2, 3, 4, 4};
  std::vector<int64_t> c_input_stride = {48, 16, 4, 1};
  std::vector<int64_t> c_other_size = {1, 3, 4, 4};
  std::vector<int64_t> c_other_stride = {48, 16, 4, 1};
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(input.sizes() == iter.operand(1).original_tensor_base().sizes());
  ASSERT_TRUE(input.strides() == iter.operand(1).original_tensor_base().strides());
  ASSERT_TRUE(c_input_size == iter.operand(1).tensor_base().sizes());
  ASSERT_TRUE(c_input_stride == iter.operand(1).tensor_base().strides());
  ASSERT_TRUE(other.sizes() == iter.operand(2).original_tensor_base().sizes());
  ASSERT_TRUE(other.strides() == iter.operand(2).original_tensor_base().strides());
  ASSERT_TRUE(c_other_size == iter.operand(2).tensor_base().sizes());
  ASSERT_TRUE(c_other_stride == iter.operand(2).tensor_base().strides());
  ASSERT_TRUE(c_input_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(c_input_size == iter_bridge.output(iter, 0).sizes());
}

TEST(binary_op, binary_memory_format_test) {
  at::Tensor input = at::ones({2, 4, 3, 4}).permute({0, 2, 3, 1})
                       .to(at::Device(at::Device::Type::MLU));
  at::Tensor other = at::ones({2, 3, 4, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output;
  auto input_size = input.sizes();
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> c_input_size = {2, 3, 4, 4};
  std::vector<int64_t> c_input_stride = {48, 16, 4, 1};
  ASSERT_TRUE(input.sizes() == iter.operand(1).original_tensor_base().sizes());
  ASSERT_TRUE(input.strides() == iter.operand(1).original_tensor_base().strides());
  ASSERT_TRUE(c_input_size == iter.operand(1).tensor_base().sizes());
  ASSERT_TRUE(c_input_stride == iter.operand(1).tensor_base().strides());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(other.strides() == iter.operand(2).tensor_base().strides());
  ASSERT_TRUE(other.sizes() == iter.operand(2).tensor_base().sizes());
  ASSERT_TRUE(c_input_size == iter.operand(0).tensor_base().sizes());
  ASSERT_TRUE(c_input_stride == iter.operand(0).tensor_base().strides());
}

TEST(binary_op, binary_memory_format_test_with_dtype) {
  at::Tensor input = at::ones({2, 4, 3, 4}).permute({0, 2, 3, 1})
                       .to(at::Device(at::Device::Type::MLU)).to(at::kHalf);
  at::Tensor other = at::ones({2, 3, 4, 4}).to(at::Device(at::Device::Type::MLU));
  at::Tensor output;
  auto input_size = input.sizes();
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "default");
  std::vector<int64_t> c_input_size = {2, 3, 4, 4};
  std::vector<int64_t> c_input_stride = {48, 16, 4, 1};
  ASSERT_TRUE(iter.operand(1).original_tensor_base().scalar_type() == at::kHalf);
  ASSERT_TRUE(iter.operand(1).tensor_base().scalar_type() == at::kFloat);
  ASSERT_TRUE(input.sizes() == iter.operand(1).original_tensor_base().sizes());
  ASSERT_TRUE(input.strides() == iter.operand(1).original_tensor_base().strides());
  ASSERT_TRUE(c_input_size == iter.operand(1).tensor_base().sizes());
  ASSERT_TRUE(c_input_stride == iter.operand(1).tensor_base().strides());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(0).original_tensor_base().defined());
  ASSERT_TRUE(other.strides() == iter.operand(2).tensor_base().strides());
  ASSERT_TRUE(other.sizes() == iter.operand(2).tensor_base().sizes());
  ASSERT_TRUE(c_input_size == iter.operand(0).tensor_base().sizes());
  ASSERT_TRUE(c_input_stride == iter.operand(0).tensor_base().strides());
}

// CNNL cast op is not support stride, so get a contiguous tensor.
TEST(binary_op, binary_stride_test_with_cast) {
  at::Tensor input = at::ones({2, 4}).to(at::kFloat)
                                     .to(at::Device(at::Device::Type::MLU))
                                     .slice(1, 0, 4, 2);
  at::Tensor other = at::ones({2, 2}).to(at::kFloat)
                                     .to(at::Device(at::Device::Type::MLU));
  at::Tensor output = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU))
    .to(at::kHalf).slice(1, 0, 4, 2);
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(output, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  // get a contiguous stride.
  std::vector<int64_t> c_output_stride = {2, 1};
  ASSERT_TRUE(input_stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(c_output_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(!iter.operand(1).original_tensor_base().defined());
  ASSERT_TRUE(!iter.operand(2).original_tensor_base().defined());
  ASSERT_TRUE(iter_bridge.output(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter_bridge.input(iter, 0).scalar_type() == at::kFloat);
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kHalf);
}

TEST(binary_op, binary_stride_test_with_cast_inplace) {
  at::Tensor input = at::ones({2, 3, 4}).to(at::kHalf)
                                     .to(at::Device(at::Device::Type::MLU))
                                     .slice(2, 0, 4, 2);
  at::Tensor other = at::ones({3, 2}).to(at::kFloat)
                                     .to(at::Device(at::Device::Type::MLU));
  auto input_stride = input.strides();
  auto iter = at::TensorIterator::binary_op(input, input, other);
  TensorIteratorBridge iter_bridge;
  // Using default op
  iter_bridge.to_build(iter, "fake_op");
  // get a contiguous stride.
  std::vector<int64_t> c_output_stride = {6, 2, 1};
  std::vector<int64_t> c_other_size = {1, 3, 2};
  std::vector<int64_t> c_other_stride = {6, 2, 1};
  ASSERT_TRUE(c_output_stride == iter_bridge.input(iter, 0).strides());
  ASSERT_TRUE(c_output_stride == iter_bridge.output(iter, 0).strides());
  ASSERT_TRUE(c_other_size == iter_bridge.input(iter, 1).sizes());
  ASSERT_TRUE(c_other_stride == iter_bridge.input(iter, 1).strides());
  ASSERT_TRUE(iter_bridge.output(iter, 0).is_same(iter_bridge.input(iter, 0)));
  ASSERT_TRUE(iter.operand(0).original_tensor()
    .is_same(iter.operand(1).original_tensor()));
  ASSERT_TRUE(iter.operand(0).original_tensor_base().scalar_type()
    == at::kHalf);
  ASSERT_TRUE(iter.operand(0).tensor_base().scalar_type()
    == at::kFloat);
}

}  // namespace torch_mlu
