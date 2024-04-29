#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/viewchain/specificViewOps.h"

namespace torch_mlu {

TEST(PermuteOp, permute_op_infer_shape) {
  PermuteOp op({0, 3, 1, 2});
  TensorInfo input_info({2, 3, 4, 5}, {60, 20, 5, 1}, 0);
  TensorInfo output_info({2, 5, 3, 4}, {60, 12, 4, 1}, 0);
  op.updateInputTensorInfo(input_info);
  op.inferShape();
  ASSERT_TRUE(output_info == op.getOutputTensorInfo());
}

TEST(SliceOp, slice_op_infer_shape) {
  SliceOp op(2, 0, 2, 1);
  TensorInfo input_info({2, 3, 4, 5}, {60, 20, 5, 1}, 0);
  TensorInfo output_info({2, 3, 2, 5}, {30, 10, 5, 1}, 0);
  op.updateInputTensorInfo(input_info);
  op.inferShape();
  ASSERT_TRUE(output_info == op.getOutputTensorInfo());
}

TEST(ExpandOp, expand_op_infer_shape) {
  ExpandOp op({2, 3, 4, 5}, false);
  TensorInfo input_info({2, 1, 1, 5}, {5, 5, 5, 1}, 0);
  TensorInfo output_info({2, 3, 4, 5}, {60, 1, 15, 3}, 0);
  op.updateInputTensorInfo(input_info);
  op.inferShape();
  std::cout << output_info << " api: " << op.getOutputTensorInfo() << std::endl;
  ASSERT_TRUE(output_info == op.getOutputTensorInfo());
}

TEST(ReshapeOp, reshape_op_infer_shape) {
  ReshapeOp op({2, 5, 3, 4});
  TensorInfo input_info({2, 3, 4, 5}, {60, 20, 5, 1}, 0);
  TensorInfo output_info({2, 5, 3, 4}, {60, 12, 4, 1}, 0);
  op.updateInputTensorInfo(input_info);
  op.inferShape();
  ASSERT_TRUE(output_info == op.getOutputTensorInfo());
}

TEST(UnfoldOp, unfold_view_coverage) {
  UnfoldOp op(2, 2, 2);
  TensorInfo input_info({2, 3, 4, 5}, {60, 20, 5, 1}, 0);
  TensorInfo output_info({2, 3, 2, 5, 2}, {60, 20, 10, 2, 1}, 0);
  op.updateInputTensorInfo(input_info);
  op.inferShape();
  ASSERT_TRUE(output_info == op.getOutputTensorInfo());
}

}  // namespace torch_mlu
