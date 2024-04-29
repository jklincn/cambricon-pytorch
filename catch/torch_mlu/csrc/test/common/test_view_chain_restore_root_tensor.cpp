#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <torch/torch.h>
#include "aten/viewchain/viewChain.h"
#include "aten/utils/tensor_util.h"

namespace torch_mlu {

void test_case(bool use_reshape) {
  auto input = at::ones({129, 128, 32, 128}).to(at::kMLU);
  auto out1 = at::slice(input, 3, 0, 64);

  at::Tensor out2;
  if (use_reshape) {
    out2 = out1.reshape({129, 128, 32, 32, 2});
  } else {
    out2 = out1.view({129, 128, 32, 32, 2});
  }
  auto out3 = at::slice(out2, 4, 0, 1);

  auto* impl = getMluTensorImpl(out3);
  auto node_size =
    dynamic_cast<MLUTensorImpl*>(impl->external_.get())->view_chain_.getViewChainNodeSize();
  ASSERT_TRUE(node_size == 3);

  std::vector<int64_t> ref_shape = {129, 128, 32, 32, 1};
  std::vector<int64_t> ref_stride = {524288, 4096, 128, 2, 1};
  ASSERT_TRUE(out3.sizes().vec() == ref_shape);
  ASSERT_TRUE(out3.strides().vec() == ref_stride);
}

TEST(RestoreRootTensor, slice_reshape_slice_test) {
  test_case(true);
}

TEST(RestoreRootTensor, slice_view_slice_test) {
  test_case(false);
}

}  // namespace torch_mlu
