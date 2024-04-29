#include <gtest/gtest.h>

namespace torch_mlu {

// TODO(kongweiguang): currently shield this case because
// the threshold op is not supported on pytorch 1.13.1. 
#if 0
TEST(ThresholdTest, threshold_backward_dtype) {
  auto grad_output = at::randn({4}).to(at::kMLU).to(at::kDouble);
  auto self = at::randn({4}).to(at::kMLU).to(at::kDouble);
  auto threshold = at::Scalar(1);
  auto ret_mlu = at::threshold_backward(grad_output, self, threshold);
  EXPECT_EQ(ret_mlu.scalar_type(), at::kDouble);
}
#endif

}  // namespace torch_mlu
