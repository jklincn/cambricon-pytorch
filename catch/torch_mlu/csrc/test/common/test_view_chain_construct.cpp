#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/viewchain/viewChain.h"

namespace torch_mlu {

// Add this copy construct test for code coverage.
// Other functions is already covered by pytest.
TEST(PermuteOp, permute_op_infer_shape) {
  ViewChain chain1;
  TensorInfo input_info({2, 3, 4, 5}, {60, 20, 5, 1}, 0);
  TensorInfo output_info({2, 5, 3, 4}, {60, 12, 4, 1}, 0);
  at::IntArrayRef dims = {0, 3, 1, 2};
  auto ptr = std::make_shared<PermuteOp>(dims);
  chain1.pushNodeToViewChain({2, 3, 4, 5}, {60, 20, 5, 1}, 0,
                             {2, 5, 3, 4}, {60, 12, 4, 1}, 0,
                             ptr);
  ViewChain chain2(chain1);
  ASSERT_TRUE(chain1 == chain2);
}

}  // namespace torch_mlu
