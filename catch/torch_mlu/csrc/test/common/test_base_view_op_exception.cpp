#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/viewchain/baseViewOps.h"

namespace torch_mlu {

#define ALL_TEST_FUNC(_)      \
  _(inferShape)               \
  _(hasCnnlSpecificFunction)  \
  _(parameterToString)

TEST(BaseViewOp, base_view_coverage) {
  BaseViewOp op(ViewsType::kInvalid);
#define TEST_EXCEPTION(FUNC_NAME)             \
  try {                                       \
    op.FUNC_NAME();                           \
  } catch(std::exception& e) {                \
    std::cout << e.what() << std::endl;       \
  }
  ALL_TEST_FUNC(TEST_EXCEPTION)
}

TEST(BaseViewOp, base_view_coverage_with_parameter) {
  BaseViewOp op(ViewsType::kInvalid);
  BaseViewOp other(ViewsType::kInvalid);
  at::Tensor undefined_tensor;
  // test isEqual in base class.
  try {
    op.isEqual(other);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
  // test runCnnlSpecificFunction in base class.
  try {
    op.runCnnlSpecificFunction(undefined_tensor, c10::nullopt);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

}  // namespace torch_mlu
