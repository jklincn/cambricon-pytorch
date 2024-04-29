#include <gtest/gtest.h>

// #include <c10/core/impl/InlineDeviceGuard.h>
#include "aten/utils/types.h"

namespace torch_mlu {

TEST(TypeRelatedTest, TestInvalid) {
  try {
    getCnnlDataType(caffe2::TypeMeta());
  } catch(std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }

  ASSERT_EQ(CNRT_INVALID, cnnlType2CnrtType(CNNL_DTYPE_INVALID));
}

}  // namespace torch_mlu
