#include <gtest/gtest.h>
#include "cnnl.h"
#include "utils/assert_tensor.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

TEST(EfficientzerotensorTest, efficientzerotensor_test) {
  std::vector<at::IntArrayRef> size_arr;
  size_arr.push_back({0});
  size_arr.push_back({1024});
  size_arr.push_back({512, 2});
  size_arr.push_back({128, 4, 2});
  for (auto size : size_arr) {
    auto zeros_cpu = at::_efficientzerotensor(size);
    auto zeros_mlu = torch_mlu::ops::cnnl__efficientzerotensor(size,
                                                               zeros_cpu.scalar_type(),
                                                               zeros_cpu.layout(),
                                                               at::kMLU);
    TORCH_INTERNAL_ASSERT(zeros_cpu.device().is_cpu());
    TORCH_INTERNAL_ASSERT(zeros_mlu.device().is_mlu());
    assertTensorsEqual(zeros_cpu, zeros_mlu.to(at::kCPU), 0.0, true, false, false);
  }
}

TEST(EfficientzerotensorTest, efficientzerotensor_test_dtype) {
  std::vector<at::ScalarType> dtype_arr{at::ScalarType::Byte, at::ScalarType::Char, at::ScalarType::Short,
                                        at::ScalarType::Int, at::ScalarType::Long, at::ScalarType::Bool,
                                        at::ScalarType::Half, at::ScalarType::Float, at::ScalarType::Double};
  for (auto dtype : dtype_arr) {
    auto zeros_cpu = at::_efficientzerotensor({16, 2, 4, 8}, dtype);
    auto zeros_mlu = torch_mlu::ops::cnnl__efficientzerotensor({16, 2, 4, 8},
                                                               dtype,
                                                               zeros_cpu.layout(),
                                                               at::kMLU);
    assertTensorsEqual(zeros_cpu, zeros_mlu.to(at::kCPU), 0.0, true, false, false);
  }
}

TEST(EfficientzerotensorTest, efficientzerotensor_test_exception) {
  // Catch wrong device
  EXPECT_THROW({
    try {
      auto zeros = torch_mlu::ops::cnnl__efficientzerotensor({1024},
                                                              at::ScalarType::Float,
                                                              at::kStrided,
                                                              at::kCPU);
    }
    catch(const c10::Error& e){
      EXPECT_STREQ("Device must be MLU", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

}  // namespace torch_mlu
