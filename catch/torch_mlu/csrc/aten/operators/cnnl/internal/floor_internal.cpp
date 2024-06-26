#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_floor_internal(at::Tensor& output, const at::Tensor& input) {
  if (input.numel() == 0) {
    return output;
  }
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
                  "floor input only support floating type");
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input, CNNL_LAYOUT_ARRAY);
  output_desc.set(output, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlFloor(handle, input_desc.desc(), input_ptr,
                             output_desc.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace torch_mlu

