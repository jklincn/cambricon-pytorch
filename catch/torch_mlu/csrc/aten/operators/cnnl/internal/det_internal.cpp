#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_det_internal(at::Tensor& output, const at::Tensor& input,
                              cnnlDetMode_t mode) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(input);
  descOutput.set(output);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlDet(handle, mode, descInput.desc(), input_ptr,
                           descOutput.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
