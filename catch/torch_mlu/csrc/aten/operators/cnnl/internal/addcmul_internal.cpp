#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_addcmul_internal(at::Tensor& output,
                                  const at::Tensor& self,
                                  const at::Tensor& tensor1,
                                  const at::Tensor& tensor2, const at::Scalar& alpha) {
  // get tensor impl
  auto f_alpha = alpha.toFloat();
  auto self_impl = getMluTensorImpl(self);
  auto tensor1_impl = getMluTensorImpl(tensor1);
  auto tensor2_impl = getMluTensorImpl(tensor2);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // for multi input operators which are not sensitive to stride,
  // the memory format needs to be the same with the first input
  auto layout = suggest_cnnl_layout(self);

  // create the desc
  CnnlTensorDescriptor desc_self;
  CnnlTensorDescriptor desc_tensor1;
  CnnlTensorDescriptor desc_tensor2;
  CnnlTensorDescriptor desc_output;
  desc_self.set(self, layout);
  desc_tensor1.set(tensor1, layout);
  desc_tensor2.set(tensor2, layout);
  desc_output.set(output, layout);

  // get the size of workspace for brodcast
  size_t workspace_size;
  auto desc_self_ = desc_self.desc();
  auto desc_tensor1_ = desc_tensor1.desc();
  auto desc_tensor2_ = desc_tensor2.desc();
  TORCH_CNNL_CHECK(cnnlGetAddcmulWorkspaceSize(handle, desc_self_, desc_tensor1_,
                                               desc_tensor2_, &workspace_size));

  auto workspace = at::empty(workspace_size, self.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);

  // get the mlu ptr
  auto self_ptr = self_impl->mlu_data_ptr();
  auto tensor1_ptr = tensor1_impl->mlu_data_ptr();
  auto tensor2_ptr = tensor2_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto workspace_ptr = workspace_impl->mlu_data_ptr();

  // compute ops
  TORCH_CNNL_CHECK(cnnlAddcmul(handle, desc_self.desc(), self_ptr, &f_alpha,
                               desc_tensor1.desc(), tensor1_ptr,
                               desc_tensor2.desc(), tensor2_ptr,
                               workspace_ptr, workspace_size,
                               desc_output.desc(), output_ptr));


  return output;
}

}  // namespace ops
}  // namespace torch_mlu
