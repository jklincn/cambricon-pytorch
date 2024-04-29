#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> cnnl_prelu_backward_internal(const at::Tensor& grad,
                                                                const at::Tensor& self,
                                                                const at::Tensor& weight) {
  int64_t weight_num = weight.numel();
  std::vector<int64_t> cnnl_weight_size(self.dim(), 1);  // case1: shared weight for all channels
  if (weight_num != 1) {  // case2: multiple weights, one for each channel
      int64_t self_ndim = self.dim();
      TORCH_CHECK(self_ndim > 0,
                  "Not allow zero-dim input tensor in cnnl_prelu_backward_internal.");

      int64_t channel_size = 1;  // channel_size default to 1
      if (self_ndim > 1) {
        channel_size = self.size(1);  // channel is the 2nd dim of input
      }
      TORCH_CHECK(channel_size == weight_num,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = ",
        weight_num, " and channel size = ", channel_size, ".");

      cnnl_weight_size[1] = weight_num;
  }

  auto grad_impl = getMluTensorImpl(grad);
  auto input_impl = getMluTensorImpl(self);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_grad = at::native::empty_like(self);
  auto weight_grad = at::native::empty_like(weight);
  auto output_grad_impl = getMluTensorImpl(output_grad);
  auto weight_grad_impl = getMluTensorImpl(weight_grad);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descGrad;
  CnnlTensorDescriptor descWeight;
  CnnlTensorDescriptor descOutputGrad;
  CnnlTensorDescriptor descWeightGrad;

  descInput.set(self, CNNL_LAYOUT_ARRAY);
  descGrad.set(grad, CNNL_LAYOUT_ARRAY);
  descWeight.set(weight, cnnl_weight_size,
                 get_contiguous_strides(cnnl_weight_size));
  descOutputGrad.set(output_grad, CNNL_LAYOUT_ARRAY);
  descWeightGrad.set(weight_grad, cnnl_weight_size,
                     get_contiguous_strides(cnnl_weight_size));

  // workspace
  size_t workspace_size = 0;
  void * worksapce_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetPreluBackwardWorkspaceSize(handle,
                          descWeightGrad.desc(), &workspace_size));
  auto workspace = at::empty(workspace_size, self.options());
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();

  // malloc mlu memory
  auto grad_ptr = grad_impl->mlu_data_ptr();
  auto input_ptr = input_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto output_grad_ptr = output_grad_impl->mlu_data_ptr();
  auto weight_grad_ptr = weight_grad_impl->mlu_data_ptr();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlPreluBackward(handle, descInput.desc(), input_ptr,
                                     descGrad.desc(), grad_ptr, descWeight.desc(), weight_ptr,
                                     workspace_ptr, workspace_size, descOutputGrad.desc(),
                                     output_grad_ptr, descWeightGrad.desc(), weight_grad_ptr));
  return std::make_tuple(output_grad, weight_grad);
}

}  // namespace ops
}  // namespace torch_mlu
