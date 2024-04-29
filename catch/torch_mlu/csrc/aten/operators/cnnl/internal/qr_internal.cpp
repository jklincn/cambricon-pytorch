#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> cnnl_qr_internal(at::Tensor& Q, at::Tensor& R,
                                                    const at::Tensor& input, bool some) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descQ;
  CnnlTensorDescriptor descR;
  descInput.set(input, CNNL_LAYOUT_ARRAY);
  descQ.set(Q, CNNL_LAYOUT_ARRAY);
  descR.set(R, CNNL_LAYOUT_ARRAY);
  // set descriptor config
  auto handle = getCurrentHandle();
  size_t workspace_size;
  TORCH_CNNL_CHECK(cnnlGetQRWorkspaceSize(handle, descInput.desc(), some, &workspace_size));
  auto workspace = at::empty({static_cast<long>(workspace_size)},
      at::TensorOptions(at::ScalarType::Byte).device(at::Device(at::Device::Type::MLU)));

  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto Q_impl = getMluTensorImpl(Q);
  auto R_impl = getMluTensorImpl(R);
  auto workspace_impl = getMluTensorImpl(workspace);
  auto input_ptr = input_impl->cnnlMalloc();
  auto Q_ptr = Q_impl->cnnlMalloc();
  auto R_ptr = R_impl->cnnlMalloc();
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  TORCH_CNNL_CHECK(cnnlQR(handle, descInput.desc(), input_ptr, descQ.desc(), Q_ptr,
                  descR.desc(), R_ptr, workspace_ptr, workspace_size, some));
  return std::make_tuple(Q, R);
}

}  // namespace ops
}  // namespace torch_mlu
