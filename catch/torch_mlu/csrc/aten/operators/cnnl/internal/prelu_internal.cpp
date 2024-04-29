#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_prelu_internal(const at::Tensor& self, const at::Tensor& weight) {
  auto output = at::native::empty_like(self);
  auto self_impl = getMluTensorImpl(self);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);

  int64_t weight_num = weight.numel();
  std::vector<int64_t> cnnl_weight_size(self.dim(), 1);  // case1: shared weight for all channels
  if (weight_num != 1) {  // case2: multiple weights, one for each channel
      int64_t self_ndim = self.dim();
      TORCH_CHECK(self_ndim > 0, "Not allow zero-dim input tensor.");

      int64_t channel_size = 1;  // channel_size default to 1
      if (self_ndim > 1) {
        channel_size = self.size(1);  // channel is the 2nd dim of input
      }
      TORCH_CHECK(channel_size == weight_num,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = ",
        weight_num, " and channel size = ", channel_size, ".");

      // cnnlPrelu supported shape of input tensor and weight tensor are as follows:
      // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., 1, ..., 1]
      // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., 1, ..., c]
      // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., b, ..., 1]
      // only one channel of weight can > 1, so we aggregate weight_num to the 2nd dim.
      cnnl_weight_size[1] = weight_num;
  }

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descSelf;
  CnnlTensorDescriptor descWeight;
  CnnlTensorDescriptor descOutput;
  descSelf.set(self);
  descWeight.set(weight, cnnl_weight_size, get_contiguous_strides(cnnl_weight_size));
  descOutput.set(output);
  // malloc mlu memory
  auto self_ptr = self_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlPrelu(handle, descSelf.desc(), self_ptr, descWeight.desc(), weight_ptr,
    descOutput.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace torch_mlu
