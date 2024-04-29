/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <c10/util/Optional.h>

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ComplexHelper.h>
#include "ATen/InferSize.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/resize.h"
#include "aten/viewchain/specificViewOps.h"

namespace torch_mlu {
namespace ops {

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
Tensor alias_with_sizes_and_strides(
    const at::Tensor& self,
    const Vec& sizes,
    const Vec& strides) {
  //caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array size is the same)
  at::Tensor self_;
  TORCH_MLU_CHECK(!(self.is_quantized()),
    "Quantized Tensor is not supported on MLU.");
  auto* self_impl = getMluTensorImpl(self);
  self_ = at::detail::make_tensor<c10::TensorImpl>(
    c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset());
  self_tmp_->set_sizes_and_strides(sizes, strides);
  auto* result_impl = getMluTensorImpl(self_);
  at::namedinference::propagate_names(self_, self);
  dynamic_cast<MLUTensorImpl*>(result_impl->external_.get())->view_chain_ =
      dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())->view_chain_;
  return self_;
}

inline at::Tensor cnnl_view_impl(const at::Tensor& self, at::IntArrayRef size) {
  
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  if ((!stride.has_value())
      && (self.dim() < 6) && (self.dim() > 3)
      && (self.is_contiguous(get_channels_last_memory_format(self.dim())))) {

    auto self_channels_first = permute_to_contiguous(self, c10::MemoryFormat::Contiguous);
    inferred_size = at::infer_size(size, self_channels_first.numel());
    stride = at::detail::computeStride(self_channels_first.sizes(),
                                       self_channels_first.strides(),
                                       inferred_size);
    auto stride_value = *stride;
    return cnnl_as_strided(self_channels_first, inferred_size,
                           stride_value, self.storage_offset());
  }
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto output = alias_with_sizes_and_strides(self, inferred_size, *stride);

  // If inferred_size is same with self tensor size, self and output tensor will be
  // completely same. So no need to push a reshape node in view chain.
  if (self.sizes().equals(inferred_size)) {
    return output;
  }
  const auto ptr = std::make_shared<ReshapeOp>(inferred_size);
  auto* output_impl = getMluTensorImpl(output);
  dynamic_cast<MLUTensorImpl*>(output_impl->external_.get())
      ->view_chain_.pushNodeToViewChain(self, output, ptr);
  return output;
}


at::Tensor cnnl_view(const at::Tensor &self, at::IntArrayRef size) {
  return cnnl_view_impl(self, size);
}


// Computes the strides for view_dtype output when the view dtype is
// smaller than the original dtype
inline at::DimVector compute_strides_for_view_dtype_downsize(at::IntArrayRef old_strides,
                                                             int64_t size_ratio,
                                                             at::ScalarType old_dtype,
                                                             at::ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();

  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  at::DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    new_strides[dim_idx] = old_strides[dim_idx] * size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}

// Computes the strides for view_dtype output when the view dtype is
// larger than the original dtype
inline at::DimVector compute_strides_for_view_dtype_upsize(at::IntArrayRef old_strides,
                                                           int64_t size_ratio,
                                                           at::ScalarType old_dtype,
                                                           at::ScalarType new_dtype) {
  const int64_t ndim = old_strides.size();
  TORCH_CHECK(
    old_strides[ndim - 1] == 1,
    "self.stride(-1) must be 1 to view ", old_dtype, " as ", new_dtype,
    " (different element sizes), but got ", old_strides[ndim - 1]);

  at::DimVector new_strides(ndim);
  for (int64_t dim_idx = 0; dim_idx < ndim - 1; dim_idx++) {
    TORCH_CHECK(
      (old_strides[dim_idx] % size_ratio) == 0,
      "self.stride(", dim_idx, ") must be divisible by ", size_ratio,
      " to view ", old_dtype, " as ", new_dtype, " (different element sizes), ",
      "but got ", old_strides[dim_idx]);

    new_strides[dim_idx] = old_strides[dim_idx] / size_ratio;
  }
  new_strides[ndim - 1] = 1;
  return new_strides;
}
// view dtype
at::Tensor cnnl_view(const at::Tensor & self, c10::ScalarType dtype) {
  if (self.scalar_type() == dtype) {
    return self;
  }
  auto* self_impl = getMluTensorImpl(self);
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_MLU_CHECK(!self.is_conj(),
    "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_MLU_CHECK(!self.is_neg(),
    "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");

  int64_t self_element_size = self.element_size();
  int64_t new_element_size = static_cast<int64_t>(type_meta.itemsize());

  c10::Storage storage = self.storage();
  auto new_tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();
  
  if (self_element_size == new_element_size) {
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(self.sizes(), self.strides());

  } else if (self.dim() == 0) {
    TORCH_CHECK(false,
      "self.dim() cannot be 0 to view ", self.scalar_type(), " as ",
      dtype, " (different element sizes)");

  } else if (self_element_size > new_element_size) {
    // Downsizing element size
    int64_t size_ratio = self_element_size / new_element_size;
    auto new_strides = compute_strides_for_view_dtype_downsize(
      self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    at::DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] *= size_ratio;

    auto new_storage_offset = size_ratio * self.storage_offset();

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);

  } else {
    // Upsizing element size
    int64_t size_ratio = new_element_size / self_element_size;

    TORCH_CHECK(
      (self.size(-1) % size_ratio) == 0,
      "self.size(-1) must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), ",
      "but got ", self.size(-1));

    TORCH_CHECK(
      (self.storage_offset() % size_ratio) == 0,
      "self.storage_offset() must be divisible by ", size_ratio, " to view ",
      self.scalar_type(), " as ", dtype, " (different element sizes), but got ",
      self.storage_offset());

    auto new_strides = compute_strides_for_view_dtype_upsize(
      self.strides(), size_ratio, self.scalar_type(), dtype);

    auto old_sizes = self.sizes();
    at::DimVector new_sizes(self.dim());
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes[self.dim() - 1] /= size_ratio;

    auto new_storage_offset = self.storage_offset() / size_ratio;

    impl->set_storage_offset(new_storage_offset);
    impl->set_sizes_and_strides(new_sizes, new_strides);
  }
  auto* result_impl = getMluTensorImpl(new_tensor);
  dynamic_cast<MLUTensorImpl*>(result_impl->external_.get())->view_chain_ =
      dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())->view_chain_;
  return new_tensor;
}
// View tensor with new dtype, storage offset, sizes and strides
inline Tensor view_tensor(
    const at::Tensor &tensor, at::ScalarType dtype,
    int64_t offset, at::IntArrayRef sizes, at::IntArrayRef strides) {
  at::Storage storage = tensor.storage();
  auto* self_impl = getMluTensorImpl(tensor);
  auto key_set = tensor.key_set().remove(at::DispatchKey::Conjugate);
  auto new_tensor = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage), key_set, scalarTypeToTypeMeta(dtype));
  auto * impl = new_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(offset);
  impl->set_sizes_and_strides(sizes, strides);
  auto* result_impl = getMluTensorImpl(new_tensor);
  dynamic_cast<MLUTensorImpl*>(result_impl->external_.get())->view_chain_ =
      dynamic_cast<MLUTensorImpl*>(self_impl->external_.get())->view_chain_;
  return new_tensor;
}

inline at::DimVector computeStrideForViewAsReal(at::IntArrayRef oldstride) {
  at::DimVector res(oldstride.size() + 1);
  for (const auto i : c10::irange(oldstride.size())) {
    res[i] = oldstride[i] * 2;
  }
  res.back() = 1;
  return res;
}

Tensor _view_as_real_physical(const at::Tensor& self) {
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  auto old_sizes = self.sizes();
  at::DimVector new_sizes(old_sizes.size() + 1);
  std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.back() = 2;
  auto new_strides = computeStrideForViewAsReal(self.strides());
  auto new_storage_offset = 2 * self.storage_offset();
  const auto float_type = c10::toRealValueType(self.scalar_type());
  auto real_tensor = view_tensor(self, float_type, new_storage_offset, new_sizes, new_strides);
  return real_tensor;
}

// expects as input a complex tensor and returns back a tensor
// with corresponding real dtype containing the complex values
// in the last two dimensions
Tensor cnnl_view_as_real(const Tensor& self) {
  TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  "
		               "To resolve the conjugate tensor so you can view it as real, "
			       "use self.resolve_conj(); however, be warned that the resulting "
			       "tensor will NOT alias the original.");
  return _view_as_real_physical(self);
}

at::Tensor cnnl_view_as_complex(const at::Tensor& self) {
  TORCH_MLU_CHECK(
    self.scalar_type() == c10::kFloat
    || self.scalar_type() == c10::kDouble
    || self.scalar_type() == c10::kHalf,
    "view_as_complex is only supported for half, float and double tensors, "
    "but got a tensor of scalar type: ",
    self.scalar_type());

  auto old_sizes = self.sizes();
  TORCH_MLU_CHECK(old_sizes.size() != 0,
   "Input tensor must have one or more dimensions");

  TORCH_MLU_CHECK(old_sizes[old_sizes.size()-1] == 2,
  "Tensor must have a last dimension of size 2");
  at::DimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);
  const auto new_strides = at::native::computeStrideForViewAsComplex(self.strides());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_MLU_CHECK(self.storage_offset() % 2 == 0,
  "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.storage_offset() / 2;
  return view_tensor(self, complex_type, new_storage_offset, new_sizes, new_strides);
}


at::Tensor cnnl__reshape_alias(const at::Tensor& self, at::IntArrayRef sizes, at::IntArrayRef strides) {
  // This is only used by `reshape` in cases where it would otherwise have dispatched
  // to `view`. This removes the overhead of calling `view` which duplicates some of
  // the work that's already been done (`infer_size_dv` and `computeStride`).
  auto output = alias_with_sizes_and_strides(self, sizes, strides);

  // If inferred_size is same with self tensor size, self and output tensor will be
  // completely same. So no need to push a reshape node in view chain.
  if (self.sizes().equals(sizes)) {
    return output;
  }
  const auto ptr = std::make_shared<ReshapeOp>(sizes);
  auto* output_impl = getMluTensorImpl(output);
  dynamic_cast<MLUTensorImpl*>(output_impl->external_.get())
      ->view_chain_.pushNodeToViewChain(self, output, ptr);
  return output;
}

}  // namespace ops
}  // namespace torch_mlu

