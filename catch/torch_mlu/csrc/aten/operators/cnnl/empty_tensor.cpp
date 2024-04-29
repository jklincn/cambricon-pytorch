#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "framework/core/mlu_guard.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_empty(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided);

  const auto dtype = c10::dtype_or_default(dtype_opt);
  // TODO: add lazyInitMLU() here?
  // at::globalContext().lazyInitMLU();
  const auto device = c10::device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mlu());
  const torch_mlu::mlu::MLUGuard device_guard(device);
  auto* allocator = getMLUCachingAllocator();
  constexpr c10::DispatchKeySet mlu_dks(c10::DispatchKey::MLU);
  return at::detail::empty_generic(
      size, allocator, mlu_dks, dtype, memory_format_opt);
}

at::Tensor cnnl_empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  // TODO: add lazyInitMLU() here?
  // at::globalContext().lazyInitMLU();
  const auto device = c10::device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mlu());
  const torch_mlu::mlu::MLUGuard device_guard(device);
  auto* allocator = getMLUCachingAllocator();
  constexpr c10::DispatchKeySet mlu_dks(c10::DispatchKey::MLU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, mlu_dks, dtype);
}

} // namespace ops
} // namespace torch_mlu
