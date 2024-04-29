#include <ATen/autocast_mode.h>
#include "aten/utils/types.h"
#include "aten/utils/internal_util.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Cause on-chip and off-chip type maybe different, so we can only use
// on-chip type to determine the cast type.
using pair = std::pair<cnnlDataType_t, cnnlDataType_t>;
static const std::map<pair, cnnlCastDataType_t> cast_map = {
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_HALF}, CNNL_CAST_FLOAT_TO_HALF},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT32}, CNNL_CAST_FLOAT_TO_INT32},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT16}, CNNL_CAST_FLOAT_TO_INT16},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT8}, CNNL_CAST_FLOAT_TO_INT8},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_UINT8}, CNNL_CAST_FLOAT_TO_UINT8},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_BOOL}, CNNL_CAST_FLOAT_TO_BOOL},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_BFLOAT16}, CNNL_CAST_FLOAT_TO_BFLOAT16},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT64}, CNNL_CAST_FLOAT_TO_INT64},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_DOUBLE}, CNNL_CAST_FLOAT_TO_DOUBLE},
    {pair{CNNL_DTYPE_BFLOAT16, CNNL_DTYPE_FLOAT}, CNNL_CAST_BFLOAT16_TO_FLOAT},
    // without half to double
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_FLOAT}, CNNL_CAST_HALF_TO_FLOAT},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT32}, CNNL_CAST_HALF_TO_INT32},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT16}, CNNL_CAST_HALF_TO_INT16},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT8}, CNNL_CAST_HALF_TO_INT8},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_UINT8}, CNNL_CAST_HALF_TO_UINT8},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_BOOL}, CNNL_CAST_HALF_TO_BOOL},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT64}, CNNL_CAST_HALF_TO_INT64},
    // without int32 to double
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT32_TO_FLOAT},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_HALF}, CNNL_CAST_INT32_TO_HALF},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT8}, CNNL_CAST_INT32_TO_INT8},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT16}, CNNL_CAST_INT32_TO_INT16},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT64}, CNNL_CAST_INT32_TO_INT64},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_BOOL}, CNNL_CAST_INT32_TO_BOOL},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT64}, CNNL_CAST_INT32_TO_INT64},
    // Only support INT16 to INT32, Float, Half
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT16_TO_FLOAT},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_HALF}, CNNL_CAST_INT16_TO_HALF},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_INT32}, CNNL_CAST_INT16_TO_INT32},
    // Only support INT8 to INT16, INT32, Float, Half
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT8_TO_FLOAT},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_HALF}, CNNL_CAST_INT8_TO_HALF},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_INT32}, CNNL_CAST_INT8_TO_INT32},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_INT16}, CNNL_CAST_INT8_TO_INT16},
    // Only support UINT8 to INT32, INT64, Float, Half
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_UINT8_TO_FLOAT},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_HALF}, CNNL_CAST_UINT8_TO_HALF},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_INT32}, CNNL_CAST_UINT8_TO_INT32},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_INT64}, CNNL_CAST_UINT8_TO_INT64},
    // Only support BOOL to INT32, Float, Half
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_FLOAT}, CNNL_CAST_BOOL_TO_FLOAT},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_HALF}, CNNL_CAST_BOOL_TO_HALF},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_INT32}, CNNL_CAST_BOOL_TO_INT32},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_INT32}, CNNL_CAST_INT64_TO_INT32},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT64_TO_FLOAT},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_HALF}, CNNL_CAST_INT64_TO_HALF},
    {pair{CNNL_DTYPE_DOUBLE, CNNL_DTYPE_FLOAT}, CNNL_CAST_DOUBLE_TO_FLOAT}};

bool check_amp_mode() {
    static bool has_enabled = false;
    if (!has_enabled) {
        has_enabled = at::autocast::is_mlu_enabled();
    }
    return has_enabled;
}

void cnnl_cast_internal(const at::Tensor& input, at::Tensor& output) {
  if (input.numel() == 0) return;
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // add canCast in here?
  cnnlDataType_t src_dtype = getCnnlType(input_impl);
  cnnlDataType_t dst_dtype = getCnnlType(output_impl);
  if (src_dtype == dst_dtype) {
    cnnl_copy_internal(output, input);
    return;
  }

  // currently do not support complex type related cast
  TORCH_MLU_CHECK(!input.is_complex() && !output.is_complex(),
    "CNNL cast currently do not support complex type");

  // Determine the data conversion type.
  auto iter = cast_map.find(std::pair<cnnlDataType_t, cnnlDataType_t>({src_dtype, dst_dtype}));
  if (iter == cast_map.end()) {
    // If output tensor is floating point type, then cast to float first.
    at::ScalarType internal_type = at::kFloat;
    // If input and output are both integral type, then cast to int32 first.
    if (c10::isIntegralType(input.scalar_type())
      && c10::isIntegralType(output.scalar_type())
      && output.scalar_type() != at::kByte) {
      internal_type = at::kInt;
    }
    auto tmp = at::empty_like(input, input.options().dtype(internal_type));
    cnnl_cast_internal(input, tmp);
    cnnl_cast_internal(tmp, output);
    return;
  }

  auto input_may_contiguous = input;
  auto output_may_contiguous = output;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  // note:
  //   1. input Size([1,1,1,1]), numel = 1, output Size([]), numel = 1.
  //   2. input and output have same memory format.
  if (is_same_format_tensor({output_may_contiguous, input_may_contiguous}) ||
      (output_may_contiguous.numel() == 1 && input_may_contiguous.numel() == 1)) {
    input_desc.set(input_may_contiguous,
                  std::vector<int64_t>({input_may_contiguous.numel()}),
                  std::vector<int64_t>({1}), CNNL_LAYOUT_ARRAY, src_dtype);
    output_desc.set(output_may_contiguous,
                    std::vector<int64_t>({output_may_contiguous.numel()}),
                    std::vector<int64_t>({1}), CNNL_LAYOUT_ARRAY, dst_dtype);
  } else {
    auto memory_format = output.suggest_memory_format();
    output_may_contiguous = cnnl_contiguous(output, memory_format);
    input_may_contiguous = cnnl_contiguous(input, memory_format);
    input_impl = getMluTensorImpl(input_may_contiguous);
    output_impl = getMluTensorImpl(output_may_contiguous);

    auto layout = suggest_cnnl_layout(input_may_contiguous);
    input_desc.set(input_may_contiguous, layout, src_dtype);
    output_desc.set(output_may_contiguous, layout, dst_dtype);
  }

  auto handle = getCurrentHandle();
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  cnnlCastDataType_t cast_type = iter->second;
  // check PyTorch AMP has enabled or not.
  if (cast_type == CNNL_CAST_HALF_TO_FLOAT && check_amp_mode()) {
    cast_type = CNNL_CAST_HALF_TO_FLOAT_INF;
    CNLOG(INFO) << "In AMP mode, cnnlCastDataType use cast_type"
                    " CNNL_CAST_HALF_TO_FLOAT_INF"
                    " instead of CNNL_CAST_HALF_TO_FLOAT.";
  }

  TORCH_CNNL_CHECK(cnnlCastDataType(handle, input_desc.desc(), input_ptr,
                                    cast_type, output_desc.desc(), output_ptr));

  if (is_copy_necessary(output, output_may_contiguous)) {
    output.copy_(output_may_contiguous);
  }

  return;
}

}  // namespace ops
}  // namespace torch_mlu
