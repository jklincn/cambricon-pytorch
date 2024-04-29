#include <ATen/autocast_mode.h>

#include <iostream>
#include <exception>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>

namespace {
using namespace at::autocast;

/*******************************
Banned functions
*******************************/

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.nms.nms");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
}

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return nms(
      at::autocast::cached_cast(at::kFloat, dets, c10::DeviceType::MLU),
      at::autocast::cached_cast(at::kFloat, scores, c10::DeviceType::MLU),
      iou_threshold);
}

at::Tensor roi_align(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) // The flag for pixel shift
// along each axis.
{
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.roi_align.roi_align");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return roi_align(
             at::autocast::cached_cast(at::kFloat, input, c10::DeviceType::MLU),
             at::autocast::cached_cast(at::kFloat, rois, c10::DeviceType::MLU),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::roi_align"), TORCH_FN(roi_align_autocast));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_autocast));
}

static at::Tensor binary_cross_entropy_banned(const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}

TORCH_LIBRARY_IMPL(_, AutocastMLU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastMLU, m) {
  // lower_precision_fp
  KERNEL_MLU(ADD_NS(_convolution), "_convolution.deprecated", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL_MLU(ADD_NS(_convolution), "_convolution", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t, bool, bool, bool, bool), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv1d), "conv1d", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv2d), "conv2d", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv3d), "conv3d", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv_tbc), "conv_tbc", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv_transpose1d), "conv_transpose1d", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::IntArrayRef), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv_transpose2d), "conv_transpose2d.input", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::IntArrayRef), lower_precision_fp)
  KERNEL_MLU(ADD_NS(conv_transpose3d), "conv_transpose3d.input", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::IntArrayRef), lower_precision_fp)
  KERNEL_MLU(ADD_NS(convolution), "convolution", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, at::IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_MLU(ADD_NS(cudnn_convolution), "cudnn_convolution", at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL_MLU(ADD_NS(cudnn_convolution_transpose), "cudnn_convolution_transpose", at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL_MLU(ADD_NS(prelu), "prelu", at::Tensor (const at::Tensor &, const at::Tensor &), lower_precision_fp)
  KERNEL_MLU(ADD_NS(addmm), "addmm", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(addmv), "addmv", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(addr), "addr", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(matmul), "matmul", at::Tensor (const at::Tensor &, const at::Tensor &), lower_precision_fp)
  KERNEL_MLU(ADD_NS(einsum), "einsum", at::Tensor (c10::string_view, at::TensorList, at::OptionalIntArrayRef), lower_precision_fp)
  KERNEL_MLU(ADD_NS(mm), "mm", at::Tensor (const at::Tensor &, const at::Tensor &), lower_precision_fp)
  KERNEL_MLU(ADD_NS(mv), "mv", at::Tensor (const at::Tensor &, const at::Tensor &), lower_precision_fp)
  KERNEL_MLU(ADD_NS(linear), "linear", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(addbmm), "addbmm", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(baddbmm), "baddbmm", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&), lower_precision_fp)
  KERNEL_MLU(ADD_NS(bmm), "bmm", at::Tensor (const at::Tensor &, const at::Tensor &), lower_precision_fp)
  KERNEL_MLU(ADD_NS(chain_matmul), "chain_matmul", at::Tensor (at::TensorList), lower_precision_fp)
  KERNEL_MLU(ADD_NS(linalg_multi_dot), "linalg_multi_dot", at::Tensor (at::TensorList), lower_precision_fp)
  // The macro doesn't like these (I think it chokes on commas inside <>) so write them manually
  m.impl(TORCH_SELECTIVE_NAME("aten::_thnn_fused_lstm_cell"),
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(_thnn_fused_lstm_cell)>::type::call)));
  m.impl("_thnn_fused_gru_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(_thnn_fused_gru_cell)>::type::call)));
  m.impl("lstm_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::TensorList, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::TensorList, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(lstm_cell)>::type::call)));
  m.impl("gru_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(gru_cell)>::type::call)));
  m.impl("rnn_tanh_cell",  // tanh unary op is executed as a cuda math library call.
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(rnn_tanh_cell)>::type::call)));
  m.impl("rnn_relu_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, at::DeviceType::MLU,
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&),
                                 &ADD_NS(rnn_relu_cell)>::type::call)));
  // fp32
  KERNEL_MLU(ADD_NS(acos), "acos", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(asin), "asin", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(cosh), "cosh", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(erfinv), "erfinv", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(exp), "exp", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(expm1), "expm1", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(log), "log", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(log10), "log10", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(log2), "log2", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(log1p), "log1p", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(reciprocal), "reciprocal", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(rsqrt), "rsqrt", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(sinh), "sinh", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(tan), "tan", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(pow), "pow.Tensor_Scalar", at::Tensor (const at::Tensor &, const at::Scalar&), fp32)
  KERNEL_MLU(ADD_NS(pow), "pow.Tensor_Tensor", at::Tensor (const at::Tensor &, const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(pow), "pow.Scalar", at::Tensor (const at::Scalar&, const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(softplus), "softplus", at::Tensor (const at::Tensor &, const at::Scalar&, const at::Scalar&), fp32)
  KERNEL_MLU(ADD_NS(layer_norm), "layer_norm", at::Tensor (const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, double, bool), fp32)
  // The macro doesn't like this one (I think it chokes on commas inside <>) so write it manually
  m.impl(TORCH_SELECTIVE_NAME("aten::native_layer_norm"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, at::DeviceType::MLU,
                                 std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor&, at::IntArrayRef, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, double),
                                 std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor&, at::IntArrayRef, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, double),
                                 &ADD_NS(native_layer_norm)>::type::call)));
  KERNEL_MLU(ADD_NS(group_norm), "group_norm", at::Tensor (const at::Tensor &, int64_t, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, double, bool), fp32)
  KERNEL_MLU(ADD_NS(frobenius_norm), "frobenius_norm", at::Tensor (const at::Tensor &), fp32)
  KERNEL_MLU(ADD_NS(frobenius_norm), "frobenius_norm.dim", at::Tensor (const at::Tensor &, at::IntArrayRef, bool), fp32)
  KERNEL_MLU(ADD_NS(nuclear_norm), "nuclear_norm", at::Tensor (const at::Tensor &, bool), fp32)
  KERNEL_MLU(ADD_NS(nuclear_norm), "nuclear_norm.dim", at::Tensor (const at::Tensor &, at::IntArrayRef, bool), fp32)
  KERNEL_MLU(ADD_NS(cosine_similarity), "cosine_similarity", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, double), fp32)
  KERNEL_MLU(ADD_NS(poisson_nll_loss), "poisson_nll_loss", at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, double, int64_t), fp32)
  KERNEL_MLU(ADD_NS(cosine_embedding_loss), "cosine_embedding_loss", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t), fp32)
  KERNEL_MLU(ADD_NS(nll_loss), "nll_loss", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, int64_t, int64_t), fp32)
  KERNEL_MLU(ADD_NS(nll_loss2d), "nll_loss2d", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, int64_t, int64_t), fp32)
  KERNEL_MLU(ADD_NS(hinge_embedding_loss), "hinge_embedding_loss", at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t), fp32)
  KERNEL_MLU(ADD_NS(kl_div), "kl_div", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, bool), fp32)
  KERNEL_MLU(ADD_NS(l1_loss), "l1_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t), fp32)
  KERNEL_MLU(ADD_NS(smooth_l1_loss), "smooth_l1_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, double), fp32)
  KERNEL_MLU(ADD_NS(huber_loss), "huber_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, double), fp32)
  KERNEL_MLU(ADD_NS(mse_loss), "mse_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t), fp32)
  KERNEL_MLU(ADD_NS(margin_ranking_loss), "margin_ranking_loss", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, double, int64_t), fp32)
  KERNEL_MLU(ADD_NS(multilabel_margin_loss), "multilabel_margin_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t), fp32)
  KERNEL_MLU(ADD_NS(soft_margin_loss), "soft_margin_loss", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t), fp32)
  KERNEL_MLU(ADD_NS(triplet_margin_loss), "triplet_margin_loss", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, double, double, double, bool, int64_t), fp32)
  KERNEL_MLU(ADD_NS(multi_margin_loss), "multi_margin_loss", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar&, const at::Scalar&, const c10::optional<at::Tensor>&, int64_t), fp32)
  KERNEL_MLU(ADD_NS(binary_cross_entropy_with_logits), "binary_cross_entropy_with_logits", at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, int64_t), fp32)
  KERNEL_MLU(ADD_NS(dist), "dist", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar&), fp32)
  KERNEL_MLU(ADD_NS(pdist), "pdist", at::Tensor (const at::Tensor &, double), fp32)
  KERNEL_MLU(ADD_NS(cdist), "cdist", at::Tensor (const at::Tensor &, const at::Tensor &, double, c10::optional<int64_t>), fp32)
  KERNEL_MLU(ADD_NS(renorm), "renorm", at::Tensor (const at::Tensor &, const at::Scalar&, int64_t, const at::Scalar&), fp32)
  KERNEL_MLU(ADD_NS(logsumexp), "logsumexp", at::Tensor (const at::Tensor &, at::IntArrayRef, bool), fp32)
  // fp32_set_opt_dtype
  KERNEL_MLU(ADD_NS(prod), "prod", at::Tensor (const at::Tensor &, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(prod), "prod.dim_int", at::Tensor (const at::Tensor &, int64_t, bool, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(prod), "prod.dim_Dimname", at::Tensor (const at::Tensor &, at::Dimname, bool, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(softmax), "softmax.int", at::Tensor (const at::Tensor &, int64_t, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(softmax), "softmax.Dimname", at::Tensor (const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(log_softmax), "log_softmax.int", at::Tensor (const at::Tensor &, int64_t, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(log_softmax), "log_softmax.Dimname", at::Tensor (const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(cumprod), "cumprod", at::Tensor (const at::Tensor &, int64_t, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(cumprod), "cumprod.dimname", at::Tensor (const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(cumsum), "cumsum", at::Tensor (const at::Tensor &, int64_t, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(cumsum), "cumsum.dimname", at::Tensor (const at::Tensor &, at::Dimname, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_MLU(ADD_NS(norm), "norm.at::ScalarOpt_dtype", at::Tensor (const at::Tensor &, c10::optional<at::Scalar>, at::ScalarType), fp32_set_opt_dtype)
  // KERNEL_MLU(ADD_NS(norm), "norm.at::ScalarOpt_dim_dtype", at::Tensor (const at::Tensor &, c10::optional<at::Scalar>, at::IntArrayRef, bool, at::ScalarType), fp32_set_opt_dtype)
  // KERNEL_MLU(ADD_NS(norm), "norm.names_at::ScalarOpt_dim_dtype", at::Tensor (const at::Tensor &, c10::optional<at::Scalar>, at::DimnameList, bool, at::ScalarType), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(sum), "sum", at::Tensor (const at::Tensor &, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(sum), "sum.dim_IntList", at::Tensor (const at::Tensor &, at::OptionalIntArrayRef, bool, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  KERNEL_MLU(ADD_NS(sum), "sum.dim_DimnameList", at::Tensor (const at::Tensor &, at::DimnameList, bool, c10::optional<at::ScalarType>), fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_MLU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.Scalar", at::Tensor (const at::Tensor &, const at::Scalar&), at::Tensor (const at::Tensor &, const c10::optional<at::Scalar>&, at::ScalarType), fp32_append_dtype)
  KERNEL_MLU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.ScalarOpt_dim", at::Tensor (const at::Tensor &, const c10::optional<at::Scalar>&, at::IntArrayRef, bool), at::Tensor (const at::Tensor &, const c10::optional<at::Scalar>&, at::IntArrayRef, bool, at::ScalarType), fp32_append_dtype)
  KERNEL_MLU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.names_ScalarOpt_dim", at::Tensor (const at::Tensor &, const c10::optional<at::Scalar>&, at::DimnameList, bool), at::Tensor (const at::Tensor &, const c10::optional<at::Scalar>&, at::DimnameList, bool, at::ScalarType), fp32_append_dtype)
  // promote
  KERNEL_MLU(ADD_NS(addcdiv), "addcdiv", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&), promote)
  KERNEL_MLU(ADD_NS(addcmul), "addcmul", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar&), promote)
  KERNEL_MLU(ADD_NS(atan2), "atan2", at::Tensor (const at::Tensor &, const at::Tensor &), promote)
  KERNEL_MLU(ADD_NS(bilinear), "bilinear", at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor>&), promote)
  KERNEL_MLU(ADD_NS(cross), "cross", at::Tensor (const at::Tensor &, const at::Tensor &, c10::optional<int64_t>), promote)
  KERNEL_MLU(ADD_NS(dot), "dot", at::Tensor (const at::Tensor &, const at::Tensor &), promote)
  KERNEL_MLU(ADD_NS(grid_sampler), "grid_sampler", at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, int64_t, bool), promote)
  KERNEL_MLU(ADD_NS(index_put), "index_put", at::Tensor (const at::Tensor &, const torch::List<c10::optional<at::Tensor>>&, const at::Tensor &, bool), promote)
  KERNEL_MLU(ADD_NS(tensordot), "tensordot", at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef), promote)
  KERNEL_MLU(ADD_NS(scatter_add), "scatter_add", at::Tensor (const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&), promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&binary_cross_entropy_banned)));
}

}  // namespace
