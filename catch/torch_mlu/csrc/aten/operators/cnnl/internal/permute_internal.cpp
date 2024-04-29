#include "aten/operators/cnnl/internal/cnnl_internal.h"
//#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace ops {

// permute behaviour is same with cpu side.
/* example:
  contiguous tensor input with sizes (2, 3, 4, 2), strides (24 ,8 ,2 ,1);
  std::vector<int64> permute({0, 2, 3, 1});
  temp_output = at::permute(input, permute);
  output = cnnl_contiguous(temp_output, MemoryFormat);
  detail:
    temp_output is not contigous, and sizes (2, 4, 2, 3) and strides (24, 2, 1, 8);
    if u need contiguous tensor with special MemoryFormat, need using like:
    output = at::permute(input, permute).contiguous(MemoryFormat);
    Python side:
      >>> a.size() original tensor
      torch.Size([2, 3, 4, 2])
      >>> a.stride()
      (24, 8, 2, 1)
      >>> b.size() permute tensor
      torch.Size([2, 4, 2, 3])
      >>> b.stride()
      (24, 2, 1, 8)
      >>> c.size() b.contiguous()
      torch.Size([2, 4, 2, 3])
      >>> c.stride()
      (24, 6, 3, 1)
      >>> d.size() b.contiguous(memory_format=torch.channels_last)
      torch.Size([2, 4, 2, 3])
      >>> d.stride()
      (24, 1, 12, 4)
*/

at::Tensor& cnnl_permute_out_internal(at::Tensor& output,
                                     const at::Tensor& self,
                                     at::IntArrayRef dims) {
  int p_dims = self.dim();
  TORCH_MLU_CHECK(p_dims == dims.size(),
    "number of dims don't match in permute.");
  TORCH_MLU_CHECK(self.is_contiguous(c10::MemoryFormat::Contiguous),
    "Self tensor Only support channels first contiguous.");

  std::vector<int> n_dims(p_dims, 0);
  for (int i = 0; i < p_dims; ++i) {
    n_dims[i] = ::at::maybe_wrap_dim(dims[i], p_dims);
  }

  auto permute = n_dims;
  auto sort_permute = permute;
  std::sort(sort_permute.begin(), sort_permute.end(), std::less<int64_t>());
  if (permute == sort_permute) {
    cnnl_copy_internal(output, self);
    return output;
  }

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self);
  output_desc.set(output);

  std::vector<int> cnnl_permute(p_dims, 0);
  for (int i = 0; i < p_dims; ++i) {
    cnnl_permute[i] = static_cast<int>(permute[i]);
  }
  CnnlTransposeDescriptor trans_desc;
  trans_desc.set(p_dims, cnnl_permute.data());

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // Get workspace
  at::Tensor trans_workspace;
  size_t workspace_size = 0;
  void* workspace_ptr = nullptr;
  cnnlGetTransposeWorkspaceSize(handle, input_desc.desc(),
                                trans_desc.desc(), &workspace_size);
  if (workspace_size != 0) {
    trans_workspace = at::empty({static_cast<long>(workspace_size)},
                                self.options().dtype(at::kByte));
    auto workspace_impl = getMluTensorImpl(trans_workspace);
    workspace_ptr = workspace_impl->mlu_data_ptr();
  }

  TORCH_CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc.desc(), input_desc.desc(),
                                    input_ptr, output_desc.desc(), output_ptr,
                                    workspace_ptr, workspace_size));
  return output;
}

at::Tensor cnnl_permute_internal(const at::Tensor& self,
                                 at::IntArrayRef dims) {
  int p_dims = self.dim();
  TORCH_MLU_CHECK(p_dims == dims.size(),
    "number of dims don't match in permute.");
  TORCH_MLU_CHECK(self.is_contiguous(c10::MemoryFormat::Contiguous),
    "Self tensor Only support channels first contiguous.");

  std::vector<int> n_dims(p_dims, 0); 
  for (int i = 0; i < p_dims; ++i) {
    n_dims[i] = ::at::maybe_wrap_dim(dims[i], p_dims);
  }

  auto permute = n_dims;

  // input
  auto input_impl = getMluTensorImpl(self);

  // output
  auto input_size = self.sizes().vec();
  std::vector<int64_t> output_size(p_dims);
  for (int i = 0; i < p_dims; ++i) {
    output_size[i] = static_cast<int64_t>(input_size[permute[i]]);
  }
  // output is CF contiguous
  auto output = at::empty(output_size, self.options(), c10::MemoryFormat::Contiguous);
  return cnnl_permute_out_internal(output, self, dims);
}

}  // namespace ops
}  // namespace torch_mlu
