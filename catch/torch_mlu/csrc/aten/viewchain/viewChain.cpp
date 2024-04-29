/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
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

#include "aten/viewchain/viewChain.h" // NOLINT
#include "framework/core/tensor_impl.h" // NOLINT
#include "aten/utils/tensor_util.h" // NOLINT

namespace torch_mlu {

ViewChain::ViewChain() {
  // lambda func is to get environment variable DISABLE_VIEWCHAIN_FUNC,
  // which is to control whether to use view chain func.
  static const bool b_enabled = ([](){
    bool enabled = false;
    const char* e = std::getenv("DISABLE_VIEWCHAIN_FUNC");
    if (e != nullptr && (strcmp(e, "on") == 0 ||
        strcmp(e, "ON") == 0 || strcmp(e, "1") == 0)) {
      enabled = true;
    }
    return enabled;
  })();
  b_closeChainFunc = b_enabled;
}

ViewChain::ViewChain(const ViewChain& other) {
  // if Graph Func is close, no need to store view chain info.
  setViewChainValue(other.v_viewChain);
  b_closeChainFunc = other.b_closeChainFunc;
  b_isFused = false;
}

ViewChain& ViewChain::operator=(const ViewChain& other) {
  // if Graph Func is close, no need to store view chain info.
  if (this == &other) {
    return *this;
  }
  setViewChainValue(other.v_viewChain);
  b_closeChainFunc = other.b_closeChainFunc;
  b_isFused = other.b_isFused;
  return *this;
}

// More details in .h file.
bool ViewChain::isViewChainContiguous(const BaseViewOp::baseViewOpPtr& ptr) {
  if (this->getViewChainNodeSize() == 0) {
    return true;
  }
  const BaseViewOp::baseViewOpPtr& last_ptr = this->v_viewChain.back();
  // Whether tensor info is same in two adjacent view node.
  // TODO(shangang): If size and stride is same, but storage offset is different.
  // Maybe add a slice view op to keep view chain contiguous.
  if (last_ptr->getOutputTensorInfo() == ptr->getInputTensorInfo()) {
    return true;
  }
  return false;
}

void ViewChain::pushNodeToViewChain(const at::Tensor& self,
                                    const at::Tensor& output,
                                    const BaseViewOp::baseViewOpPtr& ptr) {
  // Update tensor info in view chain.
  ptr->updateTensorInfo(self, output);
  // If self is contiguous tensor or view chain is not contigous,
  // clear viewChain.
  if (self.is_contiguous(self.suggest_memory_format()) ||
    !isViewChainContiguous(ptr)) {
    this->clearViewChain();
  }
  // Push a view node to view chain and set isFused to false.
  this->b_isFused = false;
  this->v_viewChain.emplace_back(ptr);
}

void ViewChain::pushNodeToViewChain(const std::vector<int64_t>& i_size,
                                    const std::vector<int64_t>& i_stride,
                                    int64_t i_storage_offset,
                                    const std::vector<int64_t>& o_size,
                                    const std::vector<int64_t>& o_stride,
                                    int64_t o_storage_offset,
                                    const BaseViewOp::baseViewOpPtr& ptr) {
  // Update tensor info in view chain.
  ptr->updateTensorInfo(TensorInfo(i_size, i_stride, i_storage_offset),
                        TensorInfo(o_size, o_stride, o_storage_offset));
  // If self is contiguous tensor or view chain is not contigous,
  // clear viewChain.
  if (torch_mlu::is_geometry_contiguous(i_size, i_stride) ||
    !isViewChainContiguous(ptr)) {
    this->clearViewChain();
  }
  // Push a view node to view chain and set isFused to false.
  this->b_isFused = false;
  this->v_viewChain.emplace_back(ptr);
}

bool ViewChain::canRunViewChain(const at::Tensor& other,
                                c10::optional<at::Tensor> output_opt) {
  // TODO(shangang): Interface with output tensor is not incomplete support.
  // After all view op inplace optimization is supported, then output interface
  // will be more smooth.
  TORCH_MLU_CHECK(!output_opt.has_value(), "output tensor is not support now.");
  // Using copy kernel to get contiguous tensor.
  // a. Check view chain is not empty.
  // b. Check b_closeChainFunc status.
  if (this->getViewChainNodeSize() == 0 || this->b_closeChainFunc == true) {
    return false;
  }
  // c. check output tensor size and strides with view chain last node stored info.
  const auto& lastNodeOutputSizes = this->v_viewChain.back()->getOutputTensorInfo().v_sizes;
  const auto& lastNodeOutputStrides = this->v_viewChain.back()->getOutputTensorInfo().v_strides;
  if (other.sizes().size() != lastNodeOutputSizes.size() ||
      !std::equal(other.sizes().begin(), other.sizes().end(), lastNodeOutputSizes.begin()) ||
      !std::equal(other.strides().begin(), other.strides().end(), lastNodeOutputStrides.begin())) {
    return false;
  }
  // If a,b,c is statisfied, this means tensor and view chain is not changed.
  // So check isFused status, if status is true, this tensor can using view chain optimization.
  // Otherwise need to check again.
  if (this->b_isFused) return true;
  // d. root tensor need be contiguous tensor.
  at::Tensor ori_tensor = restoreRootTensor(other);
  if (!ori_tensor.is_contiguous(ori_tensor.suggest_memory_format())) {
    return false;
  }
  // e. Check view chain status. If return false, then some view op don't
  // have specific IO kernel, call copy instead call serial IO kernels.
  for (const auto& ptr : this->v_viewChain) {
    if (!ptr->hasCnnlSpecificFunction()) {
      return false;
    }
  }
  return true;
}

at::Tensor ViewChain::runViewChain(const at::Tensor& other,
                                   c10::optional<at::Tensor> output_opt) {
  // restore Root Tensor.
  at::Tensor ori_tensor = restoreRootTensor(other);
  // Fused view node if possiable.
  if (!this->b_isFused && viewChainFuserManager.canFusedViewChain(this->v_viewChain)) {
    this->v_viewChain = viewChainFuserManager.runFusedViewChain(this->v_viewChain);
    this->b_isFused = true;
  }
  // lambda func is to get environment variable ENABLE_PRINT_VIEW_CHAIN,
  // which is to control whether to print view chain info.
  static const bool b_printViewChain = ([](){
    bool enabled = false;
    const char* e = std::getenv("ENABLE_PRINT_VIEW_CHAIN");
    if (e != nullptr && (strcmp(e, "on") == 0 ||
        strcmp(e, "ON") == 0 || strcmp(e, "1") == 0)) {
      enabled = true;
    }
    return enabled;
  })();
  if (b_printViewChain) {
    CNLOG(INFO) << *this << std::endl;
  }

  // TODO(shangang): Interface with output tensor is not incomplete support.
  // After all view op inplace optimization is supported, then output interface
  // will be more smooth.
  TORCH_MLU_CHECK(!output_opt.has_value(), "output tensor is not support now.");
  at::Tensor output = ori_tensor;
  for (int i = 0 ; i < this->v_viewChain.size(); ++i) {
    output = this->v_viewChain[i]->runCnnlSpecificFunction(output, c10::nullopt);
  }
  return output;
}

// Restore root Tensor based on Root Tensor Info and target Tensor storage.
at::Tensor ViewChain::restoreRootTensor(const at::Tensor& self) {
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  const auto& tensor_info = (this->v_viewChain.front())->getInputTensorInfo();
  impl->set_storage_offset(tensor_info.i_storageOffset);
  impl->set_sizes_and_strides(tensor_info.v_sizes,
                              tensor_info.v_strides);
  at::Tensor restoreTensor = at::Tensor(std::move(impl));
  return restoreTensor;
}

// Convert view op info to string.
std::string ViewChain::convertViewChainToString() const {
  std::stringstream out;
  if (this->getViewChainNodeSize() == 0) {
    out << "ViewChain().";
    return out.str();
  }
  int count = 1;
  for (const auto& ptr : this->v_viewChain) {
    out << "\n" << std::to_string(count) + "st op: " << *ptr;
    count++;
  }
  return out.str();
}

// Visual view op in view chain for debugging.
std::ostream& operator<<(std::ostream& out, const ViewChain& other) {
  out << other.convertViewChainToString();
  return out;
}

}  // end of namespace torch_mlu

