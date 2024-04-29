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

#pragma once

#include <iostream>
#include <string>
#include "aten/viewchain/viewChainFuser.h"

/**
 * Note [viewChain]
 * ~~~~~~~~~~~~~~~~
 * ViewChain is a class to store view ops info and root tensor info.
 * example: DFG(data flow graph) is like this:
 * Root tensor --> permute op --> slice op --> relu op.
 * ViewChain stores root tensor info and view op info which include permute
 * and slice op info.
 * ViewChain also can call special IO kernel to create a contiguous tensor
 * based on a root tensor.
 * 
 * ViewChain is accompanied with each mlu Tensor, and view chain will
 * always be a single chain.
 * Before set mlu tensor to cnnl kernel, CATCH will call cnnl_contiguous
 * with input or output tensors to create contiguous tensor instead 
 * discontiguous tensor. In cnnl_contigous op, view chain in tensor 
 * metdate will paly a great role.
 * 
 * Why need this?
 * Discontiguous tensor are not supported by cnnl almost kernels, or with low
 * performance.
 * 
 * Using environment variable DISABLE_VIEWCHAIN_FUNC to control whether to use
 * this special IO Kernel. This is switch is working at global level, only update
 * status when CACHE start.
 * 
 */

/**
 * Note [PrintViewChain]
 * ~~~~~~~~~~~~~~~~
 * The main purpose of view chain printing function is to debug problems
 * or IO analysis.
 * PrintViewChain will not work, if Tensor can be processed by special
 * IO functions or simple permute functions.
 *
 * If you need check view chain info, you need to export print view chain enviroment
 * before run network or ops.
 * example:
 *   export ENABLE_PRINT_VIEW_CHAIN=ON
 *   export ENABLE_PRINT_VIEW_CHAIN=on
 *   export ENABLE_PRINT_VIEW_CHAIN=1
 *
 * Default function of print view chain is closed.
 *
 */

namespace torch_mlu {

class ViewChain {
  public:
    friend class viewChainFuser;
    ViewChain();
    explicit ViewChain(const ViewChain& other);
    ViewChain& operator=(const ViewChain& other);
    inline bool operator==(const ViewChain& other) {
      return isEqual(other);
    }
    inline bool operator!=(const ViewChain& other) {
      return !((*this) == other);
    }

    // Push a new View Op node to ViewChain, and update Root
    // Node Tensor info if Root Node Tensor is not updated.
    void pushNodeToViewChain(const at::Tensor& self,
                             const at::Tensor& output,
                             const BaseViewOp::baseViewOpPtr& ptr);
    // Used for inplace op.
    void pushNodeToViewChain(const std::vector<int64_t>& i_size,
                             const std::vector<int64_t>& i_stride,
                             int64_t i_storage_offset,
                             const std::vector<int64_t>& o_size,
                             const std::vector<int64_t>& o_stride,
                             int64_t o_storage_offset,
                             const BaseViewOp::baseViewOpPtr& ptr);

    // Check whether each view op has a specific function
    // stored in ViewChain. Also check root tensor need be
    // contiguous tensor.
    // Parameter output is added for out situation, and also need
    // be contiguous.
    bool canRunViewChain(const at::Tensor& other,
                         c10::optional<at::Tensor> output = c10::nullopt);

    // Based on result of canRunViewChain();
    // if true, run each specific function stored in v_viewChain;
    // else call copy kernel to finished memory transport.
    at::Tensor runViewChain(const at::Tensor& other,
                            c10::optional<at::Tensor> output = c10::nullopt);

    // get v_viewChain node size
    inline int getViewChainNodeSize() const {
      return this->v_viewChain.size();
    }

    ~ViewChain() {
      clearViewChain();
    }

    // Convert view op info to string.
    std::string convertViewChainToString() const;

    // Visual view op in view chain for debugging.
    friend std::ostream& operator<<(std::ostream& out, const ViewChain& other);

  private:
    // Store view op node info.
    c10::SmallVector<BaseViewOp::baseViewOpPtr, MAX_STORE_INFO_NUM> v_viewChain;
    // Running DGF with view graph or not.
    bool b_closeChainFunc = false;
    // Whether has optimizer processing been performed.
    // After the optimizer completes execution, it will be setted to true.
    // This parameter should be changed to false if the following two conditions
    // occur. 1. Any new node pushed to view chain; 2. Tensor sizes and strides are
    // changed.
    bool b_isFused = false;

  private:
    // If view chain is not contiguous, so clear view chain.
    // Note view ops are series pushed into view chain, so output tensor
    // info stored in last view op need be same with input tensor info
    // in current view node.
    bool isViewChainContiguous(const BaseViewOp::baseViewOpPtr& ptr);

    // Restore Root Tensor based on t_rootTensorInfo.
    at::Tensor restoreRootTensor(const at::Tensor& self);

    // update viewChain info by using other one.
    inline void setViewChainValue(
      const c10::SmallVector<BaseViewOp::baseViewOpPtr, MAX_STORE_INFO_NUM>& other) {
      v_viewChain = other;
    }

    // ViewChain is equal or not.
    inline bool isEqual(const ViewChain& other) {
      const int v_size = this->v_viewChain.size();
      if (other.v_viewChain.size() != v_size ||
          this->b_closeChainFunc != other.b_closeChainFunc) {
        return false;
      }
      for (int i = 0; i < v_size; ++i) {
        if (*(this->v_viewChain[i]) != *(other.v_viewChain[i])) {
          return false;
        }
      }
      return true;
    }

    // Clear view chain when destroy class or found another
    // view chain root node.
    inline void clearViewChain() {
      for (auto& ptr : v_viewChain) {
        ptr.reset();
      }
      v_viewChain.clear();
    }

    // Debug to close view chain function.
    inline bool isCloseViewChain() {
      return b_closeChainFunc;
    }
};

}  // namespace torch_mlu


