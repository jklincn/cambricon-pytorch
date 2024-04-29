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

#include "aten/viewchain/viewNodeFuser.h"
#include "aten/viewchain/specificViewOps.h"

/**
 * Note [PermuteFusionOpt]
 * ~~~~~~~~~~~~~~~~
 * PermuteFusionOpt is a series of permute optimization methods.
 * 
 * example1: 
 * original view chain: permute + permute + permute
 * after fusion: permute
 *
 * example2:
 * original view chain: permute
 * after replace: reshape
 * 
 */

namespace torch_mlu {

class PermuteFusionOpt: public ViewNodeFusionOpt {
  public:
    // Fusion optimization.
    BaseViewOp::baseViewOpPtr
    runViewNodeFusionOpt(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) override {
      return getFusedPermute(other);
    }

    // Replace optimization.
    BaseViewOp::baseViewOpPtr
    runViewNodeReplaceOpt(const BaseViewOp::baseViewOpPtr& ptr) override {
      if (isPermuteOneValueDim(ptr)) {
        return insteadPermuteToReshape(ptr);
      }
      return ptr;
    }

    // Move optimization
    // Permute node don't need movement optimization, all about permute movement will be
    // handled by slice op, expand op and reshape op. Maybe unfold op and diagonal op in
    // the future.
    std::vector<BaseViewOp::baseViewOpPtr>
    runViewNodeMoveOpt(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      // First to fused same permute in view chain. this same node maybe from original
      // view chain, or cause by other op movement optimization.
      auto indexOfContinueNodes = indexOfSpecificContinueViewNode(other, ViewsType::kPermute);
      int startIndex = indexOfContinueNodes.size() - 1;
      while (startIndex >= 0) {
        const auto& pairNode = indexOfContinueNodes[startIndex];
        BaseViewOp::baseViewOpPtr permuteOpPtr = getFusedPermute({other.begin() + pairNode.first,
                                                                other.begin() + pairNode.second});
        other[pairNode.first] = permuteOpPtr;
        other.erase(other.begin() + pairNode.first + 1, other.begin() + pairNode.second);
        --startIndex;
      }
      // Using reshape to instead permute.
      auto indexOfNodes = indexOfSpecificViewNode(other, ViewsType::kPermute);
      for (auto& index : indexOfNodes) {
        other[index] = runViewNodeReplaceOpt(other[index]);
      }
      return other;
    }

  private:
    // Fused Permute view node op.
    BaseViewOp::baseViewOpPtr getFusedPermute(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) const {
      const PermuteOp* const firstNodeRawPtr = dynamic_cast<PermuteOp*>(other.front().get());
      const auto& firstNodeDims = firstNodeRawPtr->v_dims;
      const int ndim = firstNodeDims.size();
      std::vector<int64_t> newPermuteDims(firstNodeDims);
      for (int i = 1; i < other.size(); ++i) {
        const PermuteOp* const tmpRawPtr = dynamic_cast<PermuteOp*>(other[i].get());
        const auto& tmpNodeDims = tmpRawPtr->v_dims;
        TORCH_MLU_CHECK(ndim == tmpNodeDims.size(),
          "Permute view nodes parameter size need be equal.");
        std::vector<int64_t> tmpPermuteDims(ndim);
        for (int i = 0; i < ndim; ++i) {
          tmpPermuteDims[i] = newPermuteDims[tmpNodeDims[i]];
        }
        newPermuteDims = tmpPermuteDims;
      }
      BaseViewOp::baseViewOpPtr permutePtr = std::make_shared<PermuteOp>(newPermuteDims);
      permutePtr->updateInputTensorInfo(firstNodeRawPtr->getInputTensorInfo());
      permutePtr->updateOutputTensorInfo(other.back()->getOutputTensorInfo());
      return permutePtr;
    }

    // Using reshape to replace permute.
    BaseViewOp::baseViewOpPtr insteadPermuteToReshape(
        const BaseViewOp::baseViewOpPtr& ptr) const {
      BaseViewOp::baseViewOpPtr reshapePtr =
        std::make_shared<ReshapeOp>(ptr->getOutputTensorInfo().v_sizes);
      reshapePtr->updateInputTensorInfo(ptr->getInputTensorInfo());
      reshapePtr->updateOutputTensorInfo(ptr->getOutputTensorInfo());
      return reshapePtr;
    }

    // If a tensor has on any dim the size of 1, we can shift the dim either
    // leftward or rightward in any steps, the result of which can be viewed
    // as a reshape of the original tensor:
    // for example:
    // The orginal tensor that has the sizes (3, 4, 1, 6, 7)
    // after shifting, it can be in any of the following shapes:
    // (3, 1, 4, 6, 7), (1, 3, 4, 6, 7), (3, 4, 6, 1, 7) and (3, 4, 6, 7, 1)
    // all of which effectively reshaped  (3, 4, 1, 6, 7)
    // So (3, 1, 4, 6, 7), (1, 3, 4, 6, 7), (3, 4, 6, 1, 7) and (3, 4, 6, 7, 1)
    // is just reshape of tensor. And there permute parameters is:
    // (3, 1, 4, 6, 7) permute dims: (0, 2, 1, 3, 4)
    // (1, 3, 4, 6, 7) permute dims: (2, 0, 1, 3, 4)
    // (3, 4, 6, 1, 7) permute dims: (0, 1, 3, 2, 4)
    // (3, 4, 6, 7, 1) permute dims: (0, 1, 3, 4, 2)
    // If we look at those permute dims data carefully, we will find that these data
    // are in order except one value dim.
    bool isPermuteOneValueDim(const BaseViewOp::baseViewOpPtr& ptr) {
      const auto& inputSizes = ptr->getInputTensorInfo().v_sizes;
      const PermuteOp* const rawPtr = dynamic_cast<PermuteOp*>(ptr.get());
      const auto& permuteDims = rawPtr->v_dims;
      const int nDims = inputSizes.size();
      int previousIndex = -1;
      for (int i = 0; i < nDims; ++i) {
        int dimIndex = permuteDims[i];
        if (inputSizes[dimIndex] != 1) {
          if (dimIndex <= previousIndex) {
            return false;
          }
          previousIndex = dimIndex;
        }
      }
      return true;
    }
};

REGISTER_VIEW_NODE_OPT(ViewsType::kPermute, PermuteFusionOpt);

}  // end of namespace torch_mlu
