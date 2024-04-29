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
 * Note [SliceFusionOpt]
 * ~~~~~~~~~~~~~~~~
 * SliceFusionOpt is a series of slice optimization methods.
 * 
 * example1:  fusion optimization
 * original view chain: slice + slice + slice
 * after fusion: slice
 * 
 * example2: move optimization
 * original view chain: permute + slice
 * after movement: slice + permute
 * original view chain: expand + slice
 * after movement: slice + expand
 * original view chain: reshape + slice
 * after movement: slice + reshape.
 *
 */

namespace torch_mlu {

class SliceFusionOpt: public ViewNodeFusionOpt {
  public:
    // Fusion optimization.
    BaseViewOp::baseViewOpPtr
    runViewNodeFusionOpt(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) override {
      return getFusedSlice(other);
    }

    // Replace optimization.
    BaseViewOp::baseViewOpPtr
    runViewNodeReplaceOpt(const BaseViewOp::baseViewOpPtr& ptr) override {
      return inplaceSpecialSliceToReshape(ptr);
    }

    // Move optimization
    // a. slice + permute -> permute + slice
    // b. reshape + slice -> slice + reshape
    // c. expand + slice -> slice + expand
    std::vector<BaseViewOp::baseViewOpPtr>
    runViewNodeMoveOpt(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      // First to fused same permute in view chain. this same node maybe from original
      // view chain, or cause by other op movement optimization.
      auto indexOfContinueNodes = indexOfSpecificContinueViewNode(other, ViewsType::kSlice);
      int startIndex = indexOfContinueNodes.size() - 1;
      while (startIndex >= 0) {
        const auto& pairNode = indexOfContinueNodes[startIndex];
        BaseViewOp::baseViewOpPtr sliceOpPtr = getFusedSlice({other.begin() + pairNode.first,
                                                            other.begin() + pairNode.second});
        other[pairNode.first] = sliceOpPtr;
        other.erase(other.begin() + pairNode.first + 1, other.begin() + pairNode.second);
        --startIndex;
      }
      // movement optimization
      std::vector<int> index = indexOfSpecificViewNode(other, ViewsType::kSlice);
      // skip index is empty or index value equal to 0.
      if (index.size() == 0 || (index.size() == 1 && index[0] == 0)) return other;
      // First to fused same permute in view chain. this same node maybe from original
      // view chain, or cause by other op movement optimization.
      return circleIndexWithContainerOptimization(other, index);
    }

  private:
    // Fused Permute view node op.
    // slice op1: (dim1, start1, end1, step1)
    // slice op2: (dim2, start2, end2, step2)
    // If dim1 is not equal dim2, just collecting slice op2 to dim/start/end/step vector,
    // otherwise fellow this formule:
    // new_start = start1 + start2 * step1
    // new_end = start1 + end2 * step1
    // new_step = step1 * step2
    BaseViewOp::baseViewOpPtr getFusedSlice(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) const {
      const SliceOp* firstNodeRawPtr = dynamic_cast<SliceOp*>(other.front().get());
      const int firstSliceNdim = firstNodeRawPtr->v_dims.size();
      // Prepare new slice op parameter.
      std::vector<int> newDims(firstNodeRawPtr->v_dims);
      std::vector<int> newStarts(firstNodeRawPtr->v_starts);
      std::vector<int> newEnds(firstNodeRawPtr->v_ends);
      std::vector<int> newSteps(firstNodeRawPtr->v_steps);
      for (int i = 1; i < other.size(); ++i) {
        const SliceOp* tmpRawPtr = dynamic_cast<SliceOp*>(other[i].get());
        const int tmpNdim = tmpRawPtr->v_dims.size();
        for (int j = 0; j < tmpNdim; ++j) {
          auto it = std::find(newDims.begin(), newDims.end(),
                              tmpRawPtr->v_dims[j]);
          if (it != newDims.end()) {
            int offset = std::distance(newDims.begin(), it);
            int tempEnd = newStarts[offset] + tmpRawPtr->v_ends[j] * newSteps[offset];
            int tempStart = newStarts[offset] + tmpRawPtr->v_starts[j] * newSteps[offset];
            newSteps[offset] *= tmpRawPtr->v_steps[j];
            // Adjust the value out of range.
            // new parameters vector is instantiation of first node, so offset must be same.
            newStarts[offset] = tempStart > newEnds[offset]
                                ? newEnds[offset]
                                : tempStart;
            newEnds[offset] = tempEnd > newEnds[offset]
                                ? newEnds[offset]
                                : tempEnd;
            continue;
          }
          newDims.push_back(tmpRawPtr->v_dims[j]);
          newStarts.push_back(tmpRawPtr->v_starts[j]);
          newEnds.push_back(tmpRawPtr->v_ends[j]);
          newSteps.push_back(tmpRawPtr->v_steps[j]);
        }
      }
      BaseViewOp::baseViewOpPtr slicePtr = std::make_shared<SliceOp>(newDims, newStarts,
                                                                   newEnds, newSteps);
      slicePtr->updateInputTensorInfo(firstNodeRawPtr->getInputTensorInfo());
      slicePtr->updateOutputTensorInfo(other.back()->getOutputTensorInfo());
      return slicePtr;
    }

    // Modify permute to reshape when adjacent dim permute, and one of those
    // size value is equal to 1.
    BaseViewOp::baseViewOpPtr inplaceSpecialSliceToReshape(
        const BaseViewOp::baseViewOpPtr& other) const {
      // (TODO)shangang: Support in future.
      return other;
    }

    std::vector<BaseViewOp::baseViewOpPtr>
    circleIndexWithContainerOptimization(std::vector<BaseViewOp::baseViewOpPtr>& other,
                                         const std::vector<int>& index) {
      int start_node = 0;
      for (int i = 0; i < index.size(); ++i) {
        int end_node = index[i];
        // No any other view node op before slice node.
        if ((end_node - start_node) == 0) {
          start_node += 1;
        } else {
          std::vector<BaseViewOp::baseViewOpPtr> subList {other.begin() + start_node,
                                                          other.begin() + end_node + 1};
          auto newSubList = moveSliceToTopOfChain(subList);
          // Can't be optimization, just move to next sublist.
          if (newSubList == subList) {
            start_node = end_node + 1;
            continue;
          }
          // update original node list.
          const int newSubNdims = newSubList.size();
          for (int j = 0; j < newSubNdims; ++j) {
            other[start_node + j] = newSubList[j];
          }
          // find next start node to optimization.
          int sliceIndex = 0;
          while (sliceIndex < newSubNdims) {
            if (newSubList[sliceIndex]->type() == ViewsType::kSlice) {
              ++sliceIndex;
              break;
            }
            ++sliceIndex;
          }
          start_node += sliceIndex;
        }
      }
      return other;
    }

    // permute + slice -> slice + permute movement optimization.
    // Only need to conside slice dims change.
    // For example:
    // permute op: (3, 4, 5, 6) --> (3, 5, 4, 6)
    // slice op:   (3, 5, 4, 6) --> (3, 3, 4, 6)
    // After movement optimization:
    // slice op:   (3, 4, 5, 6) --> (3, 4, 3, 6)
    // permute op: (3, 4, 3, 6) --> (3, 3, 4, 6)
    std::vector<BaseViewOp::baseViewOpPtr>
    exchangeSliceAndPermutePosition(const BaseViewOp::baseViewOpPtr& permuteNode,
                                    const BaseViewOp::baseViewOpPtr& sliceNode) {
      TORCH_MLU_CHECK(permuteNode->type() == ViewsType::kPermute, "First need be permute node.");
      TORCH_MLU_CHECK(sliceNode->type() == ViewsType::kSlice, "Second need be slice node.");
      const PermuteOp* permuteOpPtr = dynamic_cast<PermuteOp*>(permuteNode.get());
      const SliceOp* sliceOpPtr = dynamic_cast<SliceOp*>(sliceNode.get());
      const auto& permuteParameterDims = permuteOpPtr->getParameterDims();
      std::vector<int> newSliceDims(sliceOpPtr->v_dims);
      const int sliceDimsNdim = newSliceDims.size();
      for (int i = 0; i < sliceDimsNdim; ++i) {
        newSliceDims[i] = permuteParameterDims[newSliceDims[i]];
      }
      auto newSliceOp = std::make_shared<SliceOp>(newSliceDims, sliceOpPtr->v_starts,
                                                  sliceOpPtr->v_ends, sliceOpPtr->v_steps);
      // Can't Using original permute op ptr, because permute op ptr is a shared ptr, which
      // maybe used by other tensor.
      auto newPermuteOp = std::make_shared<PermuteOp>(permuteParameterDims);
      newSliceOp->updateInputTensorInfo(permuteOpPtr->getInputTensorInfo());
      newSliceOp->inferShape();
      newPermuteOp->updateInputTensorInfo(newSliceOp->getOutputTensorInfo());
      newPermuteOp->updateOutputTensorInfo(sliceOpPtr->getOutputTensorInfo());
      return {newSliceOp, newPermuteOp};
    }

    std::vector<BaseViewOp::baseViewOpPtr>
    exchangeSliceAndOtherOpPosition(const BaseViewOp::baseViewOpPtr& before,
                                    const BaseViewOp::baseViewOpPtr& after) {
      std::vector<BaseViewOp::baseViewOpPtr> result;
      switch (before->type()) {
        case ViewsType::kPermute:
          result = exchangeSliceAndPermutePosition(before, after);
          break;
        case ViewsType::kReshape:
        case ViewsType::kExpand:
        case ViewsType::kUnfold:
        case ViewsType::kDiagonal:
          result = {before, after};
          break;
        default:
          TORCH_MLU_CHECK(false, "View type is not suppport.");
          break;
      }
      return result;
    }

    std::vector<BaseViewOp::baseViewOpPtr>
    moveSliceToTopOfChain(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      // (TODO)shangang: More support in next version.
      const int nDims = other.size();
      if (nDims == 1) return other;
      TORCH_MLU_CHECK(other.back()->type() == ViewsType::kSlice, "Last node must be slice op.");
      std::vector<BaseViewOp::baseViewOpPtr> newViewNodeList(other);
      int slicePosition = nDims - 1;
      while (slicePosition > 0) {
        auto& perviousNode = newViewNodeList[slicePosition - 1];
        auto& sliceNode = newViewNodeList[slicePosition];
        auto tempSubChain = exchangeSliceAndOtherOpPosition(perviousNode, sliceNode);
        // Slice node with previous node can't be optimization, so break movement
        // optimization.
        if (tempSubChain[0] == perviousNode && tempSubChain[1] == sliceNode) break;
        // move new slice node to pervious position, and the perviousNode to old slice position.
        newViewNodeList[slicePosition - 1] = tempSubChain[0];
        newViewNodeList[slicePosition] = tempSubChain[1];
        --slicePosition;
      }
      return newViewNodeList;
    }
};

REGISTER_VIEW_NODE_OPT(ViewsType::kSlice, SliceFusionOpt);

}  // end of namespace torch_mlu
