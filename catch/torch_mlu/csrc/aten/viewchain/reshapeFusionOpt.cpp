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
#include "aten/utils/tensor_util.h"

/**
 * Note [ReshapeFusionOpt]
 * ~~~~~~~~~~~~~~~~
 * ReshapeFusionOpt is a series of reshape optimization methods.
 * 
 * Basic condition: 
 * We had skip unsafe view before add a reshape node to view chain.
 * Unsafe view means using copy instead shared storage.
 * 
 * example1:  fusion optimization
 * original view chain: reshape + reshape + reshape
 * after fusion: reshape
 * 
 * example2: move optimization
 * original view chain: permute + reshape + permute
 * after movement: reshape + permute + permute
 *              or permute + permute + reshape
 * 
 * For moving optimization, now only support two situation:
 * 1) Reshape behavior is like squeeze, so perfer forward movement optimization.
 *    Like: permute + reshape + permute to reshape + permute + permute.
 *    Because the performance of multi-dims permute may be lower than that of
 *    low-dims permute.
 * 2) Otherwise, we are trying using backward movement to optimization view chain.
 *    This movement we are trying to using locality of data dimension.
 *    Like: permute + reshape + permute
 *          Focus on the reshape and permute nodes.
 *          reshape op: input sizes: (3, 4, 2, 3, 5) -> output sizes: (3, 4, 6, 5)
 *          permute op: input sizes: (3, 4, 6, 5) -> output sizes: (4, 3, 6, 5)
 *          We can find data of dimension (2, 3, 5 or 6, 5) is not changed. So based
 *          on locality of data dimension, we can using movement optimization to modify
 *          nodes.
 *          permute op: nput sizes: (3, 4, 2, 3, 5) -> output sizes: (4, 3, 2, 3, 5)
 *          reshape op: input sizes: (4, 3, 2, 3, 5) -> output sizes: (4, 3, 6, 5)
 *          then view nodes like: permute + permute + reshape. Those two permute node
 *          can be fused.
 * 
 */

namespace torch_mlu {

class ReshapeFusionOpt: public ViewNodeFusionOpt {
  public:
    // Fusion optimization. Without check in here, so you need check all
    // same type when call this function.
    BaseViewOp::baseViewOpPtr
    runViewNodeFusionOpt(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) override {
      return getFusedReshape(other);
    }

    // Replace optimization. Reshape op is no need to replace.
    BaseViewOp::baseViewOpPtr
    runViewNodeReplaceOpt(const BaseViewOp::baseViewOpPtr& ptr) override {
      return ptr;
    }

    // Move optimization
    // Move reshape out of two same node, and fusion optimization will fused those
    // same node. Only support three view op move with reshape just now, there are permute,
    // slice and expand.
    // For permute and reshape move optimization:
    // 1)
    std::vector<BaseViewOp::baseViewOpPtr>
    runViewNodeMoveOpt(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      // First to fused same permute in view chain. this same node maybe from original
      // view chain, or cause by other op movement optimization.
      auto indexOfContinueNodes = indexOfSpecificContinueViewNode(other, ViewsType::kReshape);
      int startIndex = indexOfContinueNodes.size() - 1;
      while (startIndex >= 0) {
        const auto& pairNode = indexOfContinueNodes[startIndex];
        BaseViewOp::baseViewOpPtr reshapeOpPtr = getFusedReshape({other.begin() + pairNode.first,
                                                                  other.begin() + pairNode.second});
        other[pairNode.first] = reshapeOpPtr;
        other.erase(other.begin() + pairNode.first + 1, other.begin() + pairNode.second);
        --startIndex;
      }
      return moveReshapeOutOfSameNodes(other);
    }

  private:
    // Fused Permute view node op.
    // So here we just fused reshape with using first node input tensor info
    // and second node output tensor info to creat a new reshape node.
    BaseViewOp::baseViewOpPtr getFusedReshape(
        const std::vector<BaseViewOp::baseViewOpPtr>& other) const {
      const ReshapeOp* firstNodeRawPtr = dynamic_cast<ReshapeOp*>(other.front().get());
      const ReshapeOp* lastNodeRawPtr = dynamic_cast<ReshapeOp*>(other.back().get());
      BaseViewOp::baseViewOpPtr reshapePtr =
        std::make_shared<ReshapeOp>(lastNodeRawPtr->getOutputTensorInfo().v_sizes);
      reshapePtr->updateInputTensorInfo(firstNodeRawPtr->getInputTensorInfo());
      reshapePtr->updateOutputTensorInfo(lastNodeRawPtr->getOutputTensorInfo());
      return reshapePtr;
    }

    // Reshape view node backward movement is the preferred movement
    // direction.
    // a. Reshape squeeze tensor dims, try to using reshape + permute to instead
    // permute + reshape
    // b. Others try to move reshape node backward;
    std::vector<BaseViewOp::baseViewOpPtr>
    moveReshapeFromPermutes(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      TORCH_MLU_CHECK(other.size() == 3, "Only support three nodes right now.");
      const ReshapeOp* const reshapeOpPtr = dynamic_cast<ReshapeOp*>(other[1].get());
      // Compare reshape input and output tensor dims size.
      const auto& reshapeInputSizes = reshapeOpPtr->getInputTensorInfo().v_sizes;
      const auto& reshapeOutputSizes = reshapeOpPtr->getOutputTensorInfo().v_sizes;
      // Currently, only the movement with reshape as squeeze scenario is supported.
      // For example:
      // permute + reshape + permute --> reshape + permute + permute
      // permute: (3, 2, 1, 4, 5) -> (2, 1, 3, 4, 5)
      // reshape is from (2, 1, 3, 4, 5) -> (2, 3 ,4 ,5)
      // permute: (2, 3 ,4 ,5) -> (2, 4 ,3 ,5)
      // after reshape movement.
      // reshape is from (3, 2, 1, 4, 5) -> (3, 2, 4, 5)
      // permute: (3, 2, 4, 5) -> (2, 3 ,4 ,5)
      // permute: (2, 3 ,4 ,5) -> (2, 4 ,3 ,5)
      std::vector<int> squeezeIndex = getSqueezeIndex(reshapeInputSizes, reshapeOutputSizes);
      if (squeezeIndex.size() != 0) {
        // a. Conver squeeze Index to permute input sizes index
        const PermuteOp* permuteOpPtr = dynamic_cast<PermuteOp*>(other[0].get());
        const auto& permuteDims = permuteOpPtr->getParameterDims();
        for (auto& index : squeezeIndex) {
          index = permuteDims[index];
        }
        // b. Get new permute input sizes with orignal first permute node info and squeezeIndex.
        const auto& permuteInputSizes = permuteOpPtr->getInputTensorInfo().v_sizes;
        const int permuteInputNdims = permuteInputSizes.size();
        std::vector<int64_t> newPermuteSizes;
        std::vector<int64_t> newPermuteDims;
        const int newPermuteNdims = permuteInputNdims - squeezeIndex.size();
        newPermuteSizes.reserve(newPermuteNdims);
        newPermuteDims.reserve(newPermuteNdims);
        for (int i = 0; i < permuteInputNdims; ++i) {
          auto it = std::find(squeezeIndex.begin(), squeezeIndex.end(), i);
          if (it == squeezeIndex.end()) {
            newPermuteSizes.push_back(permuteInputSizes[i]);
          }
          it = std::find(squeezeIndex.begin(), squeezeIndex.end(), permuteDims[i]);
          if (it == squeezeIndex.end()) {
            newPermuteDims.push_back(permuteDims[i]);
          }
        }

        // c. Try to get new permute info by permute parameters and squeezeIndex.
        for (auto& value : newPermuteDims) {
          for (const auto& index : squeezeIndex) {
            if (value >= index) {
              value -= 1;
            }
          }
        }
        // d. Generate new reshape and permute op. op list like: reshape + permute + permute
        BaseViewOp::baseViewOpPtr newReshapeOpPtr = std::make_shared<ReshapeOp>(newPermuteSizes);
        newReshapeOpPtr->updateInputTensorInfo(permuteOpPtr->getInputTensorInfo());
        // create new internal tensor info, is new reshape output tensor info and new permute
        // input tensor info/
        TensorInfo outputTensorInfo(newPermuteSizes,
                                    torch_mlu::get_contiguous_strides(newPermuteSizes),
                                    permuteOpPtr->getInputTensorInfo().i_storageOffset);
        newReshapeOpPtr->updateOutputTensorInfo(outputTensorInfo);
        BaseViewOp::baseViewOpPtr newPermuteOpPtr = std::make_shared<PermuteOp>(newPermuteDims);
        newPermuteOpPtr->updateInputTensorInfo(outputTensorInfo);
        newPermuteOpPtr->updateOutputTensorInfo(reshapeOpPtr->getOutputTensorInfo());
        // e. update original segement.
        other[0] = newReshapeOpPtr;
        other[1] = newPermuteOpPtr;
      } else {
        // (TODO)shangang: Backward movement optimization will be added in future.
      }
      return other;
    }

    // This function only support handle view node list like: IO view node
    // + serveral reshapes + IO view node.
    std::vector<BaseViewOp::baseViewOpPtr>
    moveReshapeFromMiddleToSide(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      // Check the view node list.
      checkViewNodeFormat(other);
      std::vector<BaseViewOp::baseViewOpPtr> newViewNodeList;
      constexpr int baseReshapeMoveUnit = 3;
      newViewNodeList.reserve(baseReshapeMoveUnit);
      // Fused internal reshape in view node list.
      {
        newViewNodeList.emplace_back(other.front());
        BaseViewOp::baseViewOpPtr reshapeOpPtr = *(other.begin() + 1);
        const int nDims = other.size();
        if (nDims > 3) {
          // Get temp view node list without first and last view node.
          // Call vector iterator constructor, this mean [left, right).
          reshapeOpPtr = runViewNodeFusionOpt({other.begin() + 1,
                                               other.begin() + nDims - 1});
        }
        newViewNodeList.emplace_back(reshapeOpPtr);
        newViewNodeList.emplace_back(other.back());
      }
      // (TODO)shangang: Here only support permute + reshape + permute optimization,
      // unfold + reshape + unfold maybe support in future. And we don't need to consider
      // slice and expand here. Because slice and expand optimization have completed the
      // processing of these two operators. Slice Optimization already move slice to the
      // top of chain, and expand Optimization already move slice to the below of chain.
      if (other.front()->type() == ViewsType::kPermute) {
        newViewNodeList = moveReshapeFromPermutes(newViewNodeList);
      }
      return newViewNodeList;
    }

    inline void checkViewNodeFormat(const std::vector<BaseViewOp::baseViewOpPtr>& other) {
      const int nDim = other.size();
      auto firstNodeType = other.front()->type();
      auto lastNodeType = other.back()->type();
      TORCH_MLU_CHECK(firstNodeType == lastNodeType,
        "First and last node type need be same.");
      TORCH_MLU_CHECK(firstNodeType != ViewsType::kReshape,
        "First and last node type are same with view type reshape.");
      for (int i = 1; i < nDim - 1; ++i) {
        TORCH_MLU_CHECK(other[i]->type() == ViewsType::kReshape,
          "Node type need be view type reshape.");
      }
    }

    // This function is to find a sub list to optimization. And sub list is like:
    // permute + reshape + reshape + permute.
    // Slice a sub list of view nodes each time, and using start index and end index
    // to control circle.
    // a: Sub list is start from a IO view node and end with a same IO view node, if each
    // other IO view node appear, then move to start index and end index;
    // b: If no any reshape between two same IO view node, the move start index and end index;
    // c: After a Sub list is optimized and updated to origin view chain, The program will
    // search from the same start index to see if any new node sequence can be optimized.
    std::vector<BaseViewOp::baseViewOpPtr>
    moveReshapeOutOfSameNodes(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      std::vector<int> reshape_index = indexOfSpecificViewNode(other, ViewsType::kReshape);
      if (reshape_index.size() == 0) return other;
      // Slice sub list of view chain.
      int startIndex = 0;
      int endIndex = 1;
      int maxNode = other.size();
      while (endIndex < maxNode && startIndex < endIndex) {
        auto startType = other[startIndex]->type();
        if (startType == ViewsType::kReshape) {
          ++startIndex;
          endIndex = startIndex + 1;
        } else {
          // gready to get continue reshape node.
          while (endIndex < maxNode && other[endIndex]->type() == ViewsType::kReshape) {
            ++endIndex;
          }
          // reshape in the end of chain.
          if (endIndex > (maxNode - 1)) break;
          if (startType != other[endIndex]->type() || (endIndex - startIndex) < 2) {
            startIndex = endIndex;
            endIndex += 1;
          } else {
            // This mean start type is equal to end type, and interval node num is
            // greater or equal than 2 between start index and end index.
            std::vector<BaseViewOp::baseViewOpPtr> subList(other.begin() + startIndex,
                                                         other.begin() + endIndex + 1);
            auto newSubList = moveReshapeFromMiddleToSide(subList);
            if (newSubList != subList) {
              // The newSubList maybe shorter than subList. So assign new view nodes and erase
              // useless nodes from original view node list.
              // a: assign the newSubList to the first of original view node list.
              const int newSubListSize = newSubList.size();
              for (int i = 0; i < newSubListSize; ++i) {
                other[startIndex + i] = newSubList[i];
              }
              // b: erase the left of subList in the original view node list.
              other.erase(other.begin() + startIndex + newSubListSize,
                          other.begin() + endIndex + 1);
            }
            // update start_index and end_index.
            startIndex += 1;
            endIndex = startIndex + 1;
            // update chain length.
            maxNode = other.size();
          }
        }
      }
      return other;
    }
};

REGISTER_VIEW_NODE_OPT(ViewsType::kReshape, ReshapeFusionOpt);

}  // end of namespace torch_mlu
