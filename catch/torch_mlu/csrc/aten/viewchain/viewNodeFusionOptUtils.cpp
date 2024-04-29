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

#include <array>
#include "aten/viewchain/viewNodeFusionOptUtils.h"

namespace torch_mlu {

// Check all same view type.
bool isAllSameViewType(const std::vector<BaseViewOp::baseViewOpPtr>& other) {
  auto type = other.back()->type();
  constexpr std::array<ViewsType, 4> notSupportType {ViewsType::kUnfold,
                                                     ViewsType::kExpand,
                                                     ViewsType::kDiagonal};
  if (std::find(notSupportType.begin(), notSupportType.end(), type)
    != notSupportType.end()) return false;
  for (const auto& iter : other) {
    if (iter->type() != type) {
      return false;
    }
  }
  return true;
}

std::vector<int>
indexOfSpecificViewNode(const std::vector<BaseViewOp::baseViewOpPtr>& other,
                           ViewsType type) {
  const int nDim = other.size();
  std::vector<int> indexOfViewNode;
  for (int i = 0; i < nDim; ++i) {
    const auto& node = other[i];
    if (node->type() == type) {
      indexOfViewNode.emplace_back(i);
    }
  }
  return indexOfViewNode;
}

std::vector<std::pair<int, int>>
indexOfSpecificContinueViewNode(const std::vector<BaseViewOp::baseViewOpPtr>& other,
                                   ViewsType type) {
  const int nDim = other.size();
  int startIndex = 0;
  int endIndex = 1;
  std::vector<std::pair<int, int>> indexOfContinueNodes;
  while (endIndex < nDim && startIndex < endIndex) {
    if (other[startIndex]->type() != type) {
      ++startIndex;
      endIndex = startIndex + 1;
    } else {
      while (endIndex < nDim && other[endIndex]->type() == type) {
        ++endIndex;
      }
      if ((endIndex - startIndex) >= 2) {
        indexOfContinueNodes.push_back({startIndex, endIndex});
      }
      startIndex = endIndex;
      endIndex = startIndex + 1;
    }
  }
  return indexOfContinueNodes;
}

std::vector<int>
getSqueezeIndex(const std::vector<int64_t>& inputSizes,
                const std::vector<int64_t>& outputSizes) {
  const int inputDims = inputSizes.size();
  const int outputDims = outputSizes.size();
  // It's not squeeze behavior.
  if (inputDims <= outputDims) return {};
  int outputIndex = 0;
  std::vector<int> squeezeIndex;
  for (int i = 0; i < inputDims; ++i) {
    if (outputIndex > (outputDims - 1)) return {};
    if (inputSizes[i] != outputSizes[outputIndex]) {
      if (inputSizes[i] != 1) return {};
      squeezeIndex.push_back(i);
      continue;
    }
    ++outputIndex;
  }
  // left part of output sizes.
  if ((inputDims - squeezeIndex.size()) != outputDims) return {};
  return squeezeIndex;
}

}  // end of namespace torch_mlu

