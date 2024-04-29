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

#include "aten/viewchain/viewChainFuser.h"

namespace torch_mlu {

viewChainFuser::viewChainFuser() {
  // lambda func is to get environment variable DISABLE_VIEWCHAIN_FUSED_FUNC,
  // which is to control whether to use view chain fuser.
  static const bool b_enabled = ([](){
    bool enabled = false;
    const char* e = std::getenv("DISABLE_VIEWCHAIN_FUSED_FUNC");
    if (e != nullptr && (strcmp(e, "on") == 0 ||
        strcmp(e, "ON") == 0 || strcmp(e, "1") == 0)) {
      enabled = true;
    }
    return enabled;
  })();
  b_closeChainFusedFunc = b_enabled;
}

viewChainFuser& viewChainFuser::getInstance() {
  static viewChainFuser instance;
  return instance;
}

// Check environment variable, chain size.
bool viewChainFuser::canFusedViewChain(const baseViewOpPtrContainer& other) const {
  // Fusion optimization requires at least two nodes.
  if (b_closeChainFusedFunc || other.size() < 2) {
    return false;
  }
  return true;
}

// More optimization strategies can be added in future.
typename viewChainFuser::baseViewOpPtrContainer
viewChainFuser::runFusedViewChain(const baseViewOpPtrContainer& other) {
  // a. Basic optimization method. According to the operator type, fusion optimization,
  // replacement optimization and movement optimization will be called in order to
  // minimize IO data volume.
  baseViewOpPtrContainer new_chain = runFusedWithSpecificNum(other, identity<int>());
  // Even we already call fusion func or instead func in movement optimization,
  // we still call fusion func and instead func here. Because movement optimization
  // creates new view chain, which may include some sublist with fusion or replace
  // demand.
  // b. fusion nodes.
  new_chain = runFusedWithSpecificNum(new_chain, identity<int, 2>());
  // c. using reshape to replace some special IO data.
  return runFusedWithSpecificNum(new_chain, identity<int, 1>());
}

// Default template is for moving op, which fellow three principles.
// Those move optimizations are trying to get same type op together.
// a: fusion consecutive slice op, and move slice op to the top of chain;
// b: fusion consecutive expand op, and move expand op to the bottom of chain;
// c: fusion consecutive permute op, and using reshape to replace some special permute;
// d: move reshape out of two permute op. For example: permute + reshape + permute ->
//    reshape + permute + permute or permute + permute + reshape.
template<typename T, T num>
viewChainFuser::baseViewOpPtrContainer
viewChainFuser::runFusedWithSpecificNum(const baseViewOpPtrContainer& other, identity<T, num>) {
  std::vector<BaseViewOp::baseViewOpPtr> newContainer(other.begin(), other.end());
  // reshape need be last one of supportMoveOptViewType.
  // (TODO) shangang: expand op optimization will add in future.
  static const std::vector<ViewsType> supportMoveOptViewType = {ViewsType::kPermute,
                                                                ViewsType::kSlice,
                                                                ViewsType::kReshape};
  for (auto& type : supportMoveOptViewType) {
    newContainer = ViewChainNodeManager.runViewNodeMoveOpt(newContainer, type);
  }
  return baseViewOpPtrContainer(newContainer);
}

// Push one view node into viewNodeFusionOpt each time, and get a new instead op.
// This is using some reshape op to instead IO kernel in some special cases.
viewChainFuser::baseViewOpPtrContainer
viewChainFuser::runFusedWithSpecificNum(const baseViewOpPtrContainer& other, identity<int, 1>) {
  baseViewOpPtrContainer new_chain;
  for (auto& node : other) {
    auto result = ViewChainNodeManager.runViewNodeReplaceOpt(node);
    new_chain.push_back(result);
  }
  return new_chain;
}

// Push two view nodes into viewNodeFusionOpt each time, and get a fusion node back.
viewChainFuser::baseViewOpPtrContainer
viewChainFuser::runFusedWithSpecificNum(const baseViewOpPtrContainer& other, identity<int, 2>) {
  // Call optimization function based on fragment
  baseViewOpPtrContainer new_chain;
  new_chain.push_back(other.front());
  for (int i = 1; i < other.size(); ++i) {
    auto view_ptr = other[i];
    std::vector<BaseViewOp::baseViewOpPtr> segment{new_chain.back(), view_ptr};
    if (isAllSameViewType(segment)) {
      view_ptr = ViewChainNodeManager.runViewNodeFusionOpt(segment);
      new_chain.back() = view_ptr;
      continue;
    }
    new_chain.push_back(view_ptr);
  }
  return new_chain;
}

}  // end of namespace torch_mlu
