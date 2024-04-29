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

#include "aten/viewchain/viewNodeFuser.h"

/**
 * Note [viewChainFuser]
 * ~~~~~~~~~~~~~~~~
 * viewChainFuser is aimed to fuse a large number of view nodes into a small
 * number of view nodes. Which will save a lot of running time of IO operators
 * on the device.
 * 
 * So there are two principles:
 * 1) If IO kernel is reduce IO data volume, we need to move this IO kernel forward in
 * view chain. Otherwise we need to move this IO kernel backward in view chain. 
 * 2) If same IO kernels in adjacent positions, we need to fused them to one IO kernel.
 * 
 * Operation stages:
 * Stage one:
 * This is Default optimization, which call fusion optimization, replace optimization and
 * movement optimization.
 * a. By viewing the whole view chain, the visible contiguous view node are fused to one node.
 *    This will simplify the movement between multiple nodes;
 * b. By circle the whole view chain to check each node, and replace a IO view node to reshape
 *    view node according to certain rules;
 * c. Accoding to the type of each node, we are trying to move view node to different position
 *    in the view chain.
 *    1) Try to move slice to the top of view chain, which include some movement optimizations to
 *       exchange positions.;
 *       Like: permute + slice -> slice + permute;
 *             expand + slice -> slice + expand;
 *             reshape + slice -> slice + reshape.
 *    2) Try to move expand to the bottom of view chain;
 *       Like: expand + permute -> permute + expand;
 *             expand + reshape -> reshape + expand.
 *    3) Try to move reshape output of two permute node.
 *       Like: permute + reshape + permute -> reshape + permute + permute
 *          or permute + reshape + permute -> permute + permute + reshape
 * 
 * Stage two:
 * Each time we push two view nodes to obtain a better combination. Assuming two same view op
 * is met, fused those to one view node. Otherwise nothing will change.
 * Like: permute + permute -> permute
 *       slice + slice -> slice and so no.
 *
 * Stage three:
 * Each time we push one view node to obtain replace one, which is using shared storage to
 * reduce IO operator.
 * Like: permute -> reshape
 * 
 * Limited:
 * unfold and diagonal view node is not supported now.
 * 
 * example: 
 * original view chain: permute --> slice --> permute
 * after fused view chain: slice --> permute
 * 
 * original view chain: permute --> slice --> expand --> permute
 * after fused view chain: slice --> permute --> expand
 * 
 * original view chain: permute --> reshape --> reshape --> permute
 * after fused view chain: permute
 * 
 * Why we need do this?
 * IO operators running time is decided by IO data volume, so we need to reduce IO
 * data volume by moving IO operator position or fused IO same(or almost same) operator.
 * 
 * More view node details you can find in torch_mlu/csrc/aten/cnnl/baseViewOps.h
 * More view chain details you can find in torch_mlu/csrc/aten/cnnl/viewChain.h
 * 
 */

namespace torch_mlu {

// This struct is using for private function runFusedWithSpecificNum in viewChainFuser.
// To solve compiled error(Explicit specialization in non-namespace scope), which will solve
// in c++17. More details can check:
// https://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope
// num = 0 is for default template.
template<typename T, T num = 0, typename
std::enable_if<std::is_same<T, int>::value, int>::type = 1>
struct identity {
  static constexpr T value = num;
  typedef T type;
};

// Singleton class
class viewChainFuser {
  private:
    // Based on env variable DISABLE_VIEWCHAIN_FUSED_FUNC.
    bool b_closeChainFusedFunc;

  public:
    typedef c10::SmallVector<BaseViewOp::baseViewOpPtr, MAX_STORE_INFO_NUM>
      baseViewOpPtrContainer;

    AT_DISALLOW_COPY_AND_ASSIGN(viewChainFuser);

    // Get instance of viewChainFuser
    static viewChainFuser& getInstance();

    // Only check view node num, fused environment and infer shape of view nodes.
    bool canFusedViewChain(const baseViewOpPtrContainer& other) const;

    // Cycle the whole container and optimize the fusion of corresponding node
    // segments.
    // There will be three stages in this function. You can see more details in
    // Note [viewChainFuser].
    // Finally, if you want to use different num of node segments to satisfy different
    // optimization strategies, add more partial specialization function of
    // runFusedWithSpecificNum. and call those partial specialization function in this
    // function based on different condition.
    baseViewOpPtrContainer runFusedViewChain(const baseViewOpPtrContainer& other);

  private:
    viewChainFuser();

    // Using template function to support more optimization strategies.
    // You can using num value to support more node optimization, like 3 or 4
    // in future.
    /* 
    This template function compile failed. More details write before in struct identity.
    template<int num>
    baseViewOpPtrContainer
    runFusedWithSpecificNum(const baseViewOpPtrContainer& other) {
      TORCH_MLU_CHECK(false, "Only support two nodes optimization right now.");
    } */
    template<typename T, T num>
    baseViewOpPtrContainer
    runFusedWithSpecificNum(const baseViewOpPtrContainer& other, identity<T, num>);

    // Only support two nodes optimization right now.
    baseViewOpPtrContainer runFusedWithSpecificNum(const baseViewOpPtrContainer& other,
                                                 identity<int, 2>);

    // Only support one nodes optimization right now.
    baseViewOpPtrContainer runFusedWithSpecificNum(const baseViewOpPtrContainer& other,
                                                 identity<int, 1>);
};

#define viewChainFuserManager viewChainFuser::getInstance()

}  // end of namespace torch_mlu

