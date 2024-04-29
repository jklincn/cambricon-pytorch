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

#include <memory>
#include <mutex>
#include <unordered_map>
#include "aten/viewchain/baseViewOps.h"
#include "aten/viewchain/viewNodeFusionOptUtils.h"

/**
 * Note [ViewNodeFusionOpt]
 * ~~~~~~~~~~~~~~~~
 * ViewNodeFusionOpt is base class of view node fused. So far, only mutual fusion or
 * position optimization between two nodes is handled.
 * 
 * This fusion strategy adopts the method of forward fusion, and the matching fusion
 * is carried out according to the last node in the node sequence. There may be two
 * nodes, three nodes or more nodes in the node sequence.
 * 
 * example1: 
 * original view chain: permute --> permute
 * after mutual fusion: permute
 * Using permute fusion opt method
 * 
 * example2: 
 * original view chain: permute --> slice
 * after position optimization: slice --> permute
 * Using slice fusion opt method
 * 
 */

namespace torch_mlu {

class ViewNodeFusionOpt {
  public:
    ViewNodeFusionOpt() = default;

    // AT_DISALLOW_COPY_AND_ASSIGN(ViewNodeFusionOpt);

    // For Fusion optimization
    virtual BaseViewOp::baseViewOpPtr
    runViewNodeFusionOpt(const std::vector<BaseViewOp::baseViewOpPtr>& other) {
      TORCH_MLU_CHECK(false, "Base view op fusion is not support this operator.");
    }

    // For replace optimization
    virtual BaseViewOp::baseViewOpPtr
    runViewNodeReplaceOpt(const BaseViewOp::baseViewOpPtr& ptr) {
      TORCH_MLU_CHECK(false, "Base view op fusion is not support this operator.");
    }

    // For move optimization
    virtual std::vector<BaseViewOp::baseViewOpPtr>
    runViewNodeMoveOpt(std::vector<BaseViewOp::baseViewOpPtr>& other) {
      TORCH_MLU_CHECK(false, "Base view op fusion is not support this operator.");
    }
    virtual ~ViewNodeFusionOpt() = default;
};

/**
 * Note [ViewNodeOptRegister]
 * ~~~~~~~~~~~~~~~~
 * ViewNodeOptRegister is class to register each view node optimization object,
 * which using unordered_map to store. Key is view type, value is unique point of
 * optimization object.
 */

class ViewNodeOptRegister {
  private:
    std::mutex m_mutex;
    std::unordered_map<ViewsType, std::unique_ptr<ViewNodeFusionOpt>> m_register;
    ViewNodeOptRegister() = default;

  public:
    // Copy constructor and Operator= is disabled.
    AT_DISALLOW_COPY_AND_ASSIGN(ViewNodeOptRegister);

    static ViewNodeOptRegister& getInstance();

    // To register a new view node opt class in m_register.
    void viewNodeRegister(ViewsType type, std::unique_ptr<ViewNodeFusionOpt> ptr) {
      std::lock_guard<std::mutex> lck(m_mutex);
      TORCH_MLU_CHECK(m_register.count(type) == 0, "Type already registered.");
      m_register[type] = std::move(ptr);
    }

    // Get view node optimation raw ptr. If ViewsType is not registered,
    // nullptr will return.
    inline ViewNodeFusionOpt* getSpecificRawPtr(ViewsType type) const {
      auto it = m_register.find(type);
      if (it != m_register.end()) {
        return it->second.get();
      }
      return nullptr;
    }

    // Call specific view node opt function.
    BaseViewOp::baseViewOpPtr
    runViewNodeFusionOpt(const std::vector<BaseViewOp::baseViewOpPtr>& other);

    BaseViewOp::baseViewOpPtr
    runViewNodeReplaceOpt(const BaseViewOp::baseViewOpPtr& ptr);

    std::vector<BaseViewOp::baseViewOpPtr>
    runViewNodeMoveOpt(std::vector<BaseViewOp::baseViewOpPtr>& other,
                       ViewsType type);

    // Deconstructor
    ~ViewNodeOptRegister() = default;
};

/**
 * Note [ViewNodeOptRegisterer]
 * ~~~~~~~~~~~~~~~~
 * ViewNodeOptRegisterer is a registerer. Constructor is used to register a 
 * specific view node opt class to ViewNodeOptRegister.
 * 
 */

template<typename X>
class ViewNodeOptRegisterer {
  public:
    /* For SFINAE to reach in and apply to the method level, those methods
    must be templates themselves.
    More detail:
    https://stackoverflow.com/questions/11531989/what-happened-to-my-sfinae-redux-conditional-template-class-members */
    template<typename T = X, typename std::enable_if<
      std::is_convertible<T, ViewNodeFusionOpt>::value,
      int>::type _dummy = 1>
    ViewNodeOptRegisterer(ViewsType type, std::unique_ptr<T> ptr) {
      ViewNodeOptRegister::getInstance().viewNodeRegister(type, std::move(ptr));
    }

    template<typename T = X, typename std::enable_if<
      std::is_convertible<T, ViewNodeFusionOpt>::value,
      int>::type _dummy = 1>
    explicit ViewNodeOptRegisterer(ViewsType type) {
      std::unique_ptr<ViewNodeFusionOpt> ptr = std::make_unique<T>();
      ViewNodeOptRegister::getInstance().viewNodeRegister(type, std::move(ptr));
    }

    ~ViewNodeOptRegisterer() = default;
};

#define ViewChainNodeManager ViewNodeOptRegister::getInstance()

#define REGISTER_VIEW_NODE_OPT(type, classname) \
  REGISTER_VIEW_NODE_OPT_UNIQ(classname, type, classname)
#define REGISTER_VIEW_NODE_OPT_UNIQ(name, type, classname) \
  static ViewNodeOptRegisterer<classname> name##_register(type)

}  // end of namespace torch_mlu

