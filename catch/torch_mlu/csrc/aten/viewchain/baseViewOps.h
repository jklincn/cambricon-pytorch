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
#include <vector>
#include <string>
#include "ATen/Tensor.h"
#include "ATen/Utils.h"
#include "aten/utils/exceptions.h"

/**
 * Note [ViewsType]
 * ~~~~~~~~~~~~~~~~
 * ViewsType is to mark a specific IO kernel. Now we just support
 * slice, permute, expand, reshape, and unfold. 
 * Also pytorch acoustic view op has a mapping relationship with
 * ViewsType.
 * | pytorch acoustic view op |   ViewsType   |
 * |   expand / expand_as     |    kExpand    |
 * |   narrow / select        |    kSlice     |
 * |   unbind / chunk         |    kSlice     |
 * |   split / hsplit         |    kSlice     |
 * |   vsplit / tensor_split  |    kSlice     |
 * |   split_with_sizes       |    kSlice     |
 * |   permute / transpose    |    kPermute   |
 * |   transpose_ / swapaxes  |    kPermute   |
 * |   t() / T / movedim      |    kPermute   |
 * |   swapdims               |    kPermute   |
 * |   unsqueeze / view       |    kReshape   |
 * |   squeeze / view_as      |    kReshape   |
 * |   reshape / reshape_as	  |    kReshape   |
 * |   squeeze_ / unsqueeze_  |    kReshape   |
 * |   unflatten              |    kReshape   |
 * |   unflod                 |    kUnfold    |
 * |   diagonal               |    kDiagonal  |
 * 
 * More details about pytorch acoustic view op:
 * https://pytorch.org/docs/1.13/tensor_view.html?highlight=view
 */

/**
 * Note [viewOps]
 * ~~~~~~~~~~~~~~~~
 * viewOps is composed of a series of data structures, which
 * are mainly used to record the information required by a 
 * specific IO operator.
 * BaseViewOp is parent class, all other ViewOp class are
 * inherited from BaseViewOp. Which does not support instantiation.
 * Subclass methods are called through shared_ptr of BaseViewOp in
 * view chain class.
 * PermuteOp / SliceOp / ExpandOp / ReshapeOp / UnfoldOp /
 * DiagonalOp are children class of BaseViewOp. Which are
 * used to store the information required by the specific IO kernel.
 * 
 */

namespace torch_mlu {

// Using in small vector to store view node or tensor info.
constexpr int MAX_STORE_INFO_NUM = 6;

// view type is to mark a specific IO kernel.
// kInvalid must be the last one in ViewsType.
enum class ViewsType {
  kSlice,
  kPermute,
  kExpand,
  kReshape,
  kUnfold,
  kDiagonal,
  kInvalid,
};

// If you add a new view type in ENUM class ViewsType,
// you need add corresponding description in this macro.
#define ALL_VIEWTYPE_TO_STRING(_)     \
  _(ViewsType::kSlice, "slice")       \
  _(ViewsType::kPermute, "permute")   \
  _(ViewsType::kExpand, "expand")     \
  _(ViewsType::kReshape, "reshape")   \
  _(ViewsType::kUnfold, "unfold")     \
  _(ViewsType::kDiagonal, "diagonal")

// Convert view type to string for debugging.
static std::string viewTypeToString(const ViewsType& type) {
  std::string name;
  switch (type) {
  #define DEFINE_CASE(type, type_name) \
    case type:                         \
      name = type_name;                \
      break;
    ALL_VIEWTYPE_TO_STRING(DEFINE_CASE)
  #undef DEFINE_CASE
    default:
      LOG(FATAL) << "Failed find valid type in support"
        << " ViewsType, and ViewsType is: "
        << std::to_string(static_cast<int>(type));
      break;
  }
  return name;
}

// This struct is used for store tensor size, stride and storage_offset info.
class TensorInfo {
  public:
    TensorInfo() = default;

    explicit TensorInfo(const std::vector<int64_t>& sizes,
                        const std::vector<int64_t>& strides,
                        const int64_t& storage_offset)
                          : v_sizes(sizes),
                          v_strides(strides),
                          i_storageOffset(storage_offset) { }

    explicit TensorInfo(const at::Tensor& self)
                          : v_sizes(self.sizes().begin(), self.sizes().end()),
                          v_strides(self.strides().begin(), self.strides().end()),
                          i_storageOffset(self.storage_offset()) { }

    // Support move construct and move operator.
    TensorInfo(const TensorInfo& other): v_sizes(other.v_sizes),
                                         v_strides(other.v_strides),
                                         i_storageOffset(other.i_storageOffset) { }

    // Support move construct and move operator.
    TensorInfo(TensorInfo&& other): v_sizes(std::move(other.v_sizes)),
                                    v_strides(std::move(other.v_strides)),
                                    i_storageOffset(std::move(other.i_storageOffset)) { }

    TensorInfo& operator=(const TensorInfo& other) {
      this->v_sizes = other.v_sizes;
      this->v_strides = other.v_strides;
      this->i_storageOffset = other.i_storageOffset;
      return *this;
    }

    TensorInfo& operator=(TensorInfo&& other) {
      this->v_sizes = std::move(other.v_sizes);
      this->v_strides = std::move(other.v_strides);
      this->i_storageOffset = std::move(other.i_storageOffset);
      return *this;
    }

    // Tensor memory format is decided by size and stride, so skip storage offset.
    inline bool operator==(const TensorInfo& other) const {
      if (this->v_sizes == other.v_sizes &&
        this->v_strides == other.v_strides &&
        this->i_storageOffset == other.i_storageOffset) {
        return true;
      }
      return false;
    }

    inline bool operator!=(const TensorInfo& other) const {
      return !(*this == other);
    }

    // Visual parameters of view op for debugging.
    friend std::ostream& operator<<(std::ostream& out, const TensorInfo& other);

  public:
    std::vector<int64_t> v_sizes;
    std::vector<int64_t> v_strides;
    int64_t i_storageOffset = 0;
};

// Base container of view ops.
class BaseViewOp {
  public:
    typedef std::shared_ptr<BaseViewOp> baseViewOpPtr;

    explicit BaseViewOp(ViewsType type): e_type(type) { }

    // Update intput and output tensor size,stride,storage_offset
    // info in view node.
    inline void updateTensorInfo(const at::Tensor& input,
                                 const at::Tensor& output) {
      this->t_inputTensorInfo = TensorInfo(input);
      this->t_outputTensorInfo = TensorInfo(output);
    }

    // Using for inplace op to store input and output tensor
    // size,stride,storage_offset info.
    inline void updateTensorInfo(TensorInfo&& input,
                                 TensorInfo&& output) {
      this->updateInputTensorInfo(std::move(input));
      this->updateOutputTensorInfo(std::move(output));
    }

    // Sometimes view chain fused will modify input and output tensor size info.
    // Like reshape + permute fused to permute. So need to modify TensorInfo stored
    // in view node.
    inline void updateInputTensorInfo(const TensorInfo& other) {
      this->t_inputTensorInfo = other;
    }

    inline void updateInputTensorInfo(TensorInfo&& other) {
      this->t_inputTensorInfo = std::move(other);
    }

    inline void updateOutputTensorInfo(const TensorInfo& other) {
      this->t_outputTensorInfo = other;
    }

    inline void updateOutputTensorInfo(TensorInfo&& other) {
      this->t_outputTensorInfo = std::move(other);
    }

    inline const TensorInfo& getInputTensorInfo() const {
      return t_inputTensorInfo;
    }

    inline const TensorInfo& getOutputTensorInfo() const {
      return t_outputTensorInfo;
    }

    // Get ViewsType
    inline ViewsType type() const {
      return e_type;
    }

    // Tensor memory format is decided by size, stride and storage offset.
    inline bool compareInputAndOutputTensorInfo() const {
      return this->t_inputTensorInfo == this->t_outputTensorInfo;
    }

    // Using for caculate output tensor info when view chain is fused.
    virtual bool inferShape() {
      TORCH_MLU_CHECK(false, "Base view op is not support this operator.");
    }

    // Judge whether there is CNNL specific IO kernel according
    // to the parameter information. If return false, call cnnlCopy
    // instead.
    virtual bool hasCnnlSpecificFunction() {
      TORCH_MLU_CHECK(false, "Base view op is not support this operator.");
    }

    // Call CNNL specific IO operator.
    virtual at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                               c10::optional<at::Tensor> output) {
      TORCH_MLU_CHECK(false, "Base view op is not support this operator.");
    }
    // Convert view Op parameter info to string.
    virtual std::string parameterToString() const {
      TORCH_MLU_CHECK(false, "Base view op is not support this operator.");
    }

    virtual bool isEqual(const BaseViewOp& other) {
      TORCH_MLU_CHECK(false, "Base view op is not support this operator.");
    }

    // Deconstructor
    virtual ~BaseViewOp() {}

    inline bool operator==(const BaseViewOp& other) {
      if (this->e_type != other.e_type ||
        this->t_inputTensorInfo != other.t_inputTensorInfo ||
        this->t_outputTensorInfo != other.t_outputTensorInfo ||
        !this->isEqual(other)) {
        return false;
      }
      return true;
    }

    inline bool operator!=(const BaseViewOp& other) {
      return !(*this == other);
    }

    // Visual parameters of view op for debugging.
    friend std::ostream& operator<<(std::ostream& out, const BaseViewOp& other);

  protected:
    ViewsType e_type = ViewsType::kInvalid;
    TensorInfo t_inputTensorInfo;
    TensorInfo t_outputTensorInfo;
};

}  // end of namespace torch_mlu
