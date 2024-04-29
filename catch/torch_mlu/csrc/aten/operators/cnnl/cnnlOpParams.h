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

#include <c10/core/ScalarType.h>
#include <c10/core/MemoryFormat.h>

namespace torch_mlu {

class CnnlOpParams {
  public:
    CnnlOpParams() = default;
    explicit CnnlOpParams(const std::vector<at::ScalarType>& support_types,
                          const std::string& name, bool implicit_type_convert = false,
                          bool allow_strided_memory = false,
                          bool allow_cpu_scalar = false,
                          bool allow_different_input_types = false,
                          c10::MemoryFormat support_memory_format
                            = c10::MemoryFormat::Preserve,
                          bool allow_different_input_sizes = false)
                          :support_types_(support_types), name_(name),
                           allow_implicit_type_convert_(implicit_type_convert),
                           allow_strided_memory_(allow_strided_memory),
                           allow_cpu_scalar_(allow_cpu_scalar),
                           allow_different_input_types_(allow_different_input_types),
                           support_memory_format_(support_memory_format),
                           allow_different_input_sizes_(allow_different_input_sizes) {
      auto it_find_double = std::find(support_types_.begin(), support_types_.end(), at::kDouble);
      auto it_find_long = std::find(support_types_.begin(), support_types_.end(), at::kLong);
      auto it_find_complexD = std::find(support_types_.begin(),
                     support_types_.end(), at::kComplexDouble);
      // Do we need to add two bool flag like support_long and support_double in here?
      if (it_find_double != support_types_.end() && it_find_long != support_types_.end() &&
          it_find_complexD != support_types_.end()) {
        this->allow_64bit_caculate_ = true;
      }
    }

    // Only support move parameter.
    CnnlOpParams& setInputMixedType(
        std::vector<std::vector<at::ScalarType>>&& input_mixed_types_list);

    CnnlOpParams& setInputMixedType(
        const std::vector<std::vector<at::ScalarType>>& input_mixed_types_list);

    inline bool isSupportMixedInputTypes() const {
      return allow_mix_input_types_;
    }
    inline bool isSupportMixedInputTypes() {
      return allow_mix_input_types_;
    }

    CnnlOpParams(const CnnlOpParams&);
    CnnlOpParams(CnnlOpParams&&);
    CnnlOpParams& operator=(const CnnlOpParams&);
    CnnlOpParams& operator=(CnnlOpParams&&);

  private:
    inline void assignCommonVariable(const CnnlOpParams& other) {
      this->name_ = other.name_;
      this->allow_implicit_type_convert_ = other.allow_implicit_type_convert_;
      this->allow_strided_memory_ = other.allow_strided_memory_;
      this->allow_mix_input_types_ = other.allow_mix_input_types_;
      this->allow_different_input_sizes_ = other.allow_different_input_sizes_;
      this->allow_different_input_types_ = other.allow_different_input_types_;
      this->support_memory_format_ = other.support_memory_format_;
    }

  public:
    // CNNL op support types list, this is different with aten op support types.
    // Like aten add op is support a lot of types, but CNNL add op is only support
    // float, half and int.
    // Also different CNNL op support type list is different, not like gpu. so we
    // need to define a support type list for each CNNL op.
    // Note: Set default support types to float and half, because those two types is
    // base type in catch.
    std::vector<at::ScalarType> support_types_ = {at::kDouble, at::kFloat, at::kHalf};

    // Op name in catch params, and there is a little different with aten op name.
    // Such as 'add.Tensor', 'add.out', 'add_.Tensor' in aten, 'opTensor' is catch
    // params name.
    std::string name_;

    // For historical reasons, some ops has been implicit converted input type to
    // CNNL support type in catch side. Like add, div and ...
    bool allow_implicit_type_convert_ = false;

    // Almost CNNL op not support strided memory, so we need to check input and output
    // tensor memory format in catch side. This option is for future use, maybe some op
    // need to support strided memory in high priority, and others are not urgent.
    bool allow_strided_memory_ = false;

    // Not using now.
    // All CNNL ops not support 64bit calculate, so we need to check input and output
    // tensor memory type in catch side. This option is for future use, maybe some op
    // need to support 64bit calculate in high priority, and others are not urgent.
    bool allow_64bit_caculate_ = false;

    // Not using now, pytorch different ops may cast to cpu scalar to different type,
    // some ops to common type of input tensor, some ops to type of output tensor.
    // Pytorch do cpu scalar to tensor type in op kernel side, so we need do more
    // research for this function.
    // Almost CNNL op not support cpu scalar, so we need to check input and output
    // tensor in catch side. This option is for future use, maybe some op need to
    // support cpu scalar in high priority, and others are not urgent.
    bool allow_cpu_scalar_ = false;

    // Index ops have different input types and only support contiguous memory format,
    // and this is very different with element-wise op. So add those two ops in here.
    bool allow_different_input_types_ = false;
    c10::MemoryFormat support_memory_format_ = c10::MemoryFormat::Preserve;
    bool allow_different_input_sizes_ = false;

    // Nullary and unary op is not supported mixed input types, so don't set this option.
    // Dynamic type cast is supported in gpu side, but not all ops support dynamic type
    // cast in catch side. Only several ops support specific mixed input types in catch side.
    // Like logic op support float + int32 input types, and output type is bool.
    // Maybe data format of mix_type_list is a little weird, but it is easy to use.
    // We need to add input types and output type in this data format.
    // {{half, int, bool}, {float, int, bool}}
    // Maybe data format of std::vector<at::ScalarType> is more easy to understand. But
    // this is not enough for CNNL op. Like logic op, int32 + float32 is supported, but
    // int32 + half is not supported.
    // Note1: need to add input types first and then add output type.
    // Note2: Only support one output tensor right now, and doesn't find
    // any op in pytorch.
    std::vector<std::vector<at::ScalarType>> input_mixed_types_list_;

  private:
    // This option is setted by invoke_input_mixed_types apt, and only for internal use.
    // More details in input_mixed_types_list_.
    bool allow_mix_input_types_ = false;
};

// Always return a CNNL op params. For most CNNL op, the params is same.
// So if op name has been registered, return the specific params.
// Otherwise return the default params.
const CnnlOpParams& getCnnlOpParams(const std::string& name);

}  // namespace torch_mlu
