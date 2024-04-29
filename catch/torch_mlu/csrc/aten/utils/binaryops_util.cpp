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

#include <algorithm>
#include "aten/utils/binaryops_util.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {

std::vector<at::ScalarType> type_vec = {at::kHalf, at::kFloat};

at::Tensor scalar_to_tensor_with_dtype(const at::Tensor& tensor,
                                       at::ScalarType ct_type) {
  if (tensor.is_mlu() || tensor.numel() != 1) {
    return tensor;
  }

  if (ct_type == at::ScalarType::Undefined) {
    ct_type = tensor.scalar_type();
  }
  at::Tensor result;
  at::Scalar scalar = tensor.item();
  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(ct_type, "create_mlu_scalar_tensor", [&] {
    result = at::full({1}, scalar.to<scalar_t>(),
                      tensor.options().dtype(ct_type).device(at::kMLU));
  });
  return result;
}

bool get_promote_type(const at::TensorList& tensors, at::ScalarType& common_type) {
  if (tensors.size() == 0) {
    return false;
  }
  bool need_promote_dtype = false;
  common_type = tensors[0].scalar_type();
  for (auto i = 0; i < tensors.size(); ++i) {
    if (tensors[0].scalar_type() != tensors[i].scalar_type()) {
      need_promote_dtype = true;
      break;
    }
  }

  if (need_promote_dtype) {
    at::native::ResultTypeState state = {};
    for (auto i = 0; i < tensors.size(); ++i) {
      state = at::native::update_result_type_state(tensors[i], state);
    }
    common_type = at::native::result_type(state);
  }
  return need_promote_dtype;
}

// use cast op to support dtype promote
std::vector<at::Tensor> promote_inputs(const at::TensorList& tensors) {
  at::ScalarType common_type = at::ScalarType::Undefined;
  bool need_promote_dtype = get_promote_type(tensors, common_type);
  std::vector<at::Tensor> outputs;
  for (auto i = 0; i < tensors.size(); ++i) {
    if (need_promote_dtype) {
      outputs.emplace_back(convertTensorType(tensors[i], common_type));
    } else {
      outputs.emplace_back(tensors[i]);
    }
  }
  return outputs;
}

at::Tensor wrapped_scalar_tensor(const at::Scalar& scalar) {
  auto tensor = c10::scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

}  // namespace torch_mlu
