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

#include "aten/viewchain/baseViewOps.h"

namespace torch_mlu {

class PermuteOp : public BaseViewOp {
  public:
    friend class PermuteFusionOpt;
    explicit PermuteOp(const at::IntArrayRef& dims)
                        : BaseViewOp(ViewsType::kPermute),
                          v_dims(dims.vec()) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline std::vector<int64_t> getParameterDims() const {
      return this->v_dims;
    }
    inline std::vector<int64_t> getParameterDims() {
      return this->v_dims;
    }
    inline bool isEqual(const BaseViewOp& other) override {
      const PermuteOp& op = dynamic_cast<const PermuteOp&>(other);
      return this->v_dims == op.v_dims;
    }
  private:
    // IntArrayRef not support deep copy, so change to std::vector.
    std::vector<int64_t> v_dims;
};

class SliceOp : public BaseViewOp {
  public:
    friend class SliceFusionOpt;
    explicit SliceOp(int64_t dim, int64_t start,
                     int64_t end, int64_t step)
                      : BaseViewOp(ViewsType::kSlice),
                        v_dims({static_cast<int>(dim)}),
                        v_starts({static_cast<int>(start)}),
                        v_ends({static_cast<int>(end)}),
                        v_steps({static_cast<int>(step)}) { }
    explicit SliceOp(const std::vector<int>& dims,
                     const std::vector<int>& starts,
                     const std::vector<int>& ends,
                     const std::vector<int>& steps)
                      : BaseViewOp(ViewsType::kSlice),
                      v_dims(dims), v_starts(starts),
                      v_ends(ends), v_steps(steps) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline bool isEqual(const BaseViewOp& other) override {
      const SliceOp& op = dynamic_cast<const SliceOp&>(other);
      return this->v_dims == op.v_dims && this->v_starts == op.v_starts
        && this->v_ends == op.v_ends && this->v_steps == op.v_steps;
    }

  private:
    std::vector<int> v_dims, v_starts, v_ends, v_steps;
};

class ExpandOp : public BaseViewOp {
  public:
    explicit ExpandOp(const at::IntArrayRef& size,
                      const bool implicit)
                        : BaseViewOp(ViewsType::kExpand),
                        v_dims(size.vec()), b_implicit(implicit) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline bool isEqual(const BaseViewOp& other) override {
      const ExpandOp& op = dynamic_cast<const ExpandOp&>(other);
      return this->v_dims == op.v_dims && this->b_implicit == op.b_implicit;
    }
  private:
    // IntArrayRef not support deep copy, so change to std::vector.
    std::vector<int64_t> v_dims;
    bool b_implicit;
};

class ReshapeOp : public BaseViewOp {
  public:
    friend class ReshapeFusionOpt;
    explicit ReshapeOp(const at::IntArrayRef& shape):
                         BaseViewOp(ViewsType::kReshape),
                         v_shape(shape.vec()) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline bool isEqual(const BaseViewOp& other) override {
      const ReshapeOp& op = dynamic_cast<const ReshapeOp&>(other);
      return this->v_shape == op.v_shape;
    }
  private:
    // IntArrayRef not support deep copy, so change to std::vector.
    std::vector<int64_t> v_shape;
};

class UnfoldOp : public BaseViewOp {
  public:
    explicit UnfoldOp(const int64_t dimension,
                      const int64_t size,
                      const int64_t step): BaseViewOp(ViewsType::kUnfold),
                        i_dimension(dimension), i_size(size), i_step(step) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline bool isEqual(const BaseViewOp& other) override {
      const UnfoldOp& op = dynamic_cast<const UnfoldOp&>(other);
      return this->i_dimension == op.i_dimension &&
        this->i_size == op.i_size &&
        this->i_step == op.i_step;
    }
  private:
    int64_t i_dimension, i_size, i_step;
};

// (TODO)shangang: torch.diagonal is not support in catch side,
// So delete this now.
/* class DiagonalOp : public BaseViewOp {
  public:
    explicit DiagonalOp(const int64_t offset,
                        const int64_t dim1,
                        const int64_t dim2): BaseViewOp(ViewsType::kDiagonal),
                          i_offset(offset), i_dim1(dim1), i_dim2(dim2) { }
    bool inferShape() override;
    bool hasCnnlSpecificFunction() override;
    at::Tensor runCnnlSpecificFunction(const at::Tensor& input,
                                       c10::optional<at::Tensor> output) override;
    std::string parameterToString() const override;
    inline bool isEqual(const BaseViewOp& other) override {
      const DiagonalOp& op = dynamic_cast<const DiagonalOp&>(other);
      return this->i_offset == op.i_offset &&
        this->i_dim1 == op.i_dim1 &&
        this->i_dim2 == op.i_dim2;
    }
  private:
    int64_t i_offset, i_dim1, i_dim2;
}; */

}  // end of namespace torch_mlu
