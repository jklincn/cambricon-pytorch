#include <gtest/gtest.h>
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
TEST(SliceTest, slice_internal_test_zero_dim_input) {
  at::Tensor input = at::empty({});
  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 0;
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("slice() cannot be applied to a 0-dim tensor.", e.msg().c_str());
      throw;
    }		  
  }, c10::Error);
}

TEST(SliceTest, slice_internal_test_non_positive_step) {
  at::Tensor input = at::empty({0});
  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 0;
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("slice step must be positive", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

// Following tests focus on MLU limitation check

TEST(SliceTest, slice_internal_test_illegal_dim) {

  at::Tensor input = at::empty({0});
  int64_t dim = -1;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 1;
  // negative dim
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("slice dim must be non negative and the value of dim must less than the dims of input", e.msg().c_str());
      throw;
    }
  }, c10::Error);
  // dim greater than dim of input
  dim = 5;
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("slice dim must be non negative and the value of dim must less than the dims of input", e.msg().c_str());
      throw;
    }
  }, c10::Error);  
}

TEST(SliceTest, slice_internal_test_illegal_start_and_end) {
  at::Tensor input = at::randn({2});
  //start >= 0
  int64_t dim = 1;
  int64_t start = -1;
  int64_t end = 0;
  int64_t step = 1;

  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("slice dim must be non negative and the value of dim must less than the dims of input", e.msg().c_str());
      throw;
    }
  }, c10::Error);
  // end >= start
  start = 0;
  end = -1;
  dim = 0;
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("when using slice, values of start and end need to meet the following conditions: 0 <= start <= end <=sizes[dim]", e.msg().c_str());
      throw;
    }
  }, c10::Error);
  // end <= sizes[dim]
  end = 300;
  EXPECT_THROW({
    try
    {
      auto out = torch_mlu::ops::cnnl_slice_internal(input, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("when using slice, values of start and end need to meet the following conditions: 0 <= start <= end <=sizes[dim]", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, slice_internal_test_non_contiguous_input) {
  at::Tensor input = at::randn({2,3});
  auto x = at::native::transpose(input,0,1);
  int64_t dim = 1;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 1;
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_slice_internal(x, dim, start, end, step);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("Input tensor needs to be contiguous.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, multi_dims_slice_internal_test_illegal_dims) {
  // empty dims
  at::Tensor input = at::randn({2,3});
  std::vector<int> empty_dims{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> steps{};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(input, empty_dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("When using multi dims slice, size of dims needs to be greater than zero, and less or equal to dims of input.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
  // dims with size greater than input dims
  std::vector<int> dims{0,0,1,1};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(input, dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("When using multi dims slice, size of dims needs to be greater than zero, and less or equal to dims of input.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, multi_dims_slice_internal_test_illegal_starts) {
  at::Tensor input = at::randn({2,3});
  std::vector<int> dims{0,1};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> steps{};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(input, dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("The size of starts needs to be equal to the size with dims.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, multi_dims_slice_internal_test_illegal_ends) {
  at::Tensor input = at::randn({2,3});
  std::vector<int> dims{0,1};
  std::vector<int> starts{0,0};
  std::vector<int> ends{};
  std::vector<int> steps{};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(input, dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("The size of ends needs to be equal to the size with dims.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, multi_dims_slice_internal_test_illegal_steps) {
  at::Tensor input = at::randn({2,3});
  std::vector<int> dims{0,1};
  std::vector<int> starts{0,0};
  std::vector<int> ends{0,0};
  std::vector<int> steps{};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(input, dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("The size of steps needs to be equal to the size with dims.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}

TEST(SliceTest, multi_dims_slice_internal_test_non_contiguous_input) {
  at::Tensor input = at::randn({2,3});
  auto x = at::native::transpose(input,0,1);
  std::vector<int> dims{0,1};
  std::vector<int> starts{0,0};
  std::vector<int> ends{0,0};
  std::vector<int> steps{0,0};
  EXPECT_THROW({
    try
    {
      auto out = ops::cnnl_multi_dims_slice_internal(x, dims, starts, ends, steps);
    }
    catch(const c10::Error& e)
    {
      EXPECT_STREQ("Input tensor needs to be contiguous.", e.msg().c_str());
      throw;
    }
  }, c10::Error);
}
}
