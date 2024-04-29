#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/utils/tensor_util.h"

namespace torch_mlu {

// testcases of is_geometry_contiguous
TEST(TestGeometryContiguous, TestIsGeometryContiguous) {
  // 1D && 2D && 3D
  {
    auto t = at::empty({3});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 2D
  {
    auto t = at::empty({3, 4});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 3D
  {
    auto t = at::empty({3, 4, 5});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 6D
  {
    auto t = at::empty({3, 4, 5, 6, 7, 8});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 4D && 5D
  {
    auto t = at::empty({3, 4, 5, 6});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  {
    auto t = at::empty({3, 4, 5, 6, 7});
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  {
    auto t = at::empty({3, 4, 5, 6}, at::TensorOptions()
      .memory_format(at::MemoryFormat::ChannelsLast));
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  {
    auto t = at::empty({3, 4, 5, 6, 7}, at::TensorOptions()
      .memory_format(at::MemoryFormat::ChannelsLast3d));
    ASSERT_TRUE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // Assert false
  // 4D not contiguous
  {
    auto t = at::empty({3, 4, 5, 6}, at::TensorOptions()).slice(1, 2, 3);
    ASSERT_FALSE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 5D not contiguous
  {
    auto t = at::empty({3, 4, 5, 6, 7}, at::TensorOptions()
      .memory_format(at::MemoryFormat::ChannelsLast3d)).permute({4, 2, 3, 1, 0});
    ASSERT_FALSE(is_geometry_contiguous(t.sizes(), t.strides()));
  }
  // 3D not contiguous
  {
    std::vector<int64_t> sizes = {4, 5, 6};
    std::vector<int64_t> strides = {20, 6, 1};  // 20 != 5 * 6
    ASSERT_FALSE(is_geometry_contiguous(at::IntArrayRef(sizes),
                                        at::IntArrayRef(strides)));
  }
  // 6D not contiguous
  {
    std::vector<int64_t> sizes = {3, 4, 5, 6, 7, 8};
    std::vector<int64_t > strides = {1344, 336, 48, 8, 1};  // 48 != 7 * 8
    ASSERT_FALSE(is_geometry_contiguous(at::IntArrayRef(sizes),
                                        at::IntArrayRef(strides)));
  }
}

}  // namespace torch_mlu

