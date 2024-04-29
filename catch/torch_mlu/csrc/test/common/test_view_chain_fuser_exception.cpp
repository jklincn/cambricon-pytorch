#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>

#include "aten/viewchain/viewNodeFuser.h"

namespace torch_mlu {

TEST(runViewNodeFusionOpt, base_view_node_fusion_test) {
  ViewNodeFusionOpt opt;
  std::vector<BaseViewOp::baseViewOpPtr> v_empty_ptr;
  // test runViewNodeFusionOpt in base class.
  try {
    opt.runViewNodeFusionOpt(v_empty_ptr);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

TEST(runViewNodeReplaceOpt, base_view_node_fusion_test) {
  ViewNodeFusionOpt opt;
  BaseViewOp::baseViewOpPtr ptr;
  // test runViewNodeReplaceOpt in base class.
  try {
    opt.runViewNodeReplaceOpt(ptr);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

TEST(runViewNodeMoveOpt, base_view_node_fusion_test) {
  ViewNodeFusionOpt opt;
  std::vector<BaseViewOp::baseViewOpPtr> v_empty_ptr;
  BaseViewOp::baseViewOpPtr ptr;
  // test runViewNodeMoveOpt in base class.
  try {
    opt.runViewNodeMoveOpt(v_empty_ptr);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

}  // namespace torch_mlu
