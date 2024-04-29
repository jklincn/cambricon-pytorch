#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {
namespace ops {

void cnnl_record_stream(at::Tensor& self, c10::Stream s) {
  torch_mlu::recordQueue(self.storage().data_ptr(), torch_mlu::Queue::unpack(s.pack()));
}

}  // namespace ops
}  // namespace torch_mlu
