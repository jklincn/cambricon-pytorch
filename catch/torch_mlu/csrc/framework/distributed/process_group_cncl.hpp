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

#include <pybind11/chrono.h>
#include <torch/extension.h>
#include <unordered_map>

#include "cncl_utils.h"
#include "framework/core/queue_guard.h"
#include "framework/core/notifier.h"
#include "aten/utils/tensor_util.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>


namespace torch_mlu {

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* CNCL_BLOCKING_WAIT = "CNCL_BLOCKING_WAIT";

constexpr const char* CNCL_BACKEND_NAME = "cncl";

// ProcessGroupCNCL implements CNCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All CNCL functions provided by this class are asynchronous functions. More
// specifically, each CNCL call is scheduled on a separate MLU stream that is
// different from the current MLU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the MLU stream their
// code works on needs to wait for the CNCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkCNCL::wait() or WorkCNCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkCNCL::isSuccess() and WorkCNCL::isCompleted() will always
// return true since ProcessGroupCNCL is single threaded. Every single CNCL
// or MLU failure will simply raise std::runtime_error.
//
// Therefore, WorkCNCL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkCNCL::finishedMLUExecution() is a helper function only
// provided by ProcessGroupCNCL to check if the CNCL operation of WorkCNCL has
// finished execution on the MLU (not just scheduled).
//
// Example on using the CNCL process group
//
//   ProcessGroupCNCL pg(store, rank, size);
//   std::shared_ptr<WorkCNCL> work = pg.allreduce(tensors);
//
//   // At this point, CNCL kernel has already by queued successfully
//   // Now, let current stream wait for the CNCL to finish, originally this function is
//   // async operation as well, but currently MLU is sync.
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class ProcessGroupCNCL : public c10d::ProcessGroup {
 public:
  class WorkCNCL : public c10d::Work,
                   public std::enable_shared_from_this<WorkCNCL> {
   public:
    // Constructor takes a list of MLU devices
    WorkCNCL(const std::vector<at::Device>& devices); // NOLINT

    virtual ~WorkCNCL();

    // Checks if request has completed. In this specific case of CNCL, it checks
    // if the CNCL operation has completed on the MLU in its own CNCL queue.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for CNCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    // Let current stream wait on the completing of the CNCL work
    // Throws on exceptions
    void synchronize() override;

    // Synchronize streams by blocking each on the CNCL stream
    void synchronizeQueues();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    std::vector<at::Tensor> result() override;

   protected:
    // The cached list of MLU devices to operate on
    std::vector<at::Device> devices_;

    // Clone of blockingWait_ from ProcessGroupCNCL.
    bool blockingWait_ = false;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrier_tensors_;

    // The start MLU notifiers of CNCL operator tracking this work item on
    // multiple MLU devices. These start MLU events are needed by desync
    // debugging if enabled.
    std::shared_ptr<std::vector<torch_mlu::Notifier>> cncl_start_notifiers_;

    // The end MLU notifiers of CNCL operator tracking this work item on
    // multiple MLU devices.
    std::shared_ptr<std::vector<torch_mlu::Notifier>> cncl_end_notifiers_;

   private:
    // Just checks whether MLU execution has completed, without modifying
    // exception_ptr.
    bool finishedMLUExecutionInternal() const;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // Store a reference to CNCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    friend class ProcessGroupCNCL;
  };

  struct Options : c10d::ProcessGroup::Options {
    // NOTE: timeout in ProcessGroupCNCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(
        bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule CNCL operations on high priority MLU streams
    bool is_high_priority_stream;
  };

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }
  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any CNCL communicators. A single CNCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another CNCL
  // communicator. These CNCL communicators are cached and reused if possible.
  ProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  virtual ~ProcessGroupCNCL();

  const std::string getBackendName() const override {
      return std::string(CNCL_BACKEND_NAME);
  }

  void startCoalescing() override;

  void endCoalescing(std::vector<c10::intrusive_ptr<c10d::Work>>& reqs) override;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
      c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dst_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int src_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  // Create a new ProcessGroupCNCL instance
  static c10::intrusive_ptr<c10d::ProcessGroup> createProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store> &store,
      int rank,
      int size,
      const std::chrono::duration<float> &timeout);

  static void groupStart();

  static void groupEnd();

 protected:
  // Helper that broadcasts cncl clique ID to all ranks through the store
  void broadcastCNCLCliqueID(
    cnclCliqueId* cncl_id,
    const bool is_p2p_op,
    const std::string& p2p_key,
    const int p2p_rank);

  // Helper that either looks up the cached CNCL communicators or creates
  // a new set of CNCL communicators as a cache entry
  std::vector<std::shared_ptr<CNCLComm>>& getCNCLComm(
      const std::string& devices_key, const std::vector<at::Device>& devices,
      c10d::OpType op_type, const int p2p_rank = 0,
      const bool is_send_recv_self = false);

  // The store is used to broadcast the CNCL unique ID of rank 0.
  c10::intrusive_ptr<c10d::Store> store_;

  const c10::intrusive_ptr<Options> options_;

  // The number of CNCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t cncl_comm_counter_{0};

  // The CNCL communicator that the process group has cached.
  // The key is a list of MLU devices that an operation is operating on
  // The MLU devices are stored in a device sequence and the cache CNCL
  // communicator is associated with this MLU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::vector<std::shared_ptr<CNCLComm>>>
      dev_cncl_comm_map_;

  // The MLU queues used by CNCL kernels
  std::unordered_map<std::string, std::vector<torch_mlu::Queue>>
      cncl_queues_;

  // The MLU notifiers used to sync CNCL queues
  std::unordered_map<std::string, std::vector<torch_mlu::Notifier>> cncl_notifiers_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  bool coalescing_active_ = false;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<std::vector<at::Device>> coalescedDevices_;

 private:
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collective(std::vector<at::Tensor>& inputs,
                                            std::vector<at::Tensor>& outputs,
                                            Fn fn, c10d::OpType op_type);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> pointToPoint(
      std::vector<at::Tensor>& tensors, Fn fn, int peer, c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  // The number of active cnclGroupStart() calls. This counter will be increased
  // by 1 when cnclGroupStart() is called and decreased by 1 when cnclGroupEnd()
  // is called.
  static thread_local uint64_t cnclActiveGroupCounter_;

  // Mutex to guard maps like dev_cncl_comm_map_
  std::mutex mutex_;
};

}  // namespace torch_mlu
