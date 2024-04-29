#include "python/memory.h"
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>


py::dict mlu_memoryStats(int device) {
  using torch_mlu::MemoryStats;
  using torch_mlu::Stat;
  using torch_mlu::StatArray;
  using torch_mlu::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const MemoryStats stats = torch_mlu::getMemoryStats(device);
  py::dict  result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result;
}

py::list mlu_memorySnapshot() {
  using torch_mlu::SegmentInfo;
  using torch_mlu::ChunkInfo;
  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict["device"] = segmentInfo.device;
    segmentDict["address"] = segmentInfo.address;
    segmentDict["total_size"] = segmentInfo.total_size;
    segmentDict["allocated_size"] = segmentInfo.allocated_size;
    segmentDict["active_size"] = segmentInfo.active_size;
    segmentDict["stream"] = int64_t(segmentInfo.queue);
    segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

    py::list chunks;
    for (const auto& chunkInfo : segmentInfo.chunks) {
      py::dict chunkDict;
      chunkDict["size"] = chunkInfo.size;
      chunkDict["state"] = (chunkInfo.allocated ? "active_allocated" : (chunkInfo.active ? "active_pending_free" : "inactive"));
      if (chunkInfo.history) {
        py::list history;
        auto h = chunkInfo.history;
        while (h) {
          py::dict history_entry;
          history_entry["addr"] = (int64_t)h->addr;
          history_entry["real_size"] = h->real_size;
          if (h->context) {
            py::list frames;
            auto sc = (StackContext*)h->context.get();
            for (auto& f : sc->frames) {
              py::dict frame;
              frame["filename"] =
                  py::reinterpret_borrow<py::object>(f.code->co_filename);
              frame["name"] =
                  py::reinterpret_borrow<py::object>(f.code->co_name);
              frame["line"] = PyCode_Addr2Line(f.code, f.lasti);
              frames.append(std::move(frame));
            }
            history_entry["frames"] = std::move(frames);
          }
          h = h->next.get();
          history.append(std::move(history_entry));
        }
        chunkDict["history"] = std::move(history);
      }
      chunks.append(chunkDict);
    }
    segmentDict["blocks"] = chunks;

    return segmentDict;
  };

  const std::vector<SegmentInfo>& snapshot = torch_mlu::snapshot();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result;
}

void mlu_recordMemoryHistory(bool enabled) {
  torch_mlu::setContextRecorder(enabled ? StackContext::gather : nullptr);
}

void* mluCachingAllocator_raw_alloc(size_t size, cnrtQueue_t queue) {
  return torch_mlu::raw_alloc_with_queue(size, queue);
}

void mluCachingAllocator_raw_delete(void* mem_ptr) {
  return torch_mlu::raw_delete(mem_ptr);
}
