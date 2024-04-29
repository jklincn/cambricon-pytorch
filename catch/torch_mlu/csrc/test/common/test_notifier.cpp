#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <unordered_set>
#include <caffe2/core/logging.h>
#include "framework/core/device.h"
#include "framework/core/notifier.h"
#include "framework/core/queue.h"
#include "c10/util/Optional.h"
#include "cnrt.h"

namespace torch_mlu {

struct ipc_sample {
  cnrtIpcMemHandle m_handle;
  cnrtIpcNotifierHandle n_handle;
  int start;
};

TEST(NotifierTest, ipc_handle) {
  unsigned int flags = CNRT_NOTIFIER_INTERPROCESS |
                       CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  Notifier notifier(flags);
  void* dev_mem;
  char* host_mem;
  size_t mem_size = 4 * sizeof(char);

  // shared info between child and parent.
  struct ipc_sample* info = (struct ipc_sample* )mmap(NULL, sizeof(struct ipc_sample),
  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
  info->start = 0;
  pid_t pid = fork();

  int major = -1;
  int device_id;
  cnrtGetDevice(&device_id);
  TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&major,
                    cnrtAttrComputeCapabilityMajor, device_id));
  if (major != 5) GTEST_SKIP() << "Skipping: IPC Handle does not supported on current device.";

  if (pid < 0) {
    printf("error fork\n");
  } else if (pid == 0) {
    printf("child process..\n");
    host_mem = (char*)calloc(1, mem_size);
    memset(host_mem, 'B', mem_size);
    while(0 == info->start) { sched_yield(); } // wait until parent process set IPC handle.
    TORCH_CNRT_CHECK(cnrtMapMemHandle(&dev_mem, info->m_handle, 0));
    Notifier notifier_new(notifier.device_index(), &info->n_handle);
    notifier_new.wait(getCurrentQueue());
    TORCH_CNRT_CHECK(cnrtMemcpyAsync((void *)host_mem, dev_mem, mem_size,
            getCurrentQueue(), cnrtMemcpyDevToHost));
    sleep(5); // wait for cp complete.
    ASSERT_TRUE(*host_mem == 'A'); // parent process set host_mem to 'A'.
    TORCH_CNRT_CHECK(cnrtUnMapMemHandle(dev_mem));
    free(host_mem);
    exit(testing::Test::HasFailure());
  } else {
    printf("parent process.. child's pid is %d\n", pid);
    host_mem = (char*)calloc(1, mem_size);
    memset(host_mem, 'A', mem_size);
    notifier.ipc_handle(&info->n_handle);
    TORCH_CNRT_CHECK(cnrtMalloc(&dev_mem, mem_size));
    TORCH_CNRT_CHECK(cnrtAcquireMemHandle(&info->m_handle, dev_mem));
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(dev_mem, (void *)host_mem, mem_size,
          getCurrentQueue(), cnrtMemcpyHostToDev));
    notifier.place();
    __sync_add_and_fetch(&info->start, 1);// tell child process IPC handle is ready.
    int status = -1;
    if (waitpid(pid, &status, 0) < 0) {
      printf("%s, waitpid error.\n", __func__);
      exit(EXIT_FAILURE);
    }
    EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS);
    EXPECT_NE(WIFEXITED(status), 0);
    TORCH_CNRT_CHECK(cnrtFree(dev_mem));
    free(host_mem);
  }
}

TEST(NotifierTest, placeNotifier) {
  Notifier notifier;
  notifier.place();
  auto queue = getQueueFromPool();
  notifier.place(queue);
}

TEST(NotifierTest, syncNotifier) {
  Notifier notifier;
  notifier.place();
  notifier.synchronize();
}

TEST(NotifierTest, elapsed_time) {
  Notifier start(CNRT_NOTIFIER_DEFAULT);
  Notifier end(CNRT_NOTIFIER_DEFAULT);
  start.place();
  end.place();
  end.synchronize();
  float time = start.elapsed_time(end);
  ASSERT_TRUE(time >= 0);
}

TEST(NotifierTest, hardware_time) {
  Notifier start;
  Notifier end;
  start.place();
  end.place();
  end.synchronize();
  float time = start.hardware_time(end);
  ASSERT_TRUE(time >= 0);
}

TEST(NotifierTest, queue_wait_notifier) {
  Notifier notifier;
  notifier.place();
  notifier.wait(getCurrentQueue());
  notifier.synchronize();
  notifier.place();
  notifier.wait(getQueueFromPool());
  notifier.synchronize();
}

TEST(NotifierTest, query_and_wait_notifier) {
  Notifier notifier;
  ASSERT_TRUE(notifier.query());
  notifier.synchronize();
  notifier.place();
  notifier.synchronize();
  ASSERT_TRUE(notifier.query());
}

TEST(NotifierTest, move_test) {
  Notifier no_1;
  Notifier no_2;
  no_2 = std::move(no_1);
  TORCH_CHECK_EQ(no_1.device_index(), no_2.device_index());
  TORCH_CHECK_EQ(no_1.isCreated(), no_2.isCreated());
  
  Notifier start = std::move(Notifier(CNRT_NOTIFIER_DEFAULT));
  Notifier end = std::move(Notifier(CNRT_NOTIFIER_DEFAULT));
  start.place();
  end.place();
  end.synchronize();
  float time = start.elapsed_time(end);
  ASSERT_TRUE(time >= 0);
}

}  // namespace torch_mlu
