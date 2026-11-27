// Qwen/Quest adaptation: GPU-launched CUDA Graph memcpy, not an SM payload copy.
// The host creates/uploads immutable copy graphs once. During model graph replay,
// a GPU node initiates each memcpy graph. Small GPU checkpoint kernels approximate
// priority control; these are not XM-Sparse's native copy-engine semaphore commands.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string xm_error;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

using U64 = unsigned long long;

__device__ void fail(U64* counters, int index) {
  atomicAdd(counters + index, 1ULL);
  asm volatile("trap;");
}

__global__ void checkpoint_kernel(int* gate, U64* counters, U64 timeout_cycles) {
  const U64 start = clock64();
  while (atomicAdd(gate, 0) != 0) {
    if (clock64() - start > timeout_cycles) fail(counters, 7);
    __nanosleep(128);
  }
  atomicAdd(counters + 4, clock64() - start);
}

__global__ void complete_kernel(
    const U64* epoch, U64* ready, const U64* issued, U64* counters,
    int layers, int layer, U64 bytes, U64 segments) {
  const U64 key = *epoch * layers + layer + 1;
  const int slot = layer % 2;
  if (atomicAdd(const_cast<U64*>(issued) + slot, 0ULL) != key) fail(counters, 7);
  atomicAdd(counters + 1, 1ULL);
  atomicAdd(counters + 2, bytes);
  atomicAdd(counters + 3, segments);
  __threadfence_system();
  atomicExch(ready + slot, key);
}

__global__ void launch_kernel(
    cudaGraphExec_t copy_graph, const U64* epoch, U64* ready, U64* issued,
    U64* counters, int layers, int layer, bool host_parent) {
  const U64 key = *epoch * layers + layer + 1;
  const int slot = layer % 2;
  const U64 old_issued = atomicAdd(issued + slot, 0ULL);
  if (old_issued == key) return;  // begin_step and read_layer may name the same copy.
  if (old_issued != atomicAdd(ready + slot, 0ULL)) fail(counters, 7);
  atomicExch(issued + slot, key);
  atomicAdd(counters + 0, 1ULL);
  if (host_parent) atomicAdd(counters + 6, 1ULL);
  __threadfence_system();
  const cudaError_t status = cudaGraphLaunch(copy_graph, cudaStreamGraphFireAndForget);
  if (status != cudaSuccess) fail(counters, 5);
}

__global__ void begin_kernel(U64* epoch, int* gate) {
  atomicAdd(epoch, 1ULL);
  atomicExch(gate, 0);
}

__global__ void wait_kernel(
    const U64* epoch, U64* ready, U64* counters, int layers, int layer,
    U64 timeout_cycles) {
  const U64 key = *epoch * layers + layer + 1;
  const U64 start = clock64();
  while (atomicAdd(ready + layer % 2, 0ULL) != key) {
    if (atomicAdd(ready + layer % 2, 0ULL) > key || clock64() - start > timeout_cycles)
      fail(counters, 7);
    __nanosleep(128);
  }
  __threadfence();
}

__global__ void gate_kernel(int* gate, int protected_operation) {
  __threadfence_system();
  atomicExch(gate, protected_operation);
}

struct CopyGraph {
  cudaGraph_t child = nullptr;
  cudaGraphExec_t child_exec = nullptr;
  cudaGraph_t parent = nullptr;
  cudaGraphExec_t parent_exec = nullptr;
  U64* epoch = nullptr;
  U64* ready = nullptr;
  U64* issued = nullptr;
  U64* counters = nullptr;
  int device = 0;
  int layers = 0;
  int layer = 0;
  U64 timeout_cycles = 0;

  ~CopyGraph() {
    if (parent_exec) cudaGraphExecDestroy(parent_exec);
    if (parent) cudaGraphDestroy(parent);
    if (child_exec) cudaGraphExecDestroy(child_exec);
    if (child) cudaGraphDestroy(child);
  }
};

void kernel_node(
    cudaGraphNode_t* node, cudaGraph_t graph, cudaGraphNode_t previous,
    void* function, void** arguments) {
  cudaKernelNodeParams params{};
  params.func = function;
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = arguments;
  check(cudaGraphAddKernelNode(node, graph, previous ? &previous : nullptr,
                              previous ? 1 : 0, &params), "cudaGraphAddKernelNode");
}

template <class F> int guarded(F function) {
  try { function(); xm_error.clear(); return 0; }
  catch (const std::exception& error) { xm_error = error.what(); return -1; }
}
}  // namespace

extern "C" const char* xm_bounds_last_error() { return xm_error.c_str(); }

extern "C" int xm_bounds_mapped_pointer(uint64_t host_pointer, int device, uint64_t* output) {
  return guarded([&] {
    check(cudaSetDevice(device), "cudaSetDevice");
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(host_pointer)),
          "cudaPointerGetAttributes(pinned bounds)");
    if (attributes.type != cudaMemoryTypeHost)
      throw std::invalid_argument("Quest bounds backing must be CUDA pinned host memory");
    void* pointer = nullptr;
    check(cudaHostGetDevicePointer(&pointer, reinterpret_cast<void*>(host_pointer), 0),
          "cudaHostGetDevicePointer");
    *output = reinterpret_cast<uint64_t>(pointer);
  });
}

extern "C" int xm_bounds_create_copy(
    uint64_t source, uint64_t destination, uint64_t bytes, uint64_t chunk_bytes,
    uint64_t epoch, uint64_t ready, uint64_t issued, uint64_t gate,
    uint64_t counters, int layers, int layer, int device, uint64_t stream,
    uint64_t* output) {
  return guarded([&] {
    if (!source || !destination || !bytes || !chunk_bytes || layers <= 0 ||
        layer < 0 || layer >= layers)
      throw std::invalid_argument("Invalid immutable Quest DMA graph geometry");
    check(cudaSetDevice(device), "cudaSetDevice");
    int runtime = 0;
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    if (runtime < 12000) throw std::runtime_error("Device graph launch needs CUDA >= 12");
    int clock_rate = 0;
    check(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device),
          "cudaDeviceGetAttribute(clock rate)");
    auto* graph = new CopyGraph;
    graph->epoch = reinterpret_cast<U64*>(epoch);
    graph->ready = reinterpret_cast<U64*>(ready);
    graph->issued = reinterpret_cast<U64*>(issued);
    graph->counters = reinterpret_cast<U64*>(counters);
    graph->device = device;
    graph->layers = layers;
    graph->layer = layer;
    graph->timeout_cycles = static_cast<U64>(clock_rate) * 1000ULL * 30ULL;
    try {
      check(cudaGraphCreate(&graph->child, 0), "cudaGraphCreate(child)");
      cudaGraphNode_t previous = nullptr;
      const U64 segments = (bytes + chunk_bytes - 1) / chunk_bytes;
      auto* gate_pointer = reinterpret_cast<int*>(gate);
      for (uint64_t offset = 0; offset < bytes; offset += chunk_bytes) {
        cudaGraphNode_t checkpoint = nullptr;
        void* checkpoint_args[] = {&gate_pointer, &graph->counters, &graph->timeout_cycles};
        kernel_node(&checkpoint, graph->child, previous,
                    reinterpret_cast<void*>(checkpoint_kernel), checkpoint_args);
        cudaGraphNode_t copy = nullptr;
        check(cudaGraphAddMemcpyNode1D(
            &copy, graph->child, &checkpoint, 1,
            reinterpret_cast<void*>(destination + offset),
            reinterpret_cast<const void*>(source + offset),
            std::min(chunk_bytes, bytes - offset), cudaMemcpyDefault),
            "cudaGraphAddMemcpyNode1D(mapped host -> HBM)");
        previous = copy;
      }
      cudaGraphNode_t complete = nullptr;
      U64 total_bytes = bytes;
      U64 segment_count = segments;
      void* complete_args[] = {
          &graph->epoch, &graph->ready, &graph->issued, &graph->counters,
          &graph->layers, &graph->layer, &total_bytes, &segment_count};
      kernel_node(&complete, graph->child, previous,
                  reinterpret_cast<void*>(complete_kernel), complete_args);
      check(cudaGraphInstantiate(&graph->child_exec, graph->child,
                                  cudaGraphInstantiateFlagDeviceLaunch),
            "cudaGraphInstantiate(device copy graph)");
      check(cudaGraphUpload(graph->child_exec, reinterpret_cast<cudaStream_t>(stream)),
            "cudaGraphUpload(device copy graph)");

      // Eager calls use a host-launched parent graph on a separate stream. The
      // payload graph is still initiated by launch_kernel on the GPU.
      check(cudaGraphCreate(&graph->parent, 0), "cudaGraphCreate(parent)");
      cudaGraphNode_t launcher = nullptr;
      bool host_parent = true;
      void* launch_args[] = {
          &graph->child_exec, &graph->epoch, &graph->ready, &graph->issued,
          &graph->counters, &graph->layers, &graph->layer, &host_parent};
      kernel_node(&launcher, graph->parent, nullptr,
                  reinterpret_cast<void*>(launch_kernel), launch_args);
      check(cudaGraphInstantiate(&graph->parent_exec, graph->parent, 0),
            "cudaGraphInstantiate(eager parent graph)");
      check(cudaGraphUpload(graph->parent_exec, reinterpret_cast<cudaStream_t>(stream)),
            "cudaGraphUpload(eager parent graph)");
      *output = reinterpret_cast<uint64_t>(graph);
    } catch (...) { delete graph; throw; }
  });
}

extern "C" int xm_bounds_begin(uint64_t epoch, uint64_t gate, int device, uint64_t stream) {
  return guarded([&] {
    check(cudaSetDevice(device), "cudaSetDevice");
    begin_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<U64*>(epoch), reinterpret_cast<int*>(gate));
    check(cudaGetLastError(), "begin_kernel");
  });
}

extern "C" int xm_bounds_launch(uint64_t handle, uint64_t stream, int captured) {
  return guarded([&] {
    auto* graph = reinterpret_cast<CopyGraph*>(handle);
    if (!graph) throw std::invalid_argument("Destroyed Quest copy graph");
    check(cudaSetDevice(graph->device), "cudaSetDevice");
    const auto target_stream = reinterpret_cast<cudaStream_t>(stream);
    if (captured) {
      launch_kernel<<<1, 1, 0, target_stream>>>(
          graph->child_exec, graph->epoch, graph->ready, graph->issued,
          graph->counters, graph->layers, graph->layer, false);
      check(cudaGetLastError(), "device graph launcher node");
    } else {
      check(cudaGraphLaunch(graph->parent_exec, target_stream),
            "cudaGraphLaunch(eager parent)");
    }
  });
}

extern "C" int xm_bounds_wait(uint64_t handle, uint64_t stream) {
  return guarded([&] {
    auto* graph = reinterpret_cast<CopyGraph*>(handle);
    if (!graph) throw std::invalid_argument("Destroyed Quest copy graph");
    check(cudaSetDevice(graph->device), "cudaSetDevice");
    wait_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        graph->epoch, graph->ready, graph->counters, graph->layers, graph->layer,
        graph->timeout_cycles);
    check(cudaGetLastError(), "wait_kernel");
  });
}

extern "C" int xm_bounds_protect(uint64_t gate, int value, int device, uint64_t stream) {
  return guarded([&] {
    check(cudaSetDevice(device), "cudaSetDevice");
    gate_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<int*>(gate), value);
    check(cudaGetLastError(), "gate_kernel");
  });
}

extern "C" int xm_bounds_destroy(uint64_t handle) {
  return guarded([&] {
    auto* graph = reinterpret_cast<CopyGraph*>(handle);
    if (!graph) return;
    check(cudaSetDevice(graph->device), "cudaSetDevice");
    check(cudaDeviceSynchronize(), "drain Quest DMA graphs before destruction");
    delete graph;
  });
}
