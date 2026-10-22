#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

void mpu_check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

// Setup is host control only; it never launches work on the storage GPU.
void mpu_enable_peer_access(int64_t compute_device, int64_t storage_device) {
  if (compute_device == storage_device) throw std::invalid_argument("MPU devices must be distinct");
  int previous = 0;
  mpu_check_cuda(cudaGetDevice(&previous), "cudaGetDevice");
  mpu_check_cuda(cudaSetDevice(static_cast<int>(compute_device)), "cudaSetDevice(compute)");
  try {
    int accessible = 0;
    mpu_check_cuda(cudaDeviceCanAccessPeer(&accessible, compute_device, storage_device), "cudaDeviceCanAccessPeer");
    if (!accessible) throw std::runtime_error("MPU peer load/store is unavailable; no CPU fallback");
    int compute_uva = 0, storage_uva = 0;
    mpu_check_cuda(cudaDeviceGetAttribute(&compute_uva, cudaDevAttrUnifiedAddressing, compute_device), "compute UVA");
    mpu_check_cuda(cudaDeviceGetAttribute(&storage_uva, cudaDevAttrUnifiedAddressing, storage_device), "storage UVA");
    if (!compute_uva || !storage_uva) throw std::runtime_error("MPU requires CUDA unified virtual addressing");
    const auto result = cudaDeviceEnablePeerAccess(static_cast<int>(storage_device), 0);
    if (result == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();  // An existing mapping is valid; clear only this setup status.
    } else {
      mpu_check_cuda(result, "cudaDeviceEnablePeerAccess(compute -> storage)");
    }
  } catch (...) {
    cudaSetDevice(previous);
    throw;
  }
  mpu_check_cuda(cudaSetDevice(previous), "restore CUDA device");
}

__device__ __forceinline__ void
mpu_copy_record(const uint64_t* source, uint64_t* destination, int lane, int64_t words) {
  for (int64_t offset = lane; offset < words; offset += 32) {
    uint64_t value;
    // Coherent global loads (not the non-coherent read-only cache) are required
    // when the same peer record can be updated by a previous write-through.
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(source + offset) : "memory");
    asm volatile("st.global.cg.u64 [%0], %1;" ::"l"(destination + offset), "l"(value) : "memory");
  }
}

__global__ void mpu_zero_kernel(const int64_t* peer_ptrs, int64_t words_per_layer) {
  auto* destination = reinterpret_cast<uint64_t*>(peer_ptrs[blockIdx.y]);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words_per_layer;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    asm volatile("st.global.cg.u64 [%0], %1;" ::"l"(destination + i), "l"(uint64_t{0}) : "memory");
  }
  // Completion/event ordering on the compute GPU must publish its peer stores.
  __threadfence_system();
}

template <bool ToPeer>
__global__ void mpu_transfer_kernel(
    const int64_t* peer_ptrs,
    const int64_t* local_ptrs,
    const int64_t* peer_indices,
    const int64_t* local_indices,
    int64_t count,
    int64_t shard_rows,
    int64_t num_shards,
    int64_t layers,
    int64_t words,
    int64_t selected_layer,
    int64_t local_rows,
    int64_t peer_rows) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (token >= count) return;
  const int lane = threadIdx.x % 32;
  const int64_t layer = selected_layer < 0 ? blockIdx.y : selected_layer;
  const int64_t global_row = peer_indices[token];
  const int64_t local_row = local_indices[token];
  if (global_row < 0 || global_row >= peer_rows || local_row < 0 || local_row >= local_rows) {
    // Keep the guard in optimized/NDEBUG builds: raw UVA must not go out of bounds.
    asm volatile("trap;");
    return;
  }
  const int64_t shard = global_row / shard_rows;
  const int64_t peer_row = global_row % shard_rows;
#pragma unroll
  for (int side = 0; side < 2; ++side) {
    auto* peer = reinterpret_cast<uint64_t*>(peer_ptrs[(shard * 2 + side) * layers + layer]) + peer_row * words;
    auto* local = reinterpret_cast<uint64_t*>(local_ptrs[side * layers + layer]) + local_row * words;
    if constexpr (ToPeer) {
      mpu_copy_record(local, peer, lane, words);
    } else {
      mpu_copy_record(peer, local, lane, words);
    }
  }
  if constexpr (ToPeer) __threadfence_system();
}

void mpu_zero_peer_kv(tvm::ffi::TensorView peer_ptrs, int64_t shard_rows, int64_t item_size_bytes) {
  const auto device = host::LaunchKernel::resolve_device(peer_ptrs.device());
  const int64_t words = shard_rows * (item_size_bytes / 8);
  const auto blocks = static_cast<unsigned int>(std::min<int64_t>((words + 255) / 256, 65535));
  const dim3 grid(blocks, peer_ptrs.shape()[0] * 2 * peer_ptrs.shape()[2]);
  host::LaunchKernel(grid, 256, device)(mpu_zero_kernel, static_cast<const int64_t*>(peer_ptrs.data_ptr()), words);
}

void mpu_transfer_peer_kv(
    tvm::ffi::TensorView peer_ptrs,
    tvm::ffi::TensorView local_ptrs,
    tvm::ffi::TensorView peer_indices,
    tvm::ffi::TensorView local_indices,
    int64_t shard_rows,
    int64_t item_size_bytes,
    bool to_peer,
    int64_t layer_id,
    int64_t local_rows,
    int64_t peer_rows) {
  const auto device = host::LaunchKernel::resolve_device(peer_ptrs.device());
  const int64_t count = peer_indices.shape()[0];
  const int64_t layers = peer_ptrs.shape()[2];
  const dim3 grid((count + 7) / 8, layer_id < 0 ? layers : 1);
  auto launch = [&](auto kernel) {
    host::LaunchKernel(grid, 256, device)(
        kernel,
        static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        static_cast<const int64_t*>(local_ptrs.data_ptr()),
        static_cast<const int64_t*>(peer_indices.data_ptr()),
        static_cast<const int64_t*>(local_indices.data_ptr()),
        count,
        shard_rows,
        peer_ptrs.shape()[0],
        layers,
        item_size_bytes / 8,
        layer_id,
        local_rows,
        peer_rows);
  };
  if (to_peer)
    launch(mpu_transfer_kernel<true>);
  else
    launch(mpu_transfer_kernel<false>);
}

}  // namespace
