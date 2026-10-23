#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace sglang {
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


// C4 slot bytes are split between value and scale runs inside a 37440B page.
template <bool C4>
__device__ uint64_t* mpu_word(int64_t base, int64_t slot, int64_t word, int64_t words) {
  if constexpr (C4) {
    return reinterpret_cast<uint64_t*>(base) + (slot / 64) * 4680 +
      (word < 72 ? (slot % 64) * 72 + word : 4608 + slot % 64);
  } else {
    return reinterpret_cast<uint64_t*>(base) + slot * words + word;
  }
}

template <bool C4, bool ToPeer>
__device__ void mpu_copy(int64_t peer, int64_t local, int64_t pslot,
                         int64_t lslot, int64_t words) {
  for (int64_t w = threadIdx.x % 32; w < words; w += 32) {
    auto* p = mpu_word<C4>(peer, pslot, w, words);
    auto* l = mpu_word<C4>(local, lslot, w, words);
    auto* src = ToPeer ? l : p;
    auto* dst = ToPeer ? p : l;
    uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(src) : "memory");
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(dst), "l"(value) : "memory");
  }
  if constexpr (ToPeer) __threadfence_system();
}

__global__ void mpu_zero_kernel(const int64_t* ptrs, int64_t words) {
  auto* dst = reinterpret_cast<uint64_t*>(ptrs[blockIdx.y]);
  for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < words;
       i += int64_t(gridDim.x)*blockDim.x) {
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(dst+i), "l"(uint64_t{0}) : "memory");
  }
  __threadfence_system();
}

void mpu_zero(tvm::ffi::TensorView ptrs, int64_t bytes) {
  auto device = host::LaunchKernel::resolve_device(ptrs.device());
  dim3 grid(std::min<int64_t>((bytes/8+255)/256, 65535), ptrs.shape()[0]);
  host::LaunchKernel(grid, 256, device)(mpu_zero_kernel,
      static_cast<const int64_t*>(ptrs.data_ptr()), bytes/8);
}

template <bool C4, bool ToPeer>
__global__ void mpu_transfer_kernel(
    const int64_t* peer, const int64_t* local, const int64_t* pi, const int64_t* li,
    int64_t count, int64_t pcap, int64_t lcap, int64_t words, int64_t selected) {
  auto layer = selected < 0 ? blockIdx.y : selected;
  for (int64_t i = int64_t(blockIdx.x)*8 + threadIdx.x/32; i < count; i += int64_t(gridDim.x)*8) {
    auto p = pi[i], l = li[i];
    if (p < 0 || p >= pcap || l < 0 || l >= lcap) { asm volatile("trap;"); return; }
    mpu_copy<C4, ToPeer>(peer[layer], local[layer], p, l, words);
  }
}

void mpu_transfer(tvm::ffi::TensorView peer, tvm::ffi::TensorView local,
    tvm::ffi::TensorView pi, tvm::ffi::TensorView li,
    int64_t pcap, int64_t lcap, int64_t bytes, int64_t selected, bool to_peer, bool c4) {
  auto device = host::LaunchKernel::resolve_device(peer.device());
  auto count = pi.shape()[0];
  if (count == 0) return;
  dim3 grid(std::min<int64_t>((count+7)/8,65535), selected < 0 ? peer.shape()[0] : 1);
  auto launch = [&](auto kernel) {
    host::LaunchKernel(grid,256,device)(kernel,
      static_cast<const int64_t*>(peer.data_ptr()), static_cast<const int64_t*>(local.data_ptr()),
      static_cast<const int64_t*>(pi.data_ptr()), static_cast<const int64_t*>(li.data_ptr()),
      count, pcap, lcap, bytes/8, selected);
  };
  if (c4) {
    if (to_peer) launch(mpu_transfer_kernel<true,true>); else launch(mpu_transfer_kernel<true,false>);
  } else {
    if (to_peer) launch(mpu_transfer_kernel<false,true>); else launch(mpu_transfer_kernel<false,false>);
  }
}

template <bool C4>
__global__ void mpu_replay_kernel(const int64_t* peer, const int64_t* local,
    const int64_t* src, const int32_t* dst, const int32_t* counts, const int32_t* real,
    int64_t width, int64_t layer, int64_t pcap, int64_t lcap, int64_t words) {
  int bid=blockIdx.y;
  if (bid >= real[0]) return;
  auto count=counts[bid];
  if (count < 0 || count > width) { asm volatile("trap;"); return; }
  for (int64_t i=int64_t(blockIdx.x)*8+threadIdx.x/32; i<count; i+=int64_t(gridDim.x)*8) {
    auto p=src[bid*width+i]; auto l=dst[bid*width+i];
    if (p < 0 || p >= pcap || l < 0 || l >= lcap) { asm volatile("trap;"); return; }
    mpu_copy<C4,false>(peer[layer],local[layer],p,l,words);
  }
}

void mpu_replay(tvm::ffi::TensorView peer, tvm::ffi::TensorView local,
    tvm::ffi::TensorView src, tvm::ffi::TensorView dst, tvm::ffi::TensorView counts,
    tvm::ffi::TensorView real, int64_t pcap, int64_t lcap, int64_t bytes,
    int64_t layer, bool c4) {
  if (src.shape()[0] == 0) return;
  auto device=host::LaunchKernel::resolve_device(peer.device());
  dim3 grid(4,src.shape()[0]);
  auto launch = [&](auto kernel) {
    host::LaunchKernel(grid,256,device)(kernel,
      static_cast<const int64_t*>(peer.data_ptr()), static_cast<const int64_t*>(local.data_ptr()),
      static_cast<const int64_t*>(src.data_ptr()), static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(counts.data_ptr()), static_cast<const int32_t*>(real.data_ptr()),
      src.shape()[1], layer, pcap, lcap, bytes/8);
  };
  if (c4) launch(mpu_replay_kernel<true>); else launch(mpu_replay_kernel<false>);
}
} // namespace

} // namespace sglang
