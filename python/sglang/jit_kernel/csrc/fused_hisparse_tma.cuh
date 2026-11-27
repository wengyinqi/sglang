#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Experimental SM90 BF16 GQA path. Every selected row is read directly from
// hot HBM or mapped host DDR by cp.async.bulk into shared memory. There is no
// standalone host-to-HBM swap-in before attention. The scalar FP32 attention
// path is a correctness-first implementation, not the author's FlashMLA code.
namespace xm_hisparse_tma {

constexpr int kQHeads = 32;
constexpr int kKVHeads = 4;
constexpr int kGroup = kQHeads / kKVHeads;
constexpr int kDim = 128;
constexpr int kThreads = (kGroup + 1) * 32;
constexpr int kPartialWidth = kDim + 2;

__device__ __forceinline__ uint32_t smem_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void init_barrier(uint64_t* p) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_address(p)) : "memory");
}

__device__ __forceinline__ void expect_bytes(uint64_t* p, uint32_t bytes) {
  uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
               : "=l"(state) : "r"(smem_address(p)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_copy(void* dst, const void* src, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];"
      :: "r"(smem_address(dst)), "l"(src), "r"(smem_address(bar)) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* p, uint32_t phase) {
  uint32_t ready;
  do {
    asm volatile("{ .reg .pred done; mbarrier.try_wait.parity.shared::cta.b64 done, [%1], %2; "
                 "selp.u32 %0, 1, 0, done; }"
                 : "=r"(ready) : "r"(smem_address(p)), "r"(phase) : "memory");
  } while (!ready);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

template <int TopK, int Splits, int TileRows>
struct Shared {
  static constexpr int kSpan = (TopK + Splits - 1) / Splits;
  alignas(16) bf16_t k[2][TileRows][kDim];
  alignas(16) bf16_t v[2][TileRows][kDim];
  alignas(8) uint64_t barrier[2];
  int ordered[kSpan];
  int count;
};

template <int TopK, int Splits, int TileRows>
__device__ __forceinline__ void issue_tile(
    Shared<TopK, Splits, TileRows>& s,
    int tile,
    int request,
    int kv_head,
    const int64_t* host_locs,
    const int32_t* device_locs,
    const bf16_t* host_k,
    const bf16_t* host_v,
    const bf16_t* hot_k,
    const bf16_t* hot_v) {
  const int buffer = tile & 1;
  const int first = tile * TileRows;
  const int rows = min(TileRows, s.count - first);
  expect_bytes(&s.barrier[buffer], rows * 2 * kDim * sizeof(bf16_t));
  for (int r = 0; r < rows; ++r) {
    const int selected = s.ordered[first + r];
    const int64_t h = host_locs[request * TopK + selected];
    const int d = device_locs[request * TopK + selected];
    const int64_t row = h >= 0 ? h : d;
    const int64_t offset = (row * kKVHeads + kv_head) * kDim;
    const bf16_t* source_k = (h >= 0 ? host_k : hot_k) + offset;
    const bf16_t* source_v = (h >= 0 ? host_v : hot_v) + offset;
    tma_copy(s.k[buffer][r], source_k, &s.barrier[buffer]);
    tma_copy(s.v[buffer][r], source_v, &s.barrier[buffer]);
  }
}

template <int TopK, int Splits, int TileRows>
__global__ void attention_kernel(
    const bf16_t* q,
    bf16_t* hot_k,
    bf16_t* hot_v,
    const bf16_t* host_k,
    const bf16_t* host_v,
    int64_t host_rows,
    int64_t hot_rows,
    const int64_t* host_locs,
    const int32_t* device_locs,
    const int32_t* actual_lens,
    const int32_t* num_real_reqs,
    float* partial,
    float scaling,
    float logits_soft_cap) {
  const int request = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  if (num_real_reqs[0] < 0 || num_real_reqs[0] > gridDim.x ||
      actual_lens[request] < 0 || actual_lens[request] > TopK) {
    asm volatile("trap;");
    return;
  }
  if (request >= num_real_reqs[0]) return;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  __shared__ Shared<TopK, Splits, TileRows> s;

  if (tid == 0) {
    init_barrier(&s.barrier[0]);
    init_barrier(&s.barrier[1]);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    s.count = 0;
    const int first = split * Shared<TopK, Splits, TileRows>::kSpan;
    const int last = min(min(actual_lens[request], TopK), first + Shared<TopK, Splits, TileRows>::kSpan);
    // Resident-first within each split. Only traversal order changes; the
    // original Quest selection and every valid selected row are preserved.
    for (int misses = 0; misses < 2; ++misses) {
      for (int i = first; i < last; ++i) {
        const int d = device_locs[request * TopK + i];
        const int64_t h = host_locs[request * TopK + i];
        if (d < 0) continue;  // padding during capture/short sequences
        if (d >= hot_rows || h < -1 || h >= host_rows) {
          asm volatile("trap;");
          return;
        }
        if (static_cast<int>(h >= 0) == misses) s.ordered[s.count++] = i;
      }
    }
    if (s.count) issue_tile<TopK, Splits, TileRows>(
        s, 0, request, kv_head, host_locs, device_locs, host_k, host_v, hot_k, hot_v);
  }
  __syncthreads();

  const int query_head = kv_head * kGroup + warp - 1;
  float q_values[4] = {};
  float numerator[4] = {};
  float maximum = -INFINITY;
  float denominator = 0.0f;
  if (warp > 0) {
    for (int j = 0; j < 4; ++j) q_values[j] = __bfloat162float(q[(request * kQHeads + query_head) * kDim + lane + j * 32]);
  }
  const int tiles = (s.count + TileRows - 1) / TileRows;
  for (int tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    wait_barrier(&s.barrier[buffer], (tile / 2) & 1);
    __syncthreads();
    // The other shared buffer is free after the previous iteration's CTA
    // barrier. Producer runs ahead while the eight query warps consume this
    // tile. The mbarrier phase toggles on each reuse of a buffer.
    if (tid == 0 && tile + 1 < tiles) issue_tile<TopK, Splits, TileRows>(
        s, tile + 1, request, kv_head, host_locs, device_locs, host_k, host_v, hot_k, hot_v);
    const int rows = min(TileRows, s.count - tile * TileRows);
    if (warp > 0) {
      for (int r = 0; r < rows; ++r) {
        float dot = 0.0f;
        for (int j = 0; j < 4; ++j) dot = fmaf(q_values[j], __bfloat162float(s.k[buffer][r][lane + j * 32]), dot);
        float score = warp_sum(dot) * scaling;
        if (logits_soft_cap > 0) score = logits_soft_cap * tanhf(score / logits_soft_cap);
        const float next_max = fmaxf(maximum, score);
        const float alpha = expf(maximum - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        for (int j = 0; j < 4; ++j) numerator[j] = numerator[j] * alpha + beta * __bfloat162float(s.v[buffer][r][lane + j * 32]);
        maximum = next_max;
      }
      // One warp per KV head refills raw BF16; the other seven query heads
      // must not duplicate this store. Reads finish before buffer reuse.
      if (warp == 1) {
        for (int i = lane; i < rows * kDim; i += 32) {
          const int r = i / kDim;
          const int d = i % kDim;
          const int selected = s.ordered[tile * TileRows + r];
          if (host_locs[request * TopK + selected] >= 0) {
            const int destination = device_locs[request * TopK + selected];
            const int64_t offset = (static_cast<int64_t>(destination) * kKVHeads + kv_head) * kDim + d;
            hot_k[offset] = s.k[buffer][r][d];
            hot_v[offset] = s.v[buffer][r][d];
          }
        }
      }
    }
    __syncthreads();
  }
  if (warp > 0) {
    const int64_t base = ((static_cast<int64_t>(request) * kQHeads + query_head) * Splits + split) * kPartialWidth;
    for (int j = 0; j < 4; ++j) partial[base + lane + j * 32] = numerator[j];
    if (lane == 0) {
      partial[base + kDim] = maximum;
      partial[base + kDim + 1] = denominator;
    }
  }
}

template <int Splits>
__global__ void merge_kernel(const float* partial, bf16_t* output, const int32_t* num_real_reqs) {
  const int request = blockIdx.x;
  const int query_head = blockIdx.y;
  const int d = threadIdx.x;
  if (d >= kDim) return;
  if (request >= num_real_reqs[0]) {
    output[(request * kQHeads + query_head) * kDim + d] = __float2bfloat16(0.0f);
    return;
  }
  float maximum = -INFINITY;
  for (int split = 0; split < Splits; ++split) {
    const int64_t base = ((static_cast<int64_t>(request) * kQHeads + query_head) * Splits + split) * kPartialWidth;
    if (partial[base + kDim + 1] > 0) maximum = fmaxf(maximum, partial[base + kDim]);
  }
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (int split = 0; split < Splits; ++split) {
    const int64_t base = ((static_cast<int64_t>(request) * kQHeads + query_head) * Splits + split) * kPartialWidth;
    const float sum = partial[base + kDim + 1];
    if (sum > 0) {
      const float weight = expf(partial[base + kDim] - maximum);
      numerator += weight * partial[base + d];
      denominator += weight * sum;
    }
  }
  output[(request * kQHeads + query_head) * kDim + d] = __float2bfloat16(denominator > 0 ? numerator / denominator : 0.0f);
}

inline int64_t mapped_source_pointer(tvm::ffi::TensorView source) {
  using namespace host;
  RuntimeCheck(is_type<bf16_t>(source.dtype()), "TMA source must contain BF16 bytes");
  RuntimeCheck(reinterpret_cast<uintptr_t>(source.data_ptr()) % 16 == 0, "TMA source must be 16-byte aligned");
  cudaPointerAttributes attributes{};
  RuntimeDeviceCheck(cudaPointerGetAttributes(&attributes, source.data_ptr()));
  RuntimeCheck(attributes.type == cudaMemoryTypeHost || attributes.type == cudaMemoryTypeDevice,
               "TMA source must be registered/mapped host memory or an explicit HBM control");
  RuntimeCheck(attributes.devicePointer != nullptr, "Host source has no CUDA mapped address");
  return reinterpret_cast<int64_t>(attributes.devicePointer);
}

template <int TopK, int Splits, int TileRows>
void fused_retrieval_attention(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView hot_k,
    tvm::ffi::TensorView hot_v,
    int64_t host_k_pointer,
    int64_t host_v_pointer,
    int64_t host_rows,
    tvm::ffi::TensorView miss_host_locs,
    tvm::ffi::TensorView device_locs,
    tvm::ffi::TensorView actual_lens,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView partial,
    tvm::ffi::TensorView output,
    double scaling,
    double logits_soft_cap) {
  using namespace host;
  static_assert(SGL_ARCH_HOPPER_OR_GREATER, "This TMA path requires SM90 or later");
  RuntimeCheck(q.ndim() == 3 && q.shape()[1] == kQHeads && q.shape()[2] == kDim, "TMA GQA requires [B,32,128]");
  RuntimeCheck(is_type<bf16_t>(q.dtype()) && is_type<bf16_t>(hot_k.dtype()) && is_type<bf16_t>(hot_v.dtype()), "TMA GQA requires BF16 Q/K/V");
  RuntimeCheck(q.device().device_type == kDLCUDA, "TMA attention requires CUDA");
  RuntimeCheck(hot_k.ndim() == 3 && hot_v.ndim() == 3 && hot_k.shape()[0] == hot_v.shape()[0] &&
               hot_k.shape()[1] == kKVHeads && hot_k.shape()[2] == kDim &&
               hot_v.shape()[1] == kKVHeads && hot_v.shape()[2] == kDim,
               "TMA hot KV requires matching [rows,4,128]");
  RuntimeCheck(host_rows > 0 && host_k_pointer > 0 && host_v_pointer > 0 &&
               host_k_pointer % 16 == 0 && host_v_pointer % 16 == 0,
               "Invalid mapped TMA backing");
  const int64_t batch = q.shape()[0];
  const DLDevice device = q.device();
  LaunchKernel(dim3(batch, kKVHeads, Splits), kThreads, device)(
      attention_kernel<TopK, Splits, TileRows>,
      static_cast<const bf16_t*>(q.data_ptr()), static_cast<bf16_t*>(hot_k.data_ptr()), static_cast<bf16_t*>(hot_v.data_ptr()),
      reinterpret_cast<const bf16_t*>(host_k_pointer), reinterpret_cast<const bf16_t*>(host_v_pointer), host_rows, hot_k.shape()[0],
      static_cast<const int64_t*>(miss_host_locs.data_ptr()), static_cast<const int32_t*>(device_locs.data_ptr()),
      static_cast<const int32_t*>(actual_lens.data_ptr()), static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      static_cast<float*>(partial.data_ptr()), static_cast<float>(scaling), static_cast<float>(logits_soft_cap));
  LaunchKernel(dim3(batch, kQHeads), kDim, device)(
      merge_kernel<Splits>, static_cast<const float*>(partial.data_ptr()), static_cast<bf16_t*>(output.data_ptr()),
      static_cast<const int32_t*>(num_real_reqs.data_ptr()));
}

}  // namespace xm_hisparse_tma
