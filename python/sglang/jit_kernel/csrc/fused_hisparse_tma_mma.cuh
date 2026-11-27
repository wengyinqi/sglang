#pragma once

#include "fused_hisparse_tma.cuh"

// Opt-in BF16 MMA backend. It preserves descriptorless TMA,
// mapped-DDR reads and raw refill without a selected-KV pre-copy.
// Register packing follows SM90 m16n8k16 and the frozen FA2 m16n16 wrapper.
namespace xm_hisparse_tma_mma {
using namespace xm_hisparse_tma;
constexpr int kMMAThreads = 5 * 32;

__device__ __forceinline__ uint32_t packed_pair(bf16_t a, bf16_t b) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(a)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(b)) << 16);
}

__device__ __forceinline__ float rcp_approx(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ void mma_16x8(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_16x16(float* c, const uint32_t* a, const uint32_t* b) {
  mma_16x8(c, a, b);
  mma_16x8(c + 4, a, b + 2);
}

template <int TopK>
struct Shared {
  union {
    struct {
      alignas(16) bf16_t k[2][32][128];
      alignas(16) bf16_t v[2][32][128];
    } payload;
    struct {
      float numerator[4][8][128];
      float maximum[4][8];
      float denominator[4][8];
    } state;
  };
  alignas(8) uint64_t barrier[2];
  int ordered[TopK];
  int count;
};

template <int TopK>
__device__ __forceinline__ void issue_tile(
    Shared<TopK>& s, int tile, int request, int kv_head,
    const int64_t* host_locs, const int32_t* device_locs,
    const bf16_t* host_k, const bf16_t* host_v,
    const bf16_t* hot_k, const bf16_t* hot_v) {
  const int buffer = tile & 1;
  const int first = tile * 32;
  const int rows = min(32, s.count - first);
  expect_bytes(&s.barrier[buffer], rows * 512);
  for (int row = 0; row < rows; ++row) {
    const int selected = s.ordered[first + row];
    const int64_t h = host_locs[request * TopK + selected];
    const int d = device_locs[request * TopK + selected];
    const int64_t offset = ((h >= 0 ? h : d) * 4 + kv_head) * 128;
    tma_copy(s.payload.k[buffer][row], (h >= 0 ? host_k : hot_k) + offset, &s.barrier[buffer]);
    tma_copy(s.payload.v[buffer][row], (h >= 0 ? host_v : hot_v) + offset, &s.barrier[buffer]);
  }
}

template <int TopK, int Splits>
__global__ void attention_kernel(
    const bf16_t* q, bf16_t* hot_k, bf16_t* hot_v,
    const bf16_t* host_k, const bf16_t* host_v, int64_t host_rows, int64_t hot_rows,
    const int64_t* host_locs, const int32_t* device_locs,
    const int32_t* actual_lens, const int32_t* num_real_reqs,
    const int32_t* kv_chunk_size, float* partial, float scaling, float soft_cap) {
  const int request = blockIdx.x, kv_head = blockIdx.y, split = blockIdx.z;
  if (num_real_reqs[0] < 0 || num_real_reqs[0] > gridDim.x ||
      actual_lens[request] < 0 || actual_lens[request] > TopK ||
      kv_chunk_size[0] < 128 || kv_chunk_size[0] > max(128, TopK) ||
      (TopK + kv_chunk_size[0] - 1) / kv_chunk_size[0] > Splits) {
    asm volatile("trap;");
    return;
  }
  if (request >= num_real_reqs[0]) return;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int row_q = lane / 4, pair = (lane % 4) * 2;
  const int query_head = kv_head * 8 + row_q;
  __shared__ Shared<TopK> s;
  if (tid == 0) {
    init_barrier(&s.barrier[0]);
    init_barrier(&s.barrier[1]);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    s.count = 0;
    const int first = split * kv_chunk_size[0];
    const int last = min(actual_lens[request], first + kv_chunk_size[0]);
    for (int i = first; i < last; ++i) {
      const int d = device_locs[request * TopK + i];
      const int64_t h = host_locs[request * TopK + i];
      if (d < 0) continue;
      if (d >= hot_rows || h < -1 || h >= host_rows) {
        asm volatile("trap;");
        return;
      }
      s.ordered[s.count++] = i;
    }
    if (s.count) issue_tile(s, 0, request, kv_head, host_locs, device_locs, host_k, host_v, hot_k, hot_v);
  }
  __syncthreads();
  uint32_t query_frag[8][4];
  float output_frag[8][8] = {};
  float maximum = -INFINITY, denominator = 0.0f;
  const float sm_scale_log2 = (soft_cap > 0 ? soft_cap : scaling) * kLog2E;
  if (warp > 0) {
    const bf16_t* query = q + (request * 32 + query_head) * 128;
    for (int dim = 0; dim < 8; ++dim) {
      query_frag[dim][0] = *reinterpret_cast<const uint32_t*>(query + dim * 16 + pair);
      query_frag[dim][1] = 0;  // eight unused packed-query rows
      query_frag[dim][2] = *reinterpret_cast<const uint32_t*>(query + dim * 16 + pair + 8);
      query_frag[dim][3] = 0;
    }
  }
  const int tiles = (s.count + 31) / 32;
  for (int tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    wait_barrier(&s.barrier[buffer], (tile / 2) & 1);
    __syncthreads();
    if (tid == 0 && tile + 1 < tiles) issue_tile(s, tile + 1, request, kv_head, host_locs, device_locs, host_k, host_v, hot_k, hot_v);
    const int rows = min(32, s.count - tile * 32);
    if (warp == tile % 4 + 1) {
      float scores[2][8] = {};
      for (int dim = 0; dim < 8; ++dim) {
        for (int key_tile = 0; key_tile < 2; ++key_tile) {
          uint32_t keys[4] = {};
          const int key = key_tile * 16 + row_q;
          if (key < rows) {
            keys[0] = *reinterpret_cast<const uint32_t*>(s.payload.k[buffer][key] + dim * 16 + pair);
            keys[1] = *reinterpret_cast<const uint32_t*>(s.payload.k[buffer][key] + dim * 16 + pair + 8);
          }
          if (key + 8 < rows) {
            keys[2] = *reinterpret_cast<const uint32_t*>(s.payload.k[buffer][key + 8] + dim * 16 + pair);
            keys[3] = *reinterpret_cast<const uint32_t*>(s.payload.k[buffer][key + 8] + dim * 16 + pair + 8);
          }
          mma_16x16(scores[key_tile], query_frag[dim], keys);
        }
      }
      const float previous_maximum = maximum;
      for (int key_tile = 0; key_tile < 2; ++key_tile) {
        for (int item = 0; item < 8; ++item) {
          const int key = key_tile * 16 + pair + (item >= 4 ? 8 : 0) + item % 2;
          if (item % 4 >= 2 || key >= rows) scores[key_tile][item] = -INFINITY;
          else {
            if (soft_cap > 0) scores[key_tile][item] = tanhf(scores[key_tile][item] * scaling * rcp_approx(soft_cap));
            maximum = fmaxf(maximum, scores[key_tile][item]);
          }
        }
      }
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1));
      const float alpha = exp2_approx(previous_maximum * sm_scale_log2 - maximum * sm_scale_log2);
      denominator *= alpha;
      for (int dim = 0; dim < 8; ++dim) for (int item = 0; item < 8; ++item) output_frag[dim][item] *= alpha;
      uint32_t probability[2][4];
      for (int key_tile = 0; key_tile < 2; ++key_tile) {
        bf16_t p[8];
        for (int item = 0; item < 8; ++item) p[item] = __float2bfloat16(
            scores[key_tile][item] > -INFINITY ? exp2_approx(scores[key_tile][item] * sm_scale_log2 - maximum * sm_scale_log2) : 0.0f);
        for (int item = 0; item < 4; ++item) probability[key_tile][item] = packed_pair(p[item * 2], p[item * 2 + 1]);
        const uint32_t ones[2] = {0x3f803f80u, 0x3f803f80u};
        float row_sum[4] = {denominator, 0.0f, 0.0f, 0.0f};
        mma_16x8(row_sum, probability[key_tile], ones);
        denominator = row_sum[0];
      }
      for (int key_tile = 0; key_tile < 2; ++key_tile) {
        for (int dim = 0; dim < 8; ++dim) {
          uint32_t values[4];
          for (int item = 0; item < 4; ++item) {
            const int key = key_tile * 16 + pair + (item % 2) * 8;
            const int column = dim * 16 + row_q + (item / 2) * 8;
            const bf16_t a = key < rows ? s.payload.v[buffer][key][column] : __float2bfloat16(0.0f);
            const bf16_t b = key + 1 < rows ? s.payload.v[buffer][key + 1][column] : __float2bfloat16(0.0f);
            values[item] = packed_pair(a, b);
          }
          mma_16x16(output_frag[dim], probability[key_tile], values);
        }
      }
      for (int i = lane; i < rows * 128; i += 32) {
        const int row = i / 128, column = i % 128;
        const int selected = s.ordered[tile * 32 + row];
        if (host_locs[request * TopK + selected] >= 0) {
          const int64_t offset = (static_cast<int64_t>(device_locs[request * TopK + selected]) * 4 + kv_head) * 128 + column;
          hot_k[offset] = s.payload.k[buffer][row][column];
          hot_v[offset] = s.payload.v[buffer][row][column];
        }
      }
    }
    __syncthreads();
  }
  // The reference FA2 kernel executes a full four-warp iteration even when
  // the last 128-row iteration has no rows for a previously active warp.
  // Its FMUL + FFMA + EX2 rescaling is not necessarily an exact no-op: the
  // first product is rounded before the fused multiply/subtract. Preserve
  // that final masked iteration without issuing zero-byte TMA operations.
  if (warp > 0 && maximum > -INFINITY && tiles % 4 != 0 && warp > tiles % 4) {
    const float scaled_maximum = __fmul_rn(maximum, sm_scale_log2);
    const float alpha = exp2_approx(__fmaf_rn(maximum, sm_scale_log2, -scaled_maximum));
    denominator *= alpha;
    for (int dim = 0; dim < 8; ++dim) {
      for (int item = 0; item < 8; ++item) output_frag[dim][item] *= alpha;
    }
  }
  // No TMA is pending. Reuse its payload storage for the four warp states.
  if (warp > 0) {
    const int kv_warp = warp - 1;
    for (int dim = 0; dim < 8; ++dim) {
      s.state.numerator[kv_warp][row_q][dim * 16 + pair] = output_frag[dim][0];
      s.state.numerator[kv_warp][row_q][dim * 16 + pair + 1] = output_frag[dim][1];
      s.state.numerator[kv_warp][row_q][dim * 16 + pair + 8] = output_frag[dim][4];
      s.state.numerator[kv_warp][row_q][dim * 16 + pair + 9] = output_frag[dim][5];
    }
    if (lane % 4 == 0) {
      s.state.maximum[kv_warp][row_q] = maximum * sm_scale_log2;
      s.state.denominator[kv_warp][row_q] = denominator;
    }
  }
  __syncthreads();
  if (warp == 1) {
    float merged_maximum = -INFINITY, merged_denominator = 1.0f;
    for (int state = 0; state < 4; ++state) {
      const float other_sum = s.state.denominator[state][row_q];
      if (other_sum > 0) {
        const float previous = merged_maximum;
        merged_maximum = fmaxf(previous, s.state.maximum[state][row_q]);
        merged_denominator = merged_denominator * exp2_approx(previous - merged_maximum) +
                             other_sum * exp2_approx(s.state.maximum[state][row_q] - merged_maximum);
      }
    }
    const float reciprocal = merged_maximum > -INFINITY ? rcp_approx(merged_denominator) : 0.0f;
    const int64_t base = ((static_cast<int64_t>(request) * 32 + query_head) * Splits + split) * kPartialWidth;
    for (int dim = 0; dim < 8; ++dim) {
      for (int item = 0; item < 4; ++item) {
        const int column = dim * 16 + pair + (item / 2) * 8 + item % 2;
        float numerator = 0.0f;
        for (int state = 0; state < 4; ++state) if (s.state.denominator[state][row_q] > 0) {
          numerator += s.state.numerator[state][row_q][column] * exp2_approx(s.state.maximum[state][row_q] - merged_maximum);
        }
        partial[base + column] = __bfloat162float(__float2bfloat16(numerator * reciprocal));
      }
    }
    if (lane % 4 == 0) {
      partial[base + 128] = reciprocal > 0 ? log2_approx(merged_denominator) + merged_maximum : -INFINITY;
      partial[base + 129] = reciprocal > 0 ? 1.0f : 0.0f;
    }
  }
}

template <int Splits>
__global__ void merge_kernel(const float* partial, bf16_t* output, const int32_t* real) {
  const int request = blockIdx.x, head = blockIdx.y, column = threadIdx.x;
  if (request >= real[0]) {
    output[(request * 32 + head) * 128 + column] = __float2bfloat16(0.0f);
    return;
  }
  float group_output[8], group_lse[8];
  for (int group = 0; group < 8; ++group) {
    float maximum = -INFINITY, denominator = 1.0f, numerator = 0.0f;
    bool present = false;
    for (int split = group; split < Splits; split += 8) {
      const int64_t base = ((static_cast<int64_t>(request) * 32 + head) * Splits + split) * kPartialWidth;
      if (partial[base + 129] > 0) {
        const float previous = maximum;
        maximum = fmaxf(previous, partial[base + 128]);
        const float alpha = exp2_approx(previous - maximum), beta = exp2_approx(partial[base + 128] - maximum);
        denominator = denominator * alpha + beta;
        numerator = numerator * alpha + partial[base + column] * beta;
        present = true;
      }
    }
    group_output[group] = present ? __bfloat162float(__float2bfloat16(__fdividef(numerator, denominator))) : 0.0f;
    group_lse[group] = present ? maximum + log2_approx(denominator) : -INFINITY;
  }
  float maximum = -INFINITY, denominator = 1.0f, numerator = 0.0f;
  bool present = false;
  for (int group = 0; group < 8; ++group) if (group_lse[group] > -INFINITY) {
    const float previous = maximum;
    maximum = fmaxf(previous, group_lse[group]);
    const float alpha = exp2_approx(previous - maximum), beta = exp2_approx(group_lse[group] - maximum);
    denominator = denominator * alpha + beta;
    numerator = numerator * alpha + group_output[group] * beta;
    present = true;
  }
  output[(request * 32 + head) * 128 + column] = __float2bfloat16(present ? __fdividef(numerator, denominator) : 0.0f);
}

template <int TopK, int Splits, int TileRows>
void fused_retrieval_attention(
    tvm::ffi::TensorView q, tvm::ffi::TensorView hot_k, tvm::ffi::TensorView hot_v,
    int64_t host_k_pointer, int64_t host_v_pointer, int64_t host_rows,
    tvm::ffi::TensorView host_locs, tvm::ffi::TensorView device_locs,
    tvm::ffi::TensorView actual_lens, tvm::ffi::TensorView real,
    tvm::ffi::TensorView chunk_size, tvm::ffi::TensorView partial,
    tvm::ffi::TensorView output, double scaling, double soft_cap) {
  using namespace host;
  static_assert(SGL_ARCH_HOPPER_OR_GREATER && TileRows == 32 && TopK <= Splits * 128);
  RuntimeCheck(q.ndim() == 3 && q.shape()[1] == 32 && q.shape()[2] == 128 && is_type<bf16_t>(q.dtype()), "MMA TMA requires BF16 Q[B,32,128]");
  RuntimeCheck(q.device().device_type == kDLCUDA, "MMA TMA attention requires CUDA");
  RuntimeCheck(is_type<bf16_t>(hot_k.dtype()) && is_type<bf16_t>(hot_v.dtype()), "MMA TMA requires BF16 hot KV");
  RuntimeCheck(hot_k.ndim() == 3 && hot_v.ndim() == 3 && hot_k.shape()[0] == hot_v.shape()[0] &&
               hot_k.shape()[1] == 4 && hot_k.shape()[2] == 128 &&
               hot_v.shape()[1] == 4 && hot_v.shape()[2] == 128,
               "MMA TMA hot KV requires matching [rows,4,128]");
  RuntimeCheck(host_rows > 0 && host_k_pointer > 0 && host_v_pointer > 0 &&
               host_k_pointer % 16 == 0 && host_v_pointer % 16 == 0,
               "Invalid mapped MMA TMA backing");
  const DLDevice device = q.device();
  LaunchKernel(dim3(q.shape()[0], 4, Splits), kMMAThreads, device)(
      attention_kernel<TopK, Splits>, static_cast<const bf16_t*>(q.data_ptr()),
      static_cast<bf16_t*>(hot_k.data_ptr()), static_cast<bf16_t*>(hot_v.data_ptr()),
      reinterpret_cast<const bf16_t*>(host_k_pointer), reinterpret_cast<const bf16_t*>(host_v_pointer),
      host_rows, hot_k.shape()[0], static_cast<const int64_t*>(host_locs.data_ptr()),
      static_cast<const int32_t*>(device_locs.data_ptr()), static_cast<const int32_t*>(actual_lens.data_ptr()),
      static_cast<const int32_t*>(real.data_ptr()), static_cast<const int32_t*>(chunk_size.data_ptr()),
      static_cast<float*>(partial.data_ptr()), static_cast<float>(scaling), static_cast<float>(soft_cap));
  LaunchKernel(dim3(q.shape()[0], 32), 128, device)(merge_kernel<Splits>,
      static_cast<const float*>(partial.data_ptr()), static_cast<bf16_t*>(output.data_ptr()),
      static_cast<const int32_t*>(real.data_ptr()));
}
}  // namespace xm_hisparse_tma_mma
