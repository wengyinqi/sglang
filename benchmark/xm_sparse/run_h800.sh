#!/usr/bin/env bash
# Existing offline runtime; no image rebuild or model download is required.
set -Eeuo pipefail
TASK_ROOT=${XM_TASK_ROOT:-/home/user/Workspace/mpu/xm-sparse-qwen-20261127-001}
SOURCE_ROOT=${XM_SOURCE_ROOT:-$TASK_ROOT/source}
MPU_ROOT=${XM_MPU_ROOT:-/home/user/Workspace/mpu}
IMAGE_ID=sha256:2e5dc318b02a18a16579bb6a9a25119db101108603bbe9a4074ac6e0b6090425
TASK_PYTHON=/opt/mpu/ci-stage01/runtime/bin/python
mkdir -p "$TASK_ROOT/results" "$TASK_ROOT/cache/torch_extensions" "$TASK_ROOT/cache/triton" "$TASK_ROOT/cache/jit"
test -d "$SOURCE_ROOT/python/sglang"
PROFILE_DOCKER_ARGS=()
if [[ -n "${XM_NSIGHT_ROOT:-}" ]]; then
  # Explicitly opt in only for independent, bounded profiling commands.
  test -f "$XM_NSIGHT_ROOT/nsys-bin-relative.txt"
  NSYS_RELATIVE_DIR=$(< "$XM_NSIGHT_ROOT/nsys-bin-relative.txt")
  test -x "$XM_NSIGHT_ROOT/$NSYS_RELATIVE_DIR/nsys"
  PROFILE_DOCKER_ARGS=(
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined
    --mount "type=bind,src=$XM_NSIGHT_ROOT,dst=/nsight,readonly"
    -e "PATH=/nsight/$NSYS_RELATIVE_DIR:/opt/mpu/ci-stage01/runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  )
fi
docker run --rm --pull never --gpus all --network host --ipc=host \
  "${PROFILE_DOCKER_ARGS[@]}" \
  --cpuset-cpus 0-31,64-95 --cpuset-mems 0 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e LD_LIBRARY_PATH=/opt/mpu/ci-stage01/runtime/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/mpu/ci-stage01/toolchain/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  -e CUDA_MODULE_LOADING=EAGER \
  -e CUDA_VISIBLE_DEVICES=0 -e CC=/usr/bin/gcc-13 -e CXX=/usr/bin/g++-13 \
  -e SGLANG_RUN_XM_TMA_GPU_TESTS=1 \
  -e NVCC_PREPEND_FLAGS=-ccbin=/usr/bin/g++-13 \
  -e SGLANG_XM_NVCC=/opt/mpu/ci-stage01/toolchain/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  -e SGLANG_XM_NATIVE_CACHE=/results/cache/xm-native \
  -e MAX_JOBS=8 -e TORCH_EXTENSIONS_DIR=/results/cache/torch_extensions \
  -e TRITON_CACHE_DIR=/root/.cache/triton -e SGLANG_JIT_CACHE_DIR=/results/cache/jit \
  -e PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/mpu/ci-stage01/source/python \
  --mount "type=bind,src=$SOURCE_ROOT,dst=/opt/mpu/ci-stage01/source,readonly" \
  --mount "type=bind,src=$MPU_ROOT/models/Qwen3-30B-A3B-Thinking-2507-144afc2f,dst=/models/qwen,readonly" \
  --mount "type=bind,src=$MPU_ROOT/data/longbench-qwen-20261119,dst=/data" \
  --mount "type=bind,src=$TASK_ROOT,dst=/results" \
  --mount "type=bind,src=$MPU_ROOT/cache/flashinfer-2e5dc318b02a-cu130-gcc13,dst=/root/.cache/flashinfer" \
  --mount "type=bind,src=$TASK_ROOT/cache/triton,dst=/root/.cache/triton" \
  -w /opt/mpu/ci-stage01/source --entrypoint "$TASK_PYTHON" \
  "$IMAGE_ID" "$@"
