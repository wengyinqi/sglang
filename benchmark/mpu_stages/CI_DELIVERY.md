# Linux CI offline delivery

`mpu-linux-delivery.yml` supplies a GPU-free alternative when the local WSL/Docker
host is unavailable. Changes to that workflow or `ci_native_delivery.py` trigger
one independent Ubuntu 24.04 job on each pushed `mpu-exp/*` branch. Each job checks
the exact event commit and the registered stage before building.

The job installs the native profile's system prerequisites, Rust 1.99.1, uv
0.13.5 and the NVIDIA **userspace-only** CUDA compatibility package
580.82.07-0ubuntu1. The latter's SHA256 is pinned from NVIDIA's Ubuntu 24.04 package
index. No GPU driver/kernel installation or model download is performed.
Python/Cargo inputs retain their existing locks; OS packages and the Ubuntu base
tag are resolved online and their observed identities are recorded, so this is
not a claim of byte-identical rebuilds.

The acceptance order is native wheel/extensions → portable/compile/JIT/imports →
runtime image/container smoke → default server CLI `--help` → offline export →
independent offline restore. The export has to pass its manifest checks before
the original build prefix, dependency cache and build image are removed. Restore
then uses a separate Docker data directory, an initially absent image, no network,
and only the bundle/new evidence mounts. It verifies production/wheel bytes,
loads relocated native/JIT libraries and repeats portable/Gloo acceptance.

Successful runs upload `offline-bundle-<stage>-<commit>` artifacts (one-day
retention); copy these to permanent local storage promptly. Evidence artifacts
are kept for seven days, including failures. No image is pushed to a registry.
An artifact name or a green CPU-only workflow does not prove offline delivery:
require `delivery-result.json: status=passed`, the matching commit, successful
`restored/restore.json`, and verify the downloaded bundle manifest/file hashes.

GitHub builds have their own native/image receipts and image IDs. They do not
replace or relabel older local build receipts. Neither route runs GPU kernels,
P2P transfers, DeepEP, model forward or performance benchmarks. Four-compute-card
stages remain Attention-DP4+EP4; Qwen remains one compute card; +0 uses CPU DDR.
