"""Production definition loaders and CPU metadata fixtures for HiSparse control tests.

CUDA streams and model state are boundaries, not simulated GPU acceptance.
On failure the seed and full operation log are included for exact replay.
"""

import ast
import hashlib
import logging
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[2]
PAGED = (ROOT / "python/sglang/srt/mem_cache/mpu_peer.py").exists()
SOURCES = {}


def definition(path, name, namespace=None, owner=None):
    source = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(source).hexdigest()
    tree = ast.parse(source.decode("utf-8"))
    body = (
        tree.body
        if owner is None
        else next(n.body for n in tree.body if getattr(n, "name", None) == owner)
    )
    node = next(n for n in body if getattr(n, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    scope = dict(
        __name__=__name__,
        torch=torch,
        threading=threading,
        logger=logging.getLogger(__name__),
    )
    scope.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), path, "exec"), scope)
    return scope[name]


def make_pool():
    path = "python/sglang/srt/mem_cache/" + (
        "mpu_peer.py" if PAGED else "mpu_memory_pool.py"
    )
    namespace = {} if PAGED else {"_positive_int": definition(path, "_positive_int")}
    cls = definition(path, "MPUSlotAllocator", namespace)
    if not PAGED:
        return cls(512)
    mixin = definition(
        "python/sglang/srt/mem_cache/pool_host/hisparse.py", "HiSparseHostPoolMixin"
    )
    pool = type("MetadataPool", (cls, mixin), {})(512, 64)
    return pool


def make_logical_allocator():
    base_init = definition(
        "python/sglang/srt/mem_cache/allocator/base.py",
        "__init__",
        {"abc": NS(abstractmethod=lambda f: f)},
        "BaseTokenToKVPoolAllocator",
    )
    base = type("Base", (), {"__init__": base_init})
    path = "python/sglang/srt/mem_cache/allocator/paged.py"
    get_pages = definition("python/sglang/srt/utils/common.py", "get_num_new_pages")
    paged = definition(
        path,
        "PagedTokenToKVPoolAllocator",
        {
            "BaseTokenToKVPoolAllocator": base,
            "get_bool_env_var": lambda _: True,
            "current_platform": NS(capabilities=NS(supports_triton=False)),
            "_is_hip": False,
            "get_num_new_pages": get_pages,
            "alloc_extend_naive": definition(path, "alloc_extend_naive"),
            "alloc_decode_naive": definition(path, "alloc_decode_naive"),
        },
    )
    pool_type = type(
        "Pool",
        (),
        {
            "register_mapping": lambda self, mapping: setattr(self, "mapping", mapping),
            "_translate_loc_to_hisparse_device": lambda self, locations: self.mapping[
                locations
            ],
        },
    )
    allocator = definition(
        "python/sglang/srt/mem_cache/allocator/hisparse.py",
        "HiSparseTokenToKVPoolAllocator",
        {
            "weakref": weakref,
            "BaseTokenToKVPoolAllocator": base,
            "HybridLinearKVPool": type("Hybrid", (), {}),
            "HiSparseDSATokenToKVPool": pool_type,
            "current_platform": NS(get_paged_allocator_cls=lambda: None),
            "PagedTokenToKVPoolAllocator": paged,
            "get_num_new_pages": get_pages,
        },
    )
    return allocator(128, 64, torch.bfloat16, "cpu", pool_type(), True, 4)


def coordinator(pool, mapping, allocated, audit):
    path = "python/sglang/srt/managers/hisparse_coordinator.py"
    methods = ["request_finished", "retract_req"]
    if "def release_backing(" in (ROOT / path).read_text(encoding="utf-8"):
        methods.append("release_backing")
    namespace = {
        "device_module": NS(
            current_stream=lambda: NS(
                wait_stream=lambda _: audit.append("wait_producer")
            )
        ),
        "MODE_QUEST": "quest",
    }
    cls = type(
        "Release",
        (),
        {
            name: definition(path, name, namespace, "HiSparseCoordinator")
            for name in methods
        },
    )
    c = cls()
    c.backing_budget = None  # This state machine exercises allocated slot ownership.
    c.decode_producer_stream = object()
    c.wait_for_pending_backup = lambda: audit.append("wait_backup")
    c._has_pending_backup = True
    c._backup_done_event = NS(wait=lambda _: audit.append("wait_backup"))
    c.mode, c._pending_bounds_events = "quest", {}
    c.quest_algorithm = NS(
        invalidate_request=lambda _: audit.append("invalidate_quest")
    )
    c.release_mpu_request = lambda _: audit.append("release_reservation")
    c.mem_pool_host = pool
    c.req_to_host_pool, c.req_to_host_pool_allocated_len = mapping, allocated
    c.req_to_token_pool = NS(
        req_to_token=torch.arange(mapping.numel()).reshape_as(mapping)
    )
    c.token_to_kv_pool_allocator = NS(
        free_hisparse_indices=lambda _: audit.append("free_hot"),
        full_to_hisparse_device_index_mapping=torch.ones(
            mapping.numel(), dtype=torch.int64
        ),
    )
    c.mem_pool_device = NS(
        translate_loc_from_full_to_compressed=lambda x: x,
        full_to_hisparse_device_index_mapping=torch.ones(
            mapping.numel(), dtype=torch.int64
        ),
    )
    c.spec_swap = NS(
        free_unrotated_scratch=lambda _: None,
        clear_scratch=lambda _: None,
        extend_owned_locs=lambda _, x: x,
    )
    c.req_device_buffer_size = torch.zeros(mapping.shape[0], dtype=torch.int64)
    c.req_to_device_buffer = torch.zeros((mapping.shape[0], 64), dtype=torch.int64)
    c.req_device_buffer_tokens = torch.full(
        (1, mapping.shape[0], 64), -1, dtype=torch.int32
    )
    c.req_device_buffer_token_locs = c.req_device_buffer_tokens.clone()
    c.lru_slots = torch.zeros_like(c.req_device_buffer_tokens, dtype=torch.int16)
    c._lru_init = torch.arange(64, dtype=torch.int16)
    c._skip_first_backup = [False] * mapping.shape[0]
    free = pool.free

    def traced_free(slots):
        audit.append("free_backing")
        return free(slots)

    pool.free = traced_free
    return c
