"""Execute actual scheduler/argument logic on CPU; no model or GPU claims."""

import ast
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def method(path, owner, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == owner)
    item = next(node for node in cls.body if getattr(node, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            item,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


LOGGER = SimpleNamespace(info=Mock())
CLOCK = SimpleNamespace(monotonic=Mock(return_value=10.0))
WAIT = method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "_xm_wait_for_correctness_cohort",
    dict(time=CLOCK, logger=LOGGER),
)
CHECK = method(
    "python/sglang/srt/server_args.py",
    "ServerArgs",
    "_check_xm_correctness_cohort_barrier",
    dict(json=json),
)


def scheduler(*, enabled=True, waiting=(), chunked=None, staging=False, running=()):
    return SimpleNamespace(
        server_args=SimpleNamespace(
            xm_correctness_cohort_barrier=enabled,
            xm_correctness_cohort_timeout_seconds=30.0,
        ),
        waiting_queue=[SimpleNamespace(rid=rid) for rid in waiting],
        chunked_req=chunked,
        hisparse_coordinator=SimpleNamespace(
            has_ongoing_staging=Mock(return_value=staging)
        ),
        running_batch=SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in running]
        ),
        _xm_cohort_wait_started=None,
    )


def args(**updates):
    values = dict(
        xm_correctness_cohort_barrier=True,
        xm_correctness_cohort_timeout_seconds=30.0,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        ep_size=1,
        nnodes=1,
        disaggregation_mode="null",
        disable_overlap_schedule=True,
        speculative_algorithm=None,
        dllm_algorithm=None,
        enable_mixed_chunk=False,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        enable_dp_attention=False,
        enable_prefill_delayer=False,
        device="cuda",
        hisparse_config='{"algorithm":"quest"}',
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        ),
        _check_quest_server_args=Mock(),
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_disabled_barrier_does_not_inspect_queues_or_change_default_state():
    state = SimpleNamespace(
        server_args=SimpleNamespace(xm_correctness_cohort_barrier=False)
    )
    CLOCK.monotonic.reset_mock()
    assert WAIT(state, None) is False
    CLOCK.monotonic.assert_not_called()
    CHECK(SimpleNamespace(xm_correctness_cohort_barrier=False))


@pytest.mark.parametrize(
    "pending",
    [dict(waiting=["q"]), dict(chunked=object()), dict(staging=True)],
)
def test_no_prefill_holds_ready_decode_until_each_pending_owner_drains(pending):
    state = scheduler(running=["ready"], **pending)
    CLOCK.monotonic.return_value = 10.0
    assert WAIT(state, None) is True
    assert state._xm_cohort_wait_started == 10.0
    state.waiting_queue.clear()
    state.chunked_req = None
    state.hisparse_coordinator.has_ongoing_staging.return_value = False
    assert WAIT(state, None) is False
    assert state._xm_cohort_wait_started is None


def test_prefill_and_staging_collection_remain_possible_while_decode_is_held():
    state = scheduler(waiting=["next"], staging=True, running=["first"])
    CLOCK.monotonic.return_value = 10.0
    assert WAIT(state, None) is True
    # A real prefill batch is always admitted normally and resets no-progress time.
    CLOCK.monotonic.return_value = 50.0
    assert WAIT(state, object()) is False
    assert state._xm_cohort_wait_started is None
    assert WAIT(state, None) is True
    assert state._xm_cohort_wait_started == 50.0


@pytest.mark.parametrize("running", [["one"], ["short0", "short1"], []])
def test_short_round_and_lifecycle_do_not_wait_for_the_global_batch_limit(running):
    state = scheduler(running=running)
    assert WAIT(state, None) is False


@pytest.mark.parametrize("running", [["ready"], []])
def test_unadmittable_cohort_times_out_explicitly_instead_of_partial_release(running):
    state = scheduler(waiting=["blocked"], running=running)
    CLOCK.monotonic.return_value = 10.0
    assert WAIT(state, None) is True
    CLOCK.monotonic.return_value = 40.0
    with pytest.raises(RuntimeError, match="not released for partial decode"):
        WAIT(state, None)
    assert state._xm_cohort_wait_started == 10.0


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_nonfinite_or_nonpositive_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        CHECK(args(xm_correctness_cohort_timeout_seconds=timeout))


@pytest.mark.parametrize("rank", ["tp_size", "dp_size", "pp_size", "ep_size", "nnodes"])
def test_parallel_ranks_are_rejected(rank):
    with pytest.raises(ValueError, match="requires"):
        CHECK(args(**{rank: 2}))


@pytest.mark.parametrize(
    "update",
    [
        dict(disaggregation_mode="decode"),
        dict(disable_overlap_schedule=False),
        dict(speculative_algorithm="EAGLE"),
        dict(dllm_algorithm="test"),
        dict(enable_mixed_chunk=True),
        dict(enable_two_batch_overlap=True),
        dict(enable_pdmux=True),
        dict(enable_dp_attention=True),
        dict(enable_prefill_delayer=True),
        dict(device="cpu"),
        dict(hisparse_config='{"algorithm":"dsa"}'),
        dict(hisparse_config="[]"),
        dict(
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
            )
        ),
    ],
)
def test_unsupported_scheduling_and_model_scope_is_rejected(update):
    with pytest.raises(ValueError):
        CHECK(args(**update))


def test_valid_qwen_quest_runs_the_existing_backend_guard():
    state = args()
    CHECK(state)
    state._check_quest_server_args.assert_called_once_with()


NEXT_BATCH = method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "get_next_batch_to_run",
    dict(set_schedule_time_batch=Mock()),
)


class Batch:
    def __init__(self, reqs=()):
        self.reqs = list(reqs)
        self.is_prefill_only = False
        self.batch_is_full = False

    def is_empty(self):
        return not self.reqs

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


def scheduler_loop(*, enabled, waiting=(), running=()):
    state = scheduler(enabled=enabled, waiting=waiting, running=running)
    state.running_batch = Batch(state.running_batch.reqs)
    state._abort_on_waiting_timeout = Mock()
    state._abort_on_running_timeout = Mock()
    state.process_pending_chunked_abort = Mock()
    state.dllm_config = None
    state.enable_hisparse = True
    state.last_batch = None
    state.require_mlp_sync = False
    state._build_hisparse_decode_batch = Batch
    state.get_new_batch_prefill = Mock(return_value=None)
    state.maybe_prepare_mlp_sync_batch = lambda value, **kwargs: value
    state._maybe_prepare_ngram_embedding = lambda value: value
    state._xm_wait_for_correctness_cohort = MethodType(WAIT, state)
    state.update_running_batch = Mock(side_effect=lambda value: value)
    state.hisparse_coordinator.collect_ready_reqs = Mock(return_value=[])
    return state


def test_scheduler_collects_three_then_one_without_advancing_the_first_three():
    state = scheduler_loop(enabled=True)
    ready = [SimpleNamespace(rid=f"request-{i}") for i in range(4)]
    state.hisparse_coordinator.collect_ready_reqs.side_effect = [ready[:3], ready[3:]]
    state.hisparse_coordinator.has_ongoing_staging.side_effect = [True, False]
    CLOCK.monotonic.return_value = 10.0
    assert NEXT_BATCH(state) is None
    assert state.running_batch.reqs == ready[:3]
    state.update_running_batch.assert_not_called()
    assert NEXT_BATCH(state).reqs == ready
    state.update_running_batch.assert_called_once_with(state.running_batch)
    assert state._xm_cohort_wait_started is None


def test_default_scheduler_can_decode_the_ready_prefix_as_before():
    state = scheduler_loop(enabled=False)
    ready = [SimpleNamespace(rid=f"request-{i}") for i in range(3)]
    state.hisparse_coordinator.collect_ready_reqs.return_value = ready
    state.hisparse_coordinator.has_ongoing_staging.return_value = True
    assert NEXT_BATCH(state).reqs == ready
    state.update_running_batch.assert_called_once()
    state.hisparse_coordinator.has_ongoing_staging.assert_not_called()


def test_scheduler_with_no_ready_request_still_times_out_an_unadmittable_queue():
    state = scheduler_loop(enabled=True, waiting=["blocked"])
    CLOCK.monotonic.return_value = 10.0
    assert NEXT_BATCH(state) is None
    state.update_running_batch.assert_not_called()
    CLOCK.monotonic.return_value = 40.0
    with pytest.raises(RuntimeError, match="waiting_rids=\\['blocked'\\]"):
        NEXT_BATCH(state)
    state.update_running_batch.assert_not_called()


def test_scheduler_prefill_progress_is_not_blocked_by_an_existing_ready_prefix():
    state = scheduler_loop(enabled=True, waiting=["prefill"], running=["ready"])
    prefill = object()
    state.get_new_batch_prefill.return_value = prefill
    state._xm_cohort_wait_started = 1.0
    CLOCK.monotonic.return_value = 60.0
    assert NEXT_BATCH(state) is prefill
    assert state._xm_cohort_wait_started is None
    state.update_running_batch.assert_not_called()


@pytest.mark.parametrize("coordinator_attribute", [True, False])
def test_hbm_quest_merges_prefill_normally_but_holds_decode_for_queued_peer(
    coordinator_attribute,
):
    state = scheduler_loop(enabled=True, waiting=["next"])
    state.enable_hisparse = False
    state.hisparse_coordinator = None
    if not coordinator_attribute:
        del state.hisparse_coordinator
    ready = SimpleNamespace(rid="ready")
    prefill = Batch([ready])
    prefill.forward_mode = SimpleNamespace(is_extend=lambda: True)
    prefill.chunked_req = None
    prefill.batch_size = lambda: len(prefill.reqs)
    prefill.filter_batch = Mock()
    state.last_batch = prefill
    CLOCK.monotonic.return_value = 10.0
    assert NEXT_BATCH(state) is None
    assert state.running_batch is prefill
    prefill.filter_batch.assert_called_once_with(chunked_req_to_exclude=[])
    state.update_running_batch.assert_not_called()
    state.last_batch = None
    state.waiting_queue.clear()
    assert NEXT_BATCH(state).reqs == [ready]
    state.update_running_batch.assert_called_once_with(prefill)
    assert state._xm_cohort_wait_started is None


IS_IDLE = method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "is_fully_idle",
    dict(DisaggregationMode=SimpleNamespace(PREFILL="prefill", DECODE="decode")),
)
ON_IDLE = method(
    "python/sglang/srt/managers/scheduler_runtime_checker_mixin.py",
    "SchedulerRuntimeCheckerMixin",
    "on_idle",
    {},
)


@pytest.mark.parametrize(
    "pending",
    [dict(running=["ready"]), dict(waiting=["blocked"]), dict(staging=True)],
)
def test_cohort_wait_is_not_idle_and_does_not_run_destructive_housekeeping(pending):
    state = scheduler_loop(
        enabled=True, **{k: v for k, v in pending.items() if k != "staging"}
    )
    state.hisparse_coordinator.has_ongoing_staging.return_value = pending.get(
        "staging", False
    )
    state.chunked_req = None
    state.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
    state.cur_batch = None
    state.enable_overlap = False
    state.pp_size = 1
    state.grammar_manager = SimpleNamespace(grammar_queue=[])
    state.disaggregation_mode = "null"
    state.enable_hierarchical_cache = False
    state.is_fully_idle = MethodType(IS_IDLE, state)
    state._xm_cohort_wait_started = 10.0
    state._check_all_pools = Mock()
    state._check_req_pool = Mock()
    state._check_tree_cache = Mock()
    state.maybe_sleep_on_idle = Mock()
    assert state.is_fully_idle() is False
    ON_IDLE(state)
    assert state._xm_cohort_wait_started == 10.0
    state._check_all_pools.assert_not_called()
    state._check_req_pool.assert_not_called()
    state._check_tree_cache.assert_not_called()
    state.maybe_sleep_on_idle.assert_not_called()


ADMISSION = method(
    "python/sglang/srt/managers/schedule_policy.py",
    "PrefillAdder",
    "add_one_req_ignore_eos",
    dict(
        IGNORE_EOS_RESERVE_TOKENS=1,
        AddReqResult=SimpleNamespace(NO_TOKEN="no-token", OTHER="other"),
    ),
)


@pytest.mark.parametrize("length", [16384, 32768])
def test_exact_payload_capacity_does_not_include_ignore_eos_admission_reserve(length):
    # Real recipe page_size=64. Four requests x128 outputs use two output pages each.
    page, output, batch = 64, 128, 4
    exact = batch * (length + output)

    def request(*, generated):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=output, ignore_eos=True),
            output_ids=[0] * generated,
            origin_input_ids=[0] * length,
            extend_input_len=length,
            fill_ids=[0] * length,
            set_extend_input_len=lambda value: None,
        )

    def admit(capacity):
        available = capacity - 3 * length
        state = SimpleNamespace(
            ceil_paged_tokens=lambda n: (n + page - 1) // page * page,
            cur_rem_tokens=available,
            rem_total_tokens=available - 3 * (output - 1) * 0.7,
            is_hybrid_swa=False,
            req_states=None,
            new_token_ratio=0.7,
            running_batch=SimpleNamespace(
                reqs=[request(generated=1) for _ in range(3)]
            ),
            can_run_list=[],
            dllm_config=None,
            rem_chunk_tokens=4096,
            _update_prefill_budget=Mock(),
            budget_state=lambda: "continue",
        )
        return ADMISSION(state, request(generated=0))

    assert admit(exact) == "no-token"
    # This is a conservative experimental padding, not a new admission policy.
    assert admit(exact + 4 * batch * page) == "continue"
