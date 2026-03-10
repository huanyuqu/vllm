# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from contextlib import nullcontext
from typing import NamedTuple

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from tests.utils import create_new_process_for_each_test

MODEL_NAME = "facebook/opt-125m"
REQ_A_ID = "req_a"
REQ_B_ID = "req_b"
SEGMENT_TOKENS = 64
REQ_B_MIN_TOKENS = 32
NUM_GPU_BLOCKS = 16
AGENT_SEGMENT_TOKENS = 256
# OPT-125m has max context length 2048 (prompt + generated tokens),
# so 256x8 generated tokens is not feasible with a non-empty prompt.
AGENT_NUM_SEGMENTS = 7
AGENT_SEGMENT_CONFIGS = [
    (AGENT_SEGMENT_TOKENS, AGENT_NUM_SEGMENTS),
]
PROFILE_WAIT_STEPS = 50
PROFILE_WARMUP_STEPS = 20
PROFILE_ACTIVE_STEPS = 300
PROFILE_ROW_LIMIT = 20
ENABLE_TORCH_PROFILER = os.environ.get("VLLM_ENABLE_TORCH_PROFILER", "1") == "1"
ATTR_SCOPES = [
    "schedule: allocate_slots",
    "schedule: update_after_schedule",
    "gpu_model_runner: preprocess",
    "gpu_model_runner: semantic_segment_metadata",
    "gpu_model_runner: forward",
    "gpu_model_runner: postprocess",
    "flash_attn: segmented_metadata_pack",
    "flash_attn: segmented_kernel",
]

PROMPTS = {
    REQ_A_ID: "User asks for a tool call and then wants the answer to continue.",
    REQ_B_ID: "A second deterministic request keeps pressure on the KV cache.",
}


class EpisodeMetrics(NamedTuple):
    tbt_s: float
    total_latency_s: float
    execute_memory_ops_time_s: float
    episode_wall_s: float
    profile_cpu_table: str
    profile_cuda_table: str
    scope_cpu_ms: dict[str, float]
    scope_cuda_ms: dict[str, float]


def _build_llm(memory_management: bool) -> LLM:
    llm_kwargs = dict(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        num_gpu_blocks_override=NUM_GPU_BLOCKS,
        max_num_batched_tokens=192,
        max_num_seqs=16,
        enforce_eager=True,
        dtype="float16",
        seed=42,
    )
    if memory_management:
        llm_kwargs.update(
            enable_semantic_segment_memory_management=True,
            enable_semantic_segment_kernel=False,
            semantic_supported_block_sizes=[16, 32, 64],
            semantic_eviction_policy="tight",
        )
    return LLM(**llm_kwargs)


def _sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
    )


def _assert_nontrivial_token_ids(token_ids: list[int], expected_len: int) -> None:
    assert len(token_ids) == expected_len
    assert token_ids
    assert any(token_id != token_ids[0] for token_id in token_ids[1:])


def _build_profiler():
    if not ENABLE_TORCH_PROFILER:
        return None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=PROFILE_WAIT_STEPS,
            warmup=PROFILE_WARMUP_STEPS,
            active=PROFILE_ACTIVE_STEPS,
            repeat=1,
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def _extract_profile_tables(prof, has_cuda: bool) -> tuple[str, str]:
    if prof is None:
        return "", ""
    try:
        key_avg = prof.key_averages()
        cpu_table = key_avg.table(
            sort_by="self_cpu_time_total", row_limit=PROFILE_ROW_LIMIT
        )
        cuda_table = ""
        if has_cuda:
            cuda_table = key_avg.table(
                sort_by="self_cuda_time_total", row_limit=PROFILE_ROW_LIMIT
            )
        return cpu_table, cuda_table
    except Exception as exc:
        return f"<failed to collect cpu profile table: {exc}>", (
            f"<failed to collect cuda profile table: {exc}>" if has_cuda else ""
        )


def _extract_profile_event_total_ms(prof, event_name: str) -> tuple[float, float]:
    if prof is None:
        return 0.0, 0.0
    cpu_total_us = 0.0
    cuda_total_us = 0.0
    try:
        for item in prof.key_averages():
            key = getattr(item, "key", None)
            if key == event_name:
                cpu_total_us += float(getattr(item, "cpu_time_total", 0.0))
                cuda_total_us += float(getattr(item, "cuda_time_total", 0.0))
    except Exception:
        return 0.0, 0.0
    return cpu_total_us / 1000.0, cuda_total_us / 1000.0


def _collect_scope_totals(
    prof, scope_names: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    cpu_totals_ms: dict[str, float] = {}
    cuda_totals_ms: dict[str, float] = {}
    for scope in scope_names:
        cpu_ms, cuda_ms = _extract_profile_event_total_ms(prof, scope)
        cpu_totals_ms[scope] = cpu_ms
        cuda_totals_ms[scope] = cuda_ms
    return cpu_totals_ms, cuda_totals_ms


def _run_basic_seal_episode(memory_management: bool) -> list[int]:
    llm = _build_llm(memory_management)
    token_ids: list[int] = []
    sealed = False

    try:
        llm.llm_engine.add_request(
            request_id=REQ_A_ID,
            prompt=PROMPTS[REQ_A_ID],
            params=_sampling_params(SEGMENT_TOKENS + 16),
        )

        while llm.llm_engine.has_unfinished_requests():
            outputs = llm.llm_engine.step()
            for out in outputs:
                if out.request_id == REQ_A_ID and out.outputs:
                    token_ids = list(out.outputs[0].token_ids)

            if memory_management and not sealed and len(token_ids) >= SEGMENT_TOKENS:
                llm.seal_segment(REQ_A_ID)
                sealed = True

        if memory_management:
            assert sealed

        return token_ids
    finally:
        del llm
        cleanup_dist_env_and_memory()


def _run_consolidation_episode(
    memory_management: bool,
    release_b: bool,
) -> tuple[list[int], dict[str, int] | None]:
    llm = _build_llm(memory_management)
    token_ids = {REQ_A_ID: [], REQ_B_ID: []}
    sealed = False
    consolidated = False
    op_summary: dict[str, int] | None = None

    try:
        llm.llm_engine.add_request(
            request_id=REQ_A_ID,
            prompt=PROMPTS[REQ_A_ID],
            params=_sampling_params(SEGMENT_TOKENS + 32),
        )
        llm.llm_engine.add_request(
            request_id=REQ_B_ID,
            prompt=PROMPTS[REQ_B_ID],
            params=_sampling_params(SEGMENT_TOKENS + 32),
        )

        while llm.llm_engine.has_unfinished_requests():
            outputs = llm.llm_engine.step()
            for out in outputs:
                if out.request_id in token_ids and out.outputs:
                    token_ids[out.request_id] = list(out.outputs[0].token_ids)

            if (
                not sealed
                and len(token_ids[REQ_A_ID]) >= SEGMENT_TOKENS
                and len(token_ids[REQ_B_ID]) >= REQ_B_MIN_TOKENS
            ):
                if memory_management:
                    llm.seal_segment(REQ_A_ID)
                sealed = True
                if release_b:
                    llm.llm_engine.abort_request([REQ_B_ID])

            if memory_management and sealed and not consolidated:
                llm.consolidate_memory(REQ_A_ID)
                op_summary = llm.execute_semantic_memory_ops()
                consolidated = True

        if memory_management:
            assert sealed
            assert consolidated
            assert op_summary is not None

        return token_ids[REQ_A_ID], op_summary
    finally:
        del llm
        cleanup_dist_env_and_memory()


def _run_agent_workload_episode(*,
                                use_semantic_segment: bool,
                                segment_tokens: int,
                                num_segments: int,
                                progress_label: str) -> EpisodeMetrics:
    if segment_tokens <= 0 or num_segments <= 1:
        raise ValueError("segment_tokens>0 and num_segments>1 are required")

    llm_kwargs = dict(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
        max_num_batched_tokens=512,
        max_num_seqs=8,
    )
    if use_semantic_segment:
        llm_kwargs.update(
            enable_semantic_segment=True,
            semantic_supported_block_sizes=[16, 32, 64, 128],
            semantic_eviction_policy="tight",
        )

    llm = LLM(**llm_kwargs)

    try:
        req_a = "agent_single_req"
        prompt_text = "User asks to call tool." * 4
        total_tokens = segment_tokens * num_segments
        prompt_token_count = len(llm.get_tokenizer().encode(prompt_text))
        engine = llm.llm_engine
        model_config = getattr(engine, "model_config", None)
        if model_config is None and hasattr(engine, "get_model_config"):
            model_config = engine.get_model_config()
        if model_config is None or not hasattr(model_config, "max_model_len"):
            raise RuntimeError(
                "Unable to read model max context length from llm_engine"
            )
        model_max_len = model_config.max_model_len
        max_new_tokens = model_max_len - prompt_token_count
        if total_tokens > max_new_tokens:
            raise ValueError(
                f"Requested {total_tokens} generated tokens exceeds model budget "
                f"for this prompt: max_new_tokens={max_new_tokens} "
                f"(max_model_len={model_max_len}, prompt_tokens={prompt_token_count})."
            )

        turn1_params = SamplingParams(
            temperature=0.0,
            max_tokens=total_tokens,
            min_tokens=total_tokens,
            ignore_eos=True,
        )
        llm.llm_engine.add_request(
            request_id=req_a,
            prompt=prompt_text,
            params=turn1_params,
        )

        finished = False
        step_count = 0
        token_steps = 0
        prev_total_token_count = 0
        generated_time_s = 0.0
        generated_token_count = 0
        execute_memory_ops_time_s = 0.0
        next_boundary = segment_tokens
        current_segment = 1
        has_cuda = torch.cuda.is_available()
        prof = _build_profiler()
        worker_profiler_started = False

        loop_ctx = prof if prof is not None else nullcontext()
        episode_start = time.perf_counter()

        if ENABLE_TORCH_PROFILER:
            try:
                llm.start_profile()
                worker_profiler_started = True
                print(f"[{progress_label}] worker profiler started", flush=True)
            except Exception as exc:
                print(
                    f"[{progress_label}] worker profiler not started: {exc}",
                    flush=True,
                )

        with loop_ctx:
            while not finished:
                step_count += 1
                t0 = time.perf_counter()
                outputs = llm.llm_engine.step()
                dt = time.perf_counter() - t0

                if prof is not None:
                    prof.step()

                if step_count % 50 == 0:
                    print(
                        f"[{progress_label}] step={step_count} "
                        f"tokens={token_steps}/{total_tokens} "
                        f"segment={current_segment}/{num_segments}",
                        flush=True,
                    )

                for out in outputs:
                    if out.request_id != req_a:
                        continue

                    if out.outputs and out.outputs[0].token_ids:
                        current_total_token_count = len(out.outputs[0].token_ids)
                        new_tokens = current_total_token_count - prev_total_token_count
                        if new_tokens > 0:
                            counted_new_tokens = min(
                                new_tokens, total_tokens - token_steps
                            )
                            token_steps += counted_new_tokens
                            prev_total_token_count = current_total_token_count
                            generated_time_s += dt
                            generated_token_count += counted_new_tokens

                        while (
                            token_steps >= next_boundary
                            and current_segment < num_segments
                        ):
                            if use_semantic_segment:
                                print(
                                    f"[{progress_label}] consolidate at "
                                    f"tokens={token_steps}, segment={current_segment}",
                                    flush=True,
                                )
                                t1 = time.perf_counter()
                                llm.seal(req_a)
                                execute_memory_ops_time_s += (
                                    time.perf_counter() - t1
                                )
                            current_segment += 1
                            next_boundary += segment_tokens

                    if out.finished:
                        finished = True

                if token_steps >= total_tokens:
                    finished = True

        if worker_profiler_started:
            llm.stop_profile()
            print(
                f"[{progress_label}] worker profiler stopped; "
                "see VLLM_TORCH_PROFILER_DIR/profiler_out_0.txt",
                flush=True,
            )

        episode_wall_s = time.perf_counter() - episode_start

        assert token_steps == total_tokens, (
            f"Expected exactly {total_tokens} generated tokens, got {token_steps}"
        )
        assert current_segment == num_segments, (
            f"Expected to reach {num_segments} segments, got {current_segment}"
        )
        assert generated_token_count == total_tokens, (
            "Expected measured generated tokens to equal total tokens, got "
            f"{generated_token_count} vs {total_tokens}"
        )
        assert generated_time_s > 0, "No generated-token latency observed"

        tbt_s = generated_time_s / generated_token_count
        total_latency_s = generated_time_s
        print(
            f"[{progress_label}] finished: steps={step_count}, "
            f"tokens={token_steps}, segments={current_segment}, "
            f"post_total={total_latency_s:.6f}s, "
            f"mem_ops={execute_memory_ops_time_s:.6f}s, "
            f"episode_wall={episode_wall_s:.6f}s",
            flush=True,
        )
        profile_cpu_table, profile_cuda_table = _extract_profile_tables(
            prof, has_cuda
        )
        scope_cpu_ms, scope_cuda_ms = _collect_scope_totals(prof, ATTR_SCOPES)
        return EpisodeMetrics(
            tbt_s=tbt_s,
            total_latency_s=total_latency_s,
            execute_memory_ops_time_s=execute_memory_ops_time_s,
            episode_wall_s=episode_wall_s,
            profile_cpu_table=profile_cpu_table,
            profile_cuda_table=profile_cuda_table,
            scope_cpu_ms=scope_cpu_ms,
            scope_cuda_ms=scope_cuda_ms,
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@create_new_process_for_each_test(method="spawn")
def test_llm_entrypoint_memory_management_only_basic_parity():
    mm_only_token_ids = _run_basic_seal_episode(memory_management=True)
    paged_token_ids = _run_basic_seal_episode(memory_management=False)

    _assert_nontrivial_token_ids(mm_only_token_ids, SEGMENT_TOKENS + 16)
    _assert_nontrivial_token_ids(paged_token_ids, SEGMENT_TOKENS + 16)
    assert mm_only_token_ids == paged_token_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("release_b", [False, True])
@create_new_process_for_each_test(method="spawn")
def test_llm_entrypoint_memory_management_only_consolidation_parity(
    release_b: bool,
):
    mm_only_token_ids, op_summary = _run_consolidation_episode(
        memory_management=True,
        release_b=release_b,
    )
    paged_token_ids, _ = _run_consolidation_episode(
        memory_management=False,
        release_b=release_b,
    )

    assert op_summary is not None
    if release_b:
        assert op_summary["num_moves"] > 0
    else:
        assert op_summary["num_swaps"] > 0

    _assert_nontrivial_token_ids(mm_only_token_ids, SEGMENT_TOKENS + 32)
    _assert_nontrivial_token_ids(paged_token_ids, SEGMENT_TOKENS + 32)
    assert mm_only_token_ids == paged_token_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "segment_tokens,num_segments",
    AGENT_SEGMENT_CONFIGS,
    ids=lambda v: str(v),
)
@create_new_process_for_each_test(method="spawn")
def test_llm_user_entry_semantic_segment_agent_ttft_tbt(
    segment_tokens: int,
    num_segments: int,
):
    """Single-request benchmark around seal + consolidate phase.

    Flow: one request inference -> seal -> tool phase (only consolidate,
    no actual tool execution) -> continue same request inference. Compare the
    post-checkpoint latencies against a baseline run without semantic segment.
    """

    profile_mode = os.environ.get("VLLM_PROFILE_MODE", "both").strip().lower()
    if profile_mode not in {"baseline", "semantic", "both"}:
        raise ValueError(
            "VLLM_PROFILE_MODE must be one of: baseline, semantic, both"
        )

    baseline_metrics = None
    semantic_metrics = None

    if profile_mode in {"baseline", "both"}:
        baseline_metrics = _run_agent_workload_episode(
            use_semantic_segment=False,
            segment_tokens=segment_tokens,
            num_segments=num_segments,
            progress_label="baseline",
        )

    if profile_mode in {"semantic", "both"}:
        semantic_metrics = _run_agent_workload_episode(
            use_semantic_segment=True,
            segment_tokens=segment_tokens,
            num_segments=num_segments,
            progress_label="semantic",
        )

    if profile_mode == "baseline":
        assert baseline_metrics is not None
        print(
            f"Baseline TBT: {baseline_metrics.tbt_s:.6f}s, "
            f"Generated: {baseline_metrics.total_latency_s:.6f}s, "
            f"Wall: {baseline_metrics.episode_wall_s:.6f}s"
        )
        print("\n===== PROFILER CPU TOP OPS: BASELINE =====")
        print(baseline_metrics.profile_cpu_table)
        print("\n===== ATTRIBUTION SCOPES (BASELINE, ms) =====")
        for scope in ATTR_SCOPES:
            print(
                f"{scope}: cpu={baseline_metrics.scope_cpu_ms[scope]:.3f}, "
                f"cuda={baseline_metrics.scope_cuda_ms[scope]:.3f}"
            )
        if baseline_metrics.profile_cuda_table:
            print("\n===== PROFILER CUDA TOP OPS: BASELINE =====")
            print(baseline_metrics.profile_cuda_table)
        return

    if profile_mode == "semantic":
        assert semantic_metrics is not None
        print(
            f"Semantic TBT: {semantic_metrics.tbt_s:.6f}s, "
            f"Generated: {semantic_metrics.total_latency_s:.6f}s, "
            f"Wall: {semantic_metrics.episode_wall_s:.6f}s, "
            f"MemoryOps: {semantic_metrics.execute_memory_ops_time_s:.6f}s"
        )
        print("\n===== PROFILER CPU TOP OPS: SEMANTIC =====")
        print(semantic_metrics.profile_cpu_table)
        print("\n===== ATTRIBUTION SCOPES (SEMANTIC, ms) =====")
        for scope in ATTR_SCOPES:
            print(
                f"{scope}: cpu={semantic_metrics.scope_cpu_ms[scope]:.3f}, "
                f"cuda={semantic_metrics.scope_cuda_ms[scope]:.3f}"
            )
        if semantic_metrics.profile_cuda_table:
            print("\n===== PROFILER CUDA TOP OPS: SEMANTIC =====")
            print(semantic_metrics.profile_cuda_table)
        return

    assert baseline_metrics is not None and semantic_metrics is not None

    assert baseline_metrics.tbt_s > 0 and baseline_metrics.total_latency_s > 0
    assert semantic_metrics.tbt_s > 0 and semantic_metrics.total_latency_s > 0

    print(
        f"Baseline TBT: {baseline_metrics.tbt_s:.6f}s, "
        f"Generated: {baseline_metrics.total_latency_s:.6f}s, "
        f"Wall: {baseline_metrics.episode_wall_s:.6f}s"
    )
    print(
        f"Semantic TBT: {semantic_metrics.tbt_s:.6f}s, "
        f"Generated: {semantic_metrics.total_latency_s:.6f}s, "
        f"Wall: {semantic_metrics.episode_wall_s:.6f}s, "
        f"MemoryOps: {semantic_metrics.execute_memory_ops_time_s:.6f}s"
    )

    print("\n===== PROFILER CPU TOP OPS: BASELINE =====")
    print(baseline_metrics.profile_cpu_table)
    print("\n===== PROFILER CPU TOP OPS: SEMANTIC =====")
    print(semantic_metrics.profile_cpu_table)

    if semantic_metrics.profile_cuda_table:
        print("\n===== PROFILER CUDA TOP OPS: BASELINE =====")
        print(baseline_metrics.profile_cuda_table)
        print("\n===== PROFILER CUDA TOP OPS: SEMANTIC =====")
        print(semantic_metrics.profile_cuda_table)

    print("\n===== ATTRIBUTION SCOPES DELTA (SEMANTIC - BASELINE, ms) =====")
    all_scope_zero = True
    for scope in ATTR_SCOPES:
        cpu_delta = (
            semantic_metrics.scope_cpu_ms[scope]
            - baseline_metrics.scope_cpu_ms[scope]
        )
        cuda_delta = (
            semantic_metrics.scope_cuda_ms[scope]
            - baseline_metrics.scope_cuda_ms[scope]
        )
        if abs(cpu_delta) > 1e-6 or abs(cuda_delta) > 1e-6:
            all_scope_zero = False
        print(f"{scope}: cpu_delta={cpu_delta:.3f}, cuda_delta={cuda_delta:.3f}")

    if all_scope_zero:
        print(
            "[attribution] all custom-scope deltas are zero in this process. "
            "Likely profiling only the client process. "
            "Use worker profiling with: "
            "VLLM_TORCH_PROFILER_DIR=<dir> "
            "VLLM_CUSTOM_SCOPES_FOR_PROFILING=1"
        )

    print("\n===== DELTA SUMMARY =====")
    print(
        f"Generated delta (semantic_no_mem_ops-baseline): "
        f"{semantic_metrics.total_latency_s - baseline_metrics.total_latency_s:.6f}s"
    )
    print(
        f"MemoryOps delta (semantic-baseline): "
        f"{semantic_metrics.execute_memory_ops_time_s - baseline_metrics.execute_memory_ops_time_s:.6f}s"
    )
    print(
        f"TBT delta (semantic-baseline): "
        f"{semantic_metrics.tbt_s - baseline_metrics.tbt_s:.6f}s/token"
    )

    # Under the assumption that tool latency can hide consolidation overhead,
    # semantic segment mode should provide a better latency signal while
    # keeping post-checkpoint total latency within a loose jitter band.
    # assert agent_total_s <= baseline_total_s
    # assert agent_tbt <= baseline_tbt
