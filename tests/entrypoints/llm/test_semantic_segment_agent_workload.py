# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "facebook/opt-125m"


def _run_until_finished(llm: LLM, request_id: str) -> list[float]:
    """Run engine steps until request finishes and return per-step latencies
    for steps that produced tokens for this request.
    """
    step_latencies: list[float] = []
    finished = False

    while not finished:
        t0 = time.perf_counter()
        outputs = llm.llm_engine.step()
        dt = time.perf_counter() - t0

        for out in outputs:
            if out.request_id != request_id:
                continue
            if out.outputs and out.outputs[0].token_ids:
                step_latencies.append(dt)
            if out.finished:
                finished = True

    return step_latencies


def _run_episode(*, consolidate: bool, tool_sleep_s: float) -> tuple[float, float, float]:
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
        enable_semantic_segment=True,
        semantic_supported_block_sizes=[16, 32, 64],
        semantic_eviction_policy="tight",
        max_num_batched_tokens=512,
        max_num_seqs=8,
    )

    try:
        req_a = "agent_turn1"
        turn1_params = SamplingParams(temperature=0.0, max_tokens=24)
        llm.llm_engine.add_request(
            request_id=req_a,
            prompt=("User asks to call tool." * 4),
            params=turn1_params,
        )

        # Step a few rounds so turn-1 request is active, then seal/consolidate.
        for _ in range(4):
            llm.llm_engine.step()

        consolidation_time_s = 0.0
        if consolidate:
            t0 = time.perf_counter()
            llm.llm_engine.seal(req_a)
            llm.llm_engine.consolidate_memory(req_a)
            consolidation_time_s = time.perf_counter() - t0

        # Drain turn-1 request.
        _run_until_finished(llm, req_a)

        # Tool call latency (assumed to hide consolidation latency).
        if tool_sleep_s > 0:
            time.sleep(tool_sleep_s)

        req_b = "agent_turn2"
        turn2_params = SamplingParams(temperature=0.0, max_tokens=8)
        llm.llm_engine.add_request(
            request_id=req_b,
            prompt=("Tool returned JSON. Summarize result." * 3),
            params=turn2_params,
        )

        step_latencies = _run_until_finished(llm, req_b)
        assert step_latencies, "No token-producing steps observed for turn-2"

        ttft_s = step_latencies[0]
        tbt_s = (
            sum(step_latencies[1:]) / len(step_latencies[1:])
            if len(step_latencies) > 1
            else ttft_s
        )
        return ttft_s, tbt_s, consolidation_time_s
    finally:
        del llm
        cleanup_dist_env_and_memory()


def test_llm_user_entry_semantic_segment_agent_ttft_tbt():
    """Use vLLM user entry (LLM) to measure TTFT/TBT under agent-like load.

    Flow: turn-1 LLM -> tool sleep -> turn-2 LLM. The test assumes tool sleep
    can cover consolidation overhead, and checks turn-2 TTFT/TBT do not
    regress materially against a baseline without consolidation.
    """

    baseline_ttft, baseline_tbt, _ = _run_episode(
        consolidate=False,
        tool_sleep_s=0.0,
    )

    # Estimate consolidation overhead first, then choose tool sleep with margin.
    _, _, estimated_consolidation = _run_episode(
        consolidate=True,
        tool_sleep_s=0.0,
    )
    tool_sleep_s = max(estimated_consolidation * 1.5, 0.01)

    agent_ttft, agent_tbt, consolidation_time_s = _run_episode(
        consolidate=True,
        tool_sleep_s=tool_sleep_s,
    )

    assert tool_sleep_s >= consolidation_time_s
    assert baseline_ttft > 0 and baseline_tbt > 0
    assert agent_ttft > 0 and agent_tbt > 0

    # Keep tolerance loose to reduce CI flakiness on shared GPUs.
    assert agent_ttft <= baseline_ttft * 2.0
    assert agent_tbt <= baseline_tbt * 2.0
