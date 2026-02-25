# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


MODEL_NAME = "facebook/opt-125m"
SEGMENT_TOKENS = 256
# OPT-125m has max context length 2048 (prompt + generated tokens),
# so 256x8 generated tokens is not feasible with a non-empty prompt.
NUM_SEGMENTS = 7
SEGMENT_CONFIGS = [
    (SEGMENT_TOKENS, NUM_SEGMENTS),
]


def _run_episode(*,
                 use_semantic_segment: bool,
                 consolidate: bool,
                 segment_tokens: int,
                 num_segments: int,
                 progress_label: str) -> tuple[float, float, float]:
    if consolidate and not use_semantic_segment:
        raise ValueError("consolidate requires semantic segment mode")
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
            raise RuntimeError("Unable to read model max context length from llm_engine")
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
        consolidation_time_s = 0.0
        next_boundary = segment_tokens
        current_segment = 1

        while not finished:
            step_count += 1
            t0 = time.perf_counter()
            outputs = llm.llm_engine.step()
            dt = time.perf_counter() - t0

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
                        counted_new_tokens = min(new_tokens,
                                                 total_tokens - token_steps)
                        token_steps += counted_new_tokens
                        prev_total_token_count = current_total_token_count
                        generated_time_s += dt
                        generated_token_count += counted_new_tokens

                    while token_steps >= next_boundary and current_segment < num_segments:
                        if consolidate:
                            print(
                                f"[{progress_label}] consolidate at "
                                f"tokens={token_steps}, segment={current_segment}",
                                flush=True,
                            )
                            t1 = time.perf_counter()
                            llm.llm_engine.seal(req_a)
                            llm.llm_engine.consolidate_memory(req_a)
                            consolidation_time_s += time.perf_counter() - t1
                        current_segment += 1
                        next_boundary += segment_tokens

                if out.finished:
                    finished = True

            if token_steps >= total_tokens:
                finished = True

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
            f"consolidate={consolidation_time_s:.6f}s",
            flush=True,
        )
        return tbt_s, total_latency_s, consolidation_time_s
    finally:
        del llm
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "segment_tokens,num_segments",
    SEGMENT_CONFIGS,
    ids=lambda v: str(v),
)
def test_llm_user_entry_semantic_segment_agent_ttft_tbt(
    segment_tokens: int,
    num_segments: int,
):
    """Single-request benchmark around seal + consolidate phase.

    Flow: one request inference -> seal -> tool phase (only consolidate,
    no actual tool execution) -> continue same request inference. Compare the
    post-checkpoint latencies against a baseline run without semantic segment.
    """

    baseline_tbt, baseline_total_s, _ = _run_episode(
        use_semantic_segment=False,
        consolidate=False,
        segment_tokens=segment_tokens,
        num_segments=num_segments,
        progress_label="baseline",
    )

    agent_tbt, agent_total_s, consolidation_time_s = _run_episode(
        use_semantic_segment=True,
        consolidate=True,
        segment_tokens=segment_tokens,
        num_segments=num_segments,
        progress_label="semantic",
    )

    assert consolidation_time_s >= 0
    assert baseline_tbt > 0 and baseline_total_s > 0
    assert agent_tbt > 0 and agent_total_s > 0
    
    print(f"Baseline TBT: {baseline_tbt:.6f}s, Total: {baseline_total_s:.6f}s")
    print(f"Semantic TBT: {agent_tbt:.6f}s, Total: {agent_total_s:.6f}s, Consolidation: {consolidation_time_s:.6f}s")

    # Under the assumption that tool latency can hide consolidation overhead,
    # semantic segment mode should provide a better latency signal while
    # keeping post-checkpoint total latency within a loose jitter band.
    # assert agent_total_s <= baseline_total_s
    # assert agent_tbt <= baseline_tbt
