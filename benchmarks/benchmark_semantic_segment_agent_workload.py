# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from typing import NamedTuple

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


DEFAULT_MODEL = os.environ.get("VLLM_BENCH_MODEL", "Qwen/Qwen3-8B")
DEFAULT_LOAD_FORMAT = os.environ.get("VLLM_BENCH_LOAD_FORMAT", "dummy")
ENABLE_TORCH_PROFILER = os.environ.get("VLLM_ENABLE_TORCH_PROFILER", "1") == "1"


class EpisodeMetrics(NamedTuple):
    tbt_s: float
    total_latency_s: float
    execute_memory_ops_time_s: float
    episode_wall_s: float


def _configure_worker_profiler_env(args: argparse.Namespace) -> None:
    if not ENABLE_TORCH_PROFILER or not args.enable_worker_profiler:
        return

    trace_dir = os.path.abspath(os.path.expanduser(args.worker_profiler_dir))
    os.makedirs(trace_dir, exist_ok=True)

    os.environ["VLLM_TORCH_PROFILER_DIR"] = trace_dir
    os.environ["VLLM_PROFILER_DELAY_ITERS"] = str(args.worker_profiler_delay_steps)
    os.environ["VLLM_PROFILER_MAX_ITERS"] = str(args.worker_profiler_active_steps)
    os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = (
        "1" if args.worker_profiler_custom_scopes else "0"
    )


def _run_episode(
    *,
    model: str,
    use_semantic_segment: bool,
    segment_tokens: int,
    num_segments: int,
    progress_label: str,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    load_format: str,
    num_hidden_layers: int | None,
) -> EpisodeMetrics:
    if segment_tokens <= 0 or num_segments <= 1:
        raise ValueError("segment_tokens>0 and num_segments>1 are required")

    llm_kwargs = dict(
        model=model,
        load_format=load_format,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if num_hidden_layers is not None:
        llm_kwargs["hf_overrides"] = {"num_hidden_layers": num_hidden_layers}
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
        execute_memory_ops_time_s = 0.0
        next_boundary = segment_tokens
        current_segment = 1
        worker_profiler_started = False
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

        while not finished:
            step_count += 1
            t0 = time.perf_counter()
            outputs = llm.llm_engine.step()
            dt = time.perf_counter() - t0

            if step_count % 100 == 0:
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
                        counted_new_tokens = min(new_tokens, total_tokens - token_steps)
                        token_steps += counted_new_tokens
                        prev_total_token_count = current_total_token_count
                        generated_time_s += dt
                        generated_token_count += counted_new_tokens

                    while token_steps >= next_boundary and current_segment < num_segments:
                        if use_semantic_segment:
                            print(
                                f"[{progress_label}] consolidate at "
                                f"tokens={token_steps}, segment={current_segment}",
                                flush=True,
                            )
                            t1 = time.perf_counter()
                            llm.seal(req_a)
                            execute_memory_ops_time_s += time.perf_counter() - t1
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

        assert generated_time_s > 0, "No generated-token latency observed"
        assert generated_token_count > 0

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

        return EpisodeMetrics(
            tbt_s=tbt_s,
            total_latency_s=total_latency_s,
            execute_memory_ops_time_s=execute_memory_ops_time_s,
            episode_wall_s=episode_wall_s,
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic segment benchmark (copied from test workload, with larger default model)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--load-format",
        type=str,
        default=DEFAULT_LOAD_FORMAT,
        help="vLLM load format. Use 'dummy' to avoid downloading model weights.",
    )
    parser.add_argument("--segment-tokens", type=int, default=4096)
    parser.add_argument("--num-segments", type=int, default=9)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "semantic", "both"],
        default="both",
        help="Run baseline / semantic / both.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=int(os.environ.get("VLLM_BENCH_NUM_HIDDEN_LAYERS", "12")),
        help="Override model config num_hidden_layers; set to 0 to disable override.",
    )
    parser.add_argument(
        "--enable-worker-profiler",
        action="store_true",
        default=os.environ.get("VLLM_ENABLE_WORKER_PROFILER", "1") == "1",
        help="Enable vLLM worker profiler and export traces.",
    )
    parser.add_argument(
        "--worker-profiler-dir",
        type=str,
        default=os.environ.get("VLLM_TORCH_PROFILER_DIR", ".trace/semseg_worker"),
        help="Trace output dir for vLLM worker profiler.",
    )
    parser.add_argument(
        "--worker-profiler-delay-steps",
        type=int,
        default=int(os.environ.get("VLLM_PROFILER_DELAY_ITERS", "36000")),
        help="Delay worker profiling start by N steps.",
    )
    parser.add_argument(
        "--worker-profiler-active-steps",
        type=int,
        default=int(os.environ.get("VLLM_PROFILER_MAX_ITERS", "10")),
        help="Maximum worker profiling steps (0 means unlimited).",
    )
    parser.add_argument(
        "--worker-profiler-custom-scopes",
        action="store_true",
        default=os.environ.get("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1") == "1",
        help="Enable custom profiling scopes in worker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_hidden_layers = args.num_hidden_layers if args.num_hidden_layers > 0 else None
    _configure_worker_profiler_env(args)

    print(f"Benchmark model: {args.model}", flush=True)
    print(f"Load format: {args.load_format}", flush=True)
    print(
        "Num hidden layers override: "
        f"{num_hidden_layers if num_hidden_layers is not None else 'disabled'}",
        flush=True,
    )
    if ENABLE_TORCH_PROFILER and args.enable_worker_profiler:
        print(
            "Worker profiler: "
            f"dir={os.environ.get('VLLM_TORCH_PROFILER_DIR')}, "
            f"delay={os.environ.get('VLLM_PROFILER_DELAY_ITERS')}, "
            f"active={os.environ.get('VLLM_PROFILER_MAX_ITERS')}, "
            f"custom_scopes={os.environ.get('VLLM_CUSTOM_SCOPES_FOR_PROFILING')}",
            flush=True,
        )

    baseline_metrics = None
    semantic_metrics = None

    if args.mode in {"baseline", "both"}:
        baseline_metrics = _run_episode(
            model=args.model,
            use_semantic_segment=False,
            segment_tokens=args.segment_tokens,
            num_segments=args.num_segments,
            progress_label="baseline",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            load_format=args.load_format,
            num_hidden_layers=num_hidden_layers,
        )

    if args.mode in {"semantic", "both"}:
        semantic_metrics = _run_episode(
            model=args.model,
            use_semantic_segment=True,
            segment_tokens=args.segment_tokens,
            num_segments=args.num_segments,
            progress_label="semantic",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            load_format=args.load_format,
            num_hidden_layers=num_hidden_layers,
        )

    if baseline_metrics is not None:
        print(
            f"Baseline TBT: {baseline_metrics.tbt_s:.6f}s, "
            f"Generated: {baseline_metrics.total_latency_s:.6f}s, "
            f"Wall: {baseline_metrics.episode_wall_s:.6f}s"
        )

    if semantic_metrics is not None:
        print(
            f"Semantic TBT: {semantic_metrics.tbt_s:.6f}s, "
            f"Generated: {semantic_metrics.total_latency_s:.6f}s, "
            f"Wall: {semantic_metrics.episode_wall_s:.6f}s, "
            f"MemoryOps: {semantic_metrics.execute_memory_ops_time_s:.6f}s"
        )

    if baseline_metrics is not None and semantic_metrics is not None:
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


if __name__ == "__main__":
    main()
