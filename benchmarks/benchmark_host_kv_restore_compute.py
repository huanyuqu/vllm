# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import statistics
import time

import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm import _custom_ops as ops
from vllm.distributed import cleanup_dist_env_and_memory

DEFAULT_MODEL = os.environ.get("VLLM_BENCH_MODEL", "Qwen/Qwen3-8B")
DEFAULT_LOAD_FORMAT = os.environ.get("VLLM_BENCH_LOAD_FORMAT", "dummy")


def _summary_ms(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: n/a"
    vals = sorted(values)
    n = len(vals)
    p50 = vals[n // 2]
    p90 = vals[min(n - 1, int(0.9 * n))]
    return (
        f"{name}: mean={statistics.mean(vals):.3f} ms, "
        f"p50={p50:.3f} ms, p90={p90:.3f} ms"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicit benchmark: simulate KV restore (host->GPU) with swap_blocks "
            "then simulate decode compute with existing 32K prefix."
        )
    )
    parser.add_argument("--prompt-len", type=int, default=32 * 1024)
    parser.add_argument("--decode-tokens", type=int, default=400)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)

    parser.add_argument(
        "--vllm-block-size",
        type=int,
        default=16,
        help="vLLM KV block size in tokens.",
    )

    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=1)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--load-format", type=str, default=DEFAULT_LOAD_FORMAT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32 * 1024 + 32)
    parser.add_argument("--max-num-seqs", type=int, default=4)

    parser.add_argument(
        "--random-mapping",
        action="store_true",
        default=True,
        help="Use random destination block mapping for restore transfer.",
    )
    parser.add_argument(
        "--sequential-mapping",
        action="store_false",
        dest="random_mapping",
        help="Disable random mapping and use sequential block mapping.",
    )
    parser.add_argument(
        "--mapping-seed",
        type=int,
        default=0,
        help="Random seed used when --random-mapping is enabled.",
    )

    return parser.parse_args()


def _build_block_mapping(
    *,
    num_blocks: int,
    random_mapping: bool,
    mapping_seed: int,
) -> torch.Tensor:
    map_src = torch.arange(num_blocks, dtype=torch.int64)
    map_dst = torch.arange(num_blocks, dtype=torch.int64)
    if random_mapping:
        # Use the same mapping generation style as benchmark_swap_blocks.py.
        torch.manual_seed(mapping_seed)
        map_src = torch.randperm(num_blocks, dtype=torch.int64)
        map_dst = torch.randperm(num_blocks, dtype=torch.int64)

    return torch.stack([map_src, map_dst], dim=1).cpu()


def _simulate_restore_ms(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    block_mapping: torch.Tensor,
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for src, dst in zip(src_layers, dst_layers):
        ops.swap_blocks(src, dst, block_mapping)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _simulate_compute_ms(
    llm: LLM,
    prompt: TokensPrompt,
    sampling_params_decode: SamplingParams,
) -> float:
    outputs = llm.generate([prompt], sampling_params_decode, use_tqdm=False)
    if not outputs:
        raise RuntimeError("No output returned by LLM.generate.")

    first_output = outputs[0]
    if not first_output.outputs:
        raise RuntimeError("RequestOutput contains no completion outputs.")

    metrics = first_output.metrics
    if metrics is None:
        raise RuntimeError(
            "Request metrics are unavailable; cannot isolate decode-only time."
        )

    decode_ms = (metrics.last_token_ts - metrics.first_token_ts) * 1000.0
    return max(0.0, decode_ms)


def _build_prompt(prompt_len: int, first_token: int) -> TokensPrompt:
    token_ids = [0] * prompt_len
    token_ids[0] = first_token
    return TokensPrompt(prompt_token_ids=token_ids)


def _build_llm(
    *,
    model: str,
    load_format: str,
    num_layers: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    max_num_seqs: int,
) -> LLM:
    llm_kwargs = dict(
        model=model,
        load_format=load_format,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    llm_kwargs["hf_overrides"] = {"num_hidden_layers": num_layers}
    return LLM(**llm_kwargs)


def _resolve_gpu_memory_utilization(requested: float) -> float:
    if not torch.cuda.is_available():
        return requested
    free_mem, total_mem = torch.cuda.mem_get_info()
    safe_margin = 0.02
    max_allowed = max(0.05, min(0.98, (free_mem / total_mem) - safe_margin))
    effective = min(requested, max_allowed)
    if effective < requested:
        print(
            "[warn] gpu_memory_utilization is auto-capped "
            f"from {requested:.3f} to {effective:.3f} "
            f"(free={free_mem / 1024**3:.2f} GiB, total={total_mem / 1024**3:.2f} GiB)",
            flush=True,
        )
    return effective


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.prompt_len <= 0:
        raise ValueError("--prompt-len must be > 0")
    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be > 0")
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be > 0")
    if args.vllm_block_size <= 0:
        raise ValueError("--vllm-block-size must be > 0")
    if args.prompt_len % args.vllm_block_size != 0:
        raise ValueError("--prompt-len must be divisible by --vllm-block-size")

    num_blocks = args.prompt_len // args.vllm_block_size
    elements_per_block = args.vllm_block_size * args.num_heads * args.head_dim * 2
    block_size_bytes = elements_per_block * 2  # float16
    total_kv_mb_per_layer = (num_blocks * block_size_bytes) / (1024**2)
    print("=== Benchmark Config ===", flush=True)
    print(
        f"prompt_len={args.prompt_len}, decode_tokens={args.decode_tokens}, "
        f"num_layers={args.num_layers}",
        flush=True,
    )
    print(
        f"num_heads={args.num_heads}, head_dim={args.head_dim}, "
        f"vllm_block_size={args.vllm_block_size}",
        flush=True,
    )
    print(
        f"num_blocks={num_blocks}, elements_per_block={elements_per_block}, "
        f"block_size={block_size_bytes / 1024:.2f} KB",
        flush=True,
    )
    print(
        f"kv_size_per_layer={total_kv_mb_per_layer:.2f} MB, "
        f"kv_size_all_layers={total_kv_mb_per_layer * args.num_layers:.2f} MB",
        flush=True,
    )
    print(
        f"mapping_mode={'random' if args.random_mapping else 'sequential'}, "
        f"mapping_seed={args.mapping_seed}",
        flush=True,
    )
    print(
        f"model={args.model}, load_format={args.load_format}, "
        f"num_layers_for_model_override={args.num_layers}",
        flush=True,
    )

    effective_gpu_mem_util = _resolve_gpu_memory_utilization(
        args.gpu_memory_utilization
    )
    print(
        f"gpu_memory_utilization(requested={args.gpu_memory_utilization:.3f}, "
        f"effective={effective_gpu_mem_util:.3f})",
        flush=True,
    )

    cpu_shape = (
        num_blocks,
        elements_per_block,
    )
    gpu_shape = cpu_shape

    src_layers = [
        torch.randn(cpu_shape, dtype=torch.float16, device="cpu").pin_memory()
        for _ in range(args.num_layers)
    ]
    dst_layers = [
        torch.empty(gpu_shape, dtype=torch.float16, device="cuda")
        for _ in range(args.num_layers)
    ]

    block_mapping = _build_block_mapping(
        num_blocks=num_blocks,
        random_mapping=args.random_mapping,
        mapping_seed=args.mapping_seed,
    )

    restore_ms_values: list[float] = []
    compute_ms_values: list[float] = []

    llm: LLM | None = None
    sampling_params_decode = SamplingParams(
        temperature=0.0,
        # Force vLLM to decode exactly N tokens for each request.
        max_tokens=args.decode_tokens,
        min_tokens=args.decode_tokens,
        ignore_eos=True,
    )
    prompt_fixed = _build_prompt(args.prompt_len, first_token=123456)

    llm = _build_llm(
        model=args.model,
        load_format=args.load_format,
        num_layers=args.num_layers,
        gpu_memory_utilization=effective_gpu_mem_util,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    # Prime once so later runs represent existing-prefix behavior.
    llm.generate([prompt_fixed], sampling_params_decode, use_tqdm=False)

    total_iters = args.warmup_iters + args.iters
    print("\nWarming up and benchmarking...", flush=True)
    try:
        for i in range(total_iters):
            restore_ms = _simulate_restore_ms(src_layers, dst_layers, block_mapping)
            compute_ms = _simulate_compute_ms(
                llm,
                prompt_fixed,
                sampling_params_decode,
            )

            if i >= args.warmup_iters:
                restore_ms_values.append(restore_ms)
                compute_ms_values.append(compute_ms)
                print(
                    f"iter={i - args.warmup_iters + 1}/{args.iters} "
                    f"restore={restore_ms:.3f}ms "
                    f"compute_real_model={compute_ms:.3f}ms",
                    flush=True,
                )
    finally:
        if llm is not None:
            del llm
        cleanup_dist_env_and_memory()

    print("\n=== Results ===", flush=True)
    print(_summary_ms("restore_host_to_gpu", restore_ms_values), flush=True)
    print(
        _summary_ms("compute_with_real_model_on_32k_prefix", compute_ms_values),
        flush=True,
    )

    if restore_ms_values and compute_ms_values:
        mean_restore = statistics.mean(restore_ms_values)
        mean_compute = statistics.mean(compute_ms_values)
        ratio = mean_restore / mean_compute if mean_compute > 0 else float("inf")
        print(
            f"time_ratio(restore:compute)={ratio:.3f}:1 "
            f"(restore={mean_restore:.3f}ms, compute={mean_compute:.3f}ms)",
            flush=True,
        )


if __name__ == "__main__":
    main()
