# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import gc
import os
import statistics
import time
from collections.abc import Callable

import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm import _custom_ops as ops
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.transformers_utils.config import get_config, get_hf_text_config

DEFAULT_MODEL = os.environ.get("VLLM_BENCH_MODEL", "Qwen/Qwen3-8B")
DEFAULT_LOAD_FORMAT = os.environ.get("VLLM_BENCH_LOAD_FORMAT", "dummy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TTFT curve for paged restore vs contiguous restore. "
            "TTFT is modeled as: restore(host->gpu) + prefill(new tokens)."
        )
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--load-format", type=str, default=DEFAULT_LOAD_FORMAT)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=0,
        help="Number of model layers to keep for this benchmark. Use 0 for all layers.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vllm-block-size", type=int, default=16)

    parser.add_argument("--restore-start-tokens", type=int, default=800)
    parser.add_argument("--restore-step-tokens", type=int, default=800)
    parser.add_argument("--restore-max-tokens", type=int, default=32 * 1024)
    parser.add_argument("--prefill-new-tokens", type=int, default=400)

    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)

    parser.add_argument(
        "--random-mapping",
        action="store_true",
        default=True,
        help="Use random mapping for paged restore.",
    )
    parser.add_argument(
        "--sequential-mapping",
        action="store_false",
        dest="random_mapping",
        help="Use sequential mapping for paged restore.",
    )
    parser.add_argument("--mapping-seed", type=int, default=0)

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--restart-llm-between-prefixes",
        action="store_true",
        default=True,
        help=(
            "Restart LLM engine for each prefill prefix point to avoid KV-cache "
            "accumulation and scheduler waiting stalls."
        ),
    )
    parser.add_argument(
        "--reuse-llm-across-prefixes",
        action="store_false",
        dest="restart_llm_between_prefixes",
        help="Reuse one LLM across all prefix points (faster but less robust).",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmarks/ttft_restore_prefill_curve.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="benchmarks/ttft_restore_prefill_curve.png",
    )
    return parser.parse_args()


def _resolve_num_layers(model: str, requested_num_layers: int) -> int:
    if requested_num_layers > 0:
        return requested_num_layers

    hf_config = get_config(model, trust_remote_code=False)
    text_config = get_hf_text_config(hf_config)
    resolved_num_layers = getattr(text_config, "num_hidden_layers", None)
    if resolved_num_layers is None:
        raise RuntimeError(
            "Unable to determine model layer count; pass --num-layers explicitly."
        )
    return int(resolved_num_layers)


def _resolve_vocab_size(model: str) -> int:
    hf_config = get_config(model, trust_remote_code=False)
    text_config = get_hf_text_config(hf_config)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(hf_config, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError(
            "Unable to determine model vocab size; cannot build synthetic prompts."
        )
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise RuntimeError(f"Invalid vocab size resolved for model: {vocab_size}")
    return vocab_size


def _resolve_model_max_len(model: str) -> int | None:
    hf_config = get_config(model, trust_remote_code=False)
    text_config = get_hf_text_config(hf_config)

    for cfg in (text_config, hf_config):
        for attr in ("max_position_embeddings", "n_positions", "seq_length",
                     "model_max_length"):
            value = getattr(cfg, attr, None)
            if value is None:
                continue
            try:
                max_len = int(value)
            except (TypeError, ValueError):
                continue
            if max_len > 0:
                return max_len
    return None


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
        enable_prefix_caching=False,
    )
    if num_layers > 0:
        llm_kwargs["hf_overrides"] = {"num_hidden_layers": num_layers}
    return LLM(**llm_kwargs)


def _build_prompt(
    prompt_len: int,
    request_index: int,
    vocab_size: int,
) -> TokensPrompt:
    token_ids = [(request_index + offset) % vocab_size for offset in range(prompt_len)]
    return TokensPrompt(prompt_token_ids=token_ids)


def _build_prompts(
    prompt_len: int,
    batch_size: int,
    vocab_size: int,
) -> list[TokensPrompt]:
    return [
        _build_prompt(prompt_len, request_index=i, vocab_size=vocab_size)
        for i in range(batch_size)
    ]


def _measure_prefill_ms(
    llm: LLM,
    prompts: list[TokensPrompt],
    sampling_params: SamplingParams,
    warmup_iters: int,
    iters: int,
    *,
    label: str,
) -> float:
    values: list[float] = []
    total_iters = warmup_iters + iters

    for i in range(total_iters):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if not outputs:
            raise RuntimeError("No outputs from LLM.generate during prefill benchmark.")

        per_req_ms: list[float] = []
        for out in outputs:
            metrics = out.metrics
            if metrics is None:
                raise RuntimeError("Request metrics unavailable.")
            if metrics.scheduled_ts is None or metrics.first_token_ts is None:
                raise RuntimeError("Missing scheduled_ts/first_token_ts in metrics.")
            per_req_ms.append(max(0.0, (metrics.first_token_ts - metrics.scheduled_ts) * 1000.0))

        iter_ms = statistics.mean(per_req_ms)
        if i >= warmup_iters:
            values.append(iter_ms)
            print(
                f"prefill[{label}] iter={i - warmup_iters + 1}/{iters} "
                f"ttft={iter_ms:.3f}ms",
                flush=True,
            )

    return statistics.mean(values)


def _measure_prefill_curve_ms(
    *,
    llm_builder: Callable[[], LLM],
    restart_llm_between_prefixes: bool,
    restore_tokens_list: list[int],
    prefill_new_tokens: int,
    batch_size: int,
    vocab_size: int,
    sampling_params: SamplingParams,
    warmup_iters: int,
    iters: int,
) -> list[float]:
    prefill_curve_ms: list[float] = []
    shared_llm: LLM | None = None

    if not restart_llm_between_prefixes:
        shared_llm = llm_builder()

    try:
        for prefix_tokens in restore_tokens_list:
            prompt_len_prefix = prefix_tokens
            prompt_len_prefix_plus_new = prefix_tokens + prefill_new_tokens

            prompts_prefix = _build_prompts(
                prompt_len_prefix,
                batch_size,
                vocab_size,
            )
            prompts_prefix_plus_new = _build_prompts(
                prompt_len_prefix_plus_new,
                batch_size,
                vocab_size,
            )

            local_llm: LLM | None = shared_llm
            if restart_llm_between_prefixes:
                local_llm = llm_builder()

            try:
                print(
                    f"\nBenchmarking prefill increment at prefix={prefix_tokens} "
                    f"(measure {prefix_tokens}->{prefix_tokens + prefill_new_tokens})...",
                    flush=True,
                )
                ttft_prefix_ms = _measure_prefill_ms(
                    local_llm,
                    prompts_prefix,
                    sampling_params,
                    warmup_iters=warmup_iters,
                    iters=iters,
                    label=f"prefix={prompt_len_prefix}",
                )
                ttft_prefix_plus_new_ms = _measure_prefill_ms(
                    local_llm,
                    prompts_prefix_plus_new,
                    sampling_params,
                    warmup_iters=warmup_iters,
                    iters=iters,
                    label=f"prefix+new={prompt_len_prefix_plus_new}",
                )

                incremental_prefill_ms = max(0.0,
                                             ttft_prefix_plus_new_ms - ttft_prefix_ms)
                prefill_curve_ms.append(incremental_prefill_ms)
                print(
                    f"prefill_increment_ms(prefix={prefix_tokens}, new={prefill_new_tokens})="
                    f"{incremental_prefill_ms:.3f} "
                    f"(prefix_ttft={ttft_prefix_ms:.3f}, "
                    f"prefix_plus_new_ttft={ttft_prefix_plus_new_ms:.3f})",
                    flush=True,
                )
            finally:
                if restart_llm_between_prefixes and local_llm is not None:
                    del local_llm
                    cleanup_dist_env_and_memory()
    finally:
        if shared_llm is not None:
            del shared_llm
            cleanup_dist_env_and_memory()

    return prefill_curve_ms


def _build_block_mapping(
    *,
    num_blocks: int,
    random_mapping: bool,
    mapping_seed: int,
) -> torch.Tensor:
    map_src = torch.arange(num_blocks, dtype=torch.int64)
    map_dst = torch.arange(num_blocks, dtype=torch.int64)
    if random_mapping:
        torch.manual_seed(mapping_seed)
        map_src = torch.randperm(num_blocks, dtype=torch.int64)
        map_dst = torch.randperm(num_blocks, dtype=torch.int64)
    return torch.stack([map_src, map_dst], dim=1).cpu()


def _measure_restore_paged_ms(
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


def _measure_restore_contiguous_ms(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for src, dst in zip(src_layers, dst_layers):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _measure_restore_curve(
    *,
    restore_tokens_list: list[int],
    batch_size: int,
    num_layers: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    random_mapping: bool,
    mapping_seed: int,
    warmup_iters: int,
    iters: int,
) -> tuple[list[float], list[float]]:
    paged_values: list[float] = []
    contiguous_values: list[float] = []

    elements_per_block = block_size * num_heads * head_dim * 2
    total_iters = warmup_iters + iters

    for restore_tokens in restore_tokens_list:
        num_blocks_per_req = restore_tokens // block_size
        num_blocks_total = num_blocks_per_req * batch_size
        cpu_shape = (num_blocks_total, elements_per_block)

        src_layers = [
            torch.randn(cpu_shape, dtype=torch.float16, device="cpu").pin_memory()
            for _ in range(num_layers)
        ]
        dst_layers = [
            torch.empty(cpu_shape, dtype=torch.float16, device="cuda")
            for _ in range(num_layers)
        ]
        block_mapping = _build_block_mapping(
            num_blocks=num_blocks_total,
            random_mapping=random_mapping,
            mapping_seed=mapping_seed,
        )

        paged_iter_values: list[float] = []
        contiguous_iter_values: list[float] = []

        print(
            f"\nrestore_tokens={restore_tokens} (blocks={num_blocks_total})",
            flush=True,
        )

        try:
            for i in range(total_iters):
                paged_ms = _measure_restore_paged_ms(src_layers, dst_layers,
                                                     block_mapping)
                contiguous_ms = _measure_restore_contiguous_ms(src_layers,
                                                               dst_layers)
                if i >= warmup_iters:
                    paged_iter_values.append(paged_ms)
                    contiguous_iter_values.append(contiguous_ms)
                    print(
                        f"  iter={i - warmup_iters + 1}/{iters} "
                        f"paged={paged_ms:.3f}ms contiguous={contiguous_ms:.3f}ms",
                        flush=True,
                    )
        finally:
            del src_layers
            del dst_layers
            del block_mapping
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        paged_values.append(statistics.mean(paged_iter_values))
        contiguous_values.append(statistics.mean(contiguous_iter_values))

    return paged_values, contiguous_values


def _save_csv(
    *,
    output_csv: str,
    restore_tokens_list: list[int],
    restore_paged_ms: list[float],
    restore_contiguous_ms: list[float],
    prefill_ms_list: list[float],
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "restore_tokens",
            "restore_paged_ms",
            "restore_contiguous_ms",
            "prefill_ms",
            "ttft_paged_ms",
            "ttft_contiguous_ms",
        ])
        for x, paged_ms, contiguous_ms, prefill_ms in zip(
            restore_tokens_list,
            restore_paged_ms,
            restore_contiguous_ms,
            prefill_ms_list,
        ):
            writer.writerow([
                x,
                f"{paged_ms:.6f}",
                f"{contiguous_ms:.6f}",
                f"{prefill_ms:.6f}",
                f"{paged_ms + prefill_ms:.6f}",
                f"{contiguous_ms + prefill_ms:.6f}",
            ])


def _plot_curve(
    *,
    output_plot: str,
    restore_tokens_list: list[int],
    ttft_paged_ms: list[float],
    ttft_contiguous_ms: list[float],
    prefill_ms_list: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Please install it first."
        ) from exc

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        restore_tokens_list,
        ttft_paged_ms,
        marker="o",
        linewidth=2,
        label="TTFT (vLLM)",
    )
    ax.plot(
        restore_tokens_list,
        ttft_contiguous_ms,
        marker="o",
        linewidth=2,
        label="TTFT (Ours)",
    )
    ax.plot(
        restore_tokens_list,
        prefill_ms_list,
        marker="^",
        linewidth=1.5,
        linestyle="--",
        color="gray",
        label="Prefill (incremental)",
    )

    improvement_list = [
        (vllm_ms - ours_ms) / vllm_ms if vllm_ms > 0 else 0.0
        for vllm_ms, ours_ms in zip(ttft_paged_ms, ttft_contiguous_ms)
    ]
    max_idx = max(range(len(improvement_list)), key=improvement_list.__getitem__)
    max_improvement_pct = improvement_list[max_idx] * 100.0
    x_max_gain = restore_tokens_list[max_idx]
    y_max_gain = max(ttft_paged_ms[max_idx], ttft_contiguous_ms[max_idx])

    ax.axvline(
        x=x_max_gain,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.annotate(
        f"Max gain: {max_improvement_pct:.2f}%",
        xy=(x_max_gain, y_max_gain),
        xytext=(-150, 50),
        textcoords="offset points",
        fontsize=18,
        color="crimson",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.2},
    )

    ax.set_xlabel("Number of Prefix Tokens", fontsize=24)
    ax.set_ylabel("TTFT (ms)", fontsize=24)
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=24)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_layers < 0:
        raise ValueError("--num-layers must be >= 0")
    if args.vllm_block_size <= 0:
        raise ValueError("--vllm-block-size must be > 0")
    if args.restore_start_tokens <= 0 or args.restore_step_tokens <= 0:
        raise ValueError("restore start/step tokens must be > 0")
    if args.restore_max_tokens < args.restore_start_tokens:
        raise ValueError("--restore-max-tokens must be >= --restore-start-tokens")
    if args.prefill_new_tokens <= 0:
        raise ValueError("--prefill-new-tokens must be > 0")

    for key, val in {
        "restore_start_tokens": args.restore_start_tokens,
        "restore_step_tokens": args.restore_step_tokens,
        "restore_max_tokens": args.restore_max_tokens,
    }.items():
        if val % args.vllm_block_size != 0:
            raise ValueError(
                f"{key}={val} must be divisible by --vllm-block-size={args.vllm_block_size}"
            )

    restore_tokens_list = list(
        range(
            args.restore_start_tokens,
            args.restore_max_tokens + 1,
            args.restore_step_tokens,
        )
    )

    effective_num_layers = _resolve_num_layers(args.model, args.num_layers)
    vocab_size = _resolve_vocab_size(args.model)
    model_max_len = _resolve_model_max_len(args.model)
    effective_gpu_mem_util = _resolve_gpu_memory_utilization(
        args.gpu_memory_utilization
    )

    if effective_gpu_mem_util < 0.10:
        free_mem, total_mem = torch.cuda.mem_get_info()
        raise RuntimeError(
            "Insufficient free GPU memory to reliably run this benchmark. "
            f"Only {free_mem / 1024**3:.2f} / {total_mem / 1024**3:.2f} GiB is free. "
            "Please stop other GPU processes (for example stale VLLM::EngineCore) "
            "or switch to a less loaded GPU."
        )

    required_prompt_len = restore_tokens_list[-1] + args.prefill_new_tokens
    if model_max_len is not None and required_prompt_len > model_max_len:
        raise ValueError(
            "Requested prefill benchmark exceeds model context length: "
            f"required max prompt len={required_prompt_len} "
            f"(restore_max_tokens + prefill_new_tokens), "
            f"model_max_len={model_max_len}. "
            "Please reduce --restore-max-tokens or --prefill-new-tokens."
        )

    print("=== Scenario ===", flush=True)
    print(
        "Assumption: each round has 400 decode tokens and 400 tool tokens. "
        "So restored KV tokens grow as 800, 1600, ... until 64K.",
        flush=True,
    )
    print(
        f"restore_tokens_range=[{restore_tokens_list[0]}, ..., {restore_tokens_list[-1]}], "
        f"step={args.restore_step_tokens}",
        flush=True,
    )
    print(
        f"prefill_new_tokens={args.prefill_new_tokens}, batch_size={args.batch_size}, "
        f"num_layers={effective_num_layers}",
        flush=True,
    )

    prefill_ms_curve: list[float] = []
    sampling_params_prefill = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
    )

    def llm_builder() -> LLM:
        return _build_llm(
            model=args.model,
            load_format=args.load_format,
            num_layers=effective_num_layers,
            gpu_memory_utilization=effective_gpu_mem_util,
            max_num_batched_tokens=max(
                args.max_num_batched_tokens,
                args.batch_size * (args.prefill_new_tokens + 1),
            ),
            max_num_seqs=max(args.max_num_seqs, args.batch_size),
        )

    print(
        "\nBenchmarking prefill curve with varying prefix lengths "
        f"(restart_llm_between_prefixes={args.restart_llm_between_prefixes})...",
        flush=True,
    )
    prefill_ms_curve = _measure_prefill_curve_ms(
        llm_builder=llm_builder,
        restart_llm_between_prefixes=args.restart_llm_between_prefixes,
        restore_tokens_list=restore_tokens_list,
        prefill_new_tokens=args.prefill_new_tokens,
        batch_size=args.batch_size,
        vocab_size=vocab_size,
        sampling_params=sampling_params_prefill,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    print(
        "prefill_curve_ms(first,last)="
        f"({prefill_ms_curve[0]:.3f}, {prefill_ms_curve[-1]:.3f})",
        flush=True,
    )

    print("\nBenchmarking restore curves...", flush=True)
    restore_paged_ms, restore_contiguous_ms = _measure_restore_curve(
        restore_tokens_list=restore_tokens_list,
        batch_size=args.batch_size,
        num_layers=effective_num_layers,
        block_size=args.vllm_block_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        random_mapping=args.random_mapping,
        mapping_seed=args.mapping_seed,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    ttft_paged_ms = [x + y for x, y in zip(restore_paged_ms, prefill_ms_curve)]
    ttft_contiguous_ms = [
        x + y for x, y in zip(restore_contiguous_ms, prefill_ms_curve)
    ]

    _save_csv(
        output_csv=args.output_csv,
        restore_tokens_list=restore_tokens_list,
        restore_paged_ms=restore_paged_ms,
        restore_contiguous_ms=restore_contiguous_ms,
        prefill_ms_list=prefill_ms_curve,
    )
    _plot_curve(
        output_plot=args.output_plot,
        restore_tokens_list=restore_tokens_list,
        ttft_paged_ms=ttft_paged_ms,
        ttft_contiguous_ms=ttft_contiguous_ms,
        prefill_ms_list=prefill_ms_curve,
    )

    print("\n=== Results ===", flush=True)
    print(f"output_csv={args.output_csv}", flush=True)
    print(f"output_plot={args.output_plot}", flush=True)
    print(
        "first_point(restore_tokens, prefill_ms, ttft_paged_ms, ttft_contiguous_ms)="
        f"({restore_tokens_list[0]}, {prefill_ms_curve[0]:.3f}, "
        f"{ttft_paged_ms[0]:.3f}, {ttft_contiguous_ms[0]:.3f})",
        flush=True,
    )
    print(
        "last_point(restore_tokens, prefill_ms, ttft_paged_ms, ttft_contiguous_ms)="
        f"({restore_tokens_list[-1]}, {prefill_ms_curve[-1]:.3f}, "
        f"{ttft_paged_ms[-1]:.3f}, {ttft_contiguous_ms[-1]:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()