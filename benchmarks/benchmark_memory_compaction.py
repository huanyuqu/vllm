# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import random
import statistics
import time
from dataclasses import dataclass

import torch


DEFAULT_BLOCK_SIZE = 16
DEFAULT_DTYPE = "float16"


@dataclass
class Scenario:
    mode: str
    moves: list[tuple[int, int, int, int]]
    swaps: list[tuple[int, int, int, int]]
    copy_size_blocks: int
    op_count: int
    total_blocks: int


@dataclass
class TimingResult:
    mode: str
    iterations: int
    op_count: int
    copy_size_blocks: int
    total_blocks: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


class CompactionExecutor:
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.device = device
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.total_slots = num_blocks * block_size
        self.kv_caches = [
            torch.empty(
                self.total_slots,
                2,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def fill_random(self, seed: int) -> None:
        torch.manual_seed(seed)
        for kv_cache in self.kv_caches:
            kv_cache.normal_()
        self.sync()

    def apply_semantic_segment_memory_ops(
        self,
        moves: list[tuple[int, int, int, int]],
        swaps: list[tuple[int, int, int, int]],
    ) -> None:
        if not moves and not swaps:
            return

        if moves:
            for group_id, src_addr, dst_addr, size in moves:
                if group_id < len(self.kv_caches):
                    kv_cache = self.kv_caches[group_id]
                    flat_cache = kv_cache.view(-1, *kv_cache.shape[2:])
                    flat_cache[dst_addr:dst_addr + size].copy_(
                        flat_cache[src_addr:src_addr + size]
                    )

        if swaps:
            for group_id, addr1, addr2, size in swaps:
                if group_id < len(self.kv_caches):
                    kv_cache = self.kv_caches[group_id]
                    flat_cache = kv_cache.view(-1, *kv_cache.shape[2:])
                    temp = flat_cache[addr1:addr1 + size].clone()
                    flat_cache[addr1:addr1 + size].copy_(
                        flat_cache[addr2:addr2 + size]
                    )
                    flat_cache[addr2:addr2 + size].copy_(temp)

    def sync(self) -> None:
        torch.cuda.synchronize(self.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark manual semantic memory compaction copies on CUDA tensors. "
            "This avoids model loading and focuses on copy execution cost."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["move", "swap", "both"],
        default="both",
        help="Benchmark move-only, swap-only, or both scenarios.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--rebuild-each-iteration",
        action="store_true",
        help="Rebuild tensors and op list before each timing sample.",
    )
    parser.add_argument(
        "--num-ops",
        type=int,
        default=128,
        help="Number of move/swap ops in one compaction round.",
    )
    parser.add_argument(
        "--copy-size-blocks",
        type=int,
        default=1,
        help="How many contiguous blocks each op copies.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["random", "reverse", "strided"],
        default="random",
        help="Address distribution used to construct move/swap ops.",
    )
    parser.add_argument(
        "--src-stride-blocks",
        type=int,
        default=2,
        help="Source stride in blocks when --layout=strided.",
    )
    parser.add_argument(
        "--dst-stride-blocks",
        type=int,
        default=3,
        help="Destination stride in blocks when --layout=strided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print stage progress so long runs do not look stuck.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _make_chunk_starts(
    *,
    start_block: int,
    end_block: int,
    chunk_blocks: int,
) -> list[int]:
    if end_block <= start_block:
        return []
    return list(range(start_block, end_block - chunk_blocks + 1, chunk_blocks))


def _build_random_pairs(
    src_candidates: list[int],
    dst_candidates: list[int],
    num_ops: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if num_ops > len(src_candidates) or num_ops > len(dst_candidates):
        raise ValueError("Not enough candidate chunks for requested num_ops")
    src_choices = rng.sample(src_candidates, num_ops)
    dst_choices = rng.sample(dst_candidates, num_ops)
    return list(zip(src_choices, dst_choices))


def _build_reverse_pairs(
    src_candidates: list[int],
    dst_candidates: list[int],
    num_ops: int,
) -> list[tuple[int, int]]:
    if num_ops > len(src_candidates) or num_ops > len(dst_candidates):
        raise ValueError("Not enough candidate chunks for requested num_ops")
    src_choices = list(reversed(src_candidates[-num_ops:]))
    dst_choices = dst_candidates[:num_ops]
    return list(zip(src_choices, dst_choices))


def _build_strided_pairs(
    src_candidates: list[int],
    dst_candidates: list[int],
    num_ops: int,
    src_stride: int,
    dst_stride: int,
    chunk_blocks: int,
) -> list[tuple[int, int]]:
    src_step = max(1, src_stride) * chunk_blocks
    dst_step = max(1, dst_stride) * chunk_blocks
    src_choices = src_candidates[::src_step][:num_ops]
    dst_choices = dst_candidates[::dst_step][:num_ops]
    if len(src_choices) < num_ops or len(dst_choices) < num_ops:
        raise ValueError(
            "Strided layout cannot generate enough ops; adjust strides or num_blocks"
        )
    return list(zip(src_choices, dst_choices))


def _build_pairs(
    args: argparse.Namespace,
    *,
    rng: random.Random,
) -> list[tuple[int, int]]:
    chunk_blocks = args.copy_size_blocks
    midpoint = args.num_blocks // 2
    src_candidates = _make_chunk_starts(
        start_block=midpoint,
        end_block=args.num_blocks,
        chunk_blocks=chunk_blocks,
    )
    dst_candidates = _make_chunk_starts(
        start_block=0,
        end_block=midpoint,
        chunk_blocks=chunk_blocks,
    )
    if args.layout == "random":
        return _build_random_pairs(src_candidates, dst_candidates, args.num_ops, rng)
    if args.layout == "reverse":
        return _build_reverse_pairs(src_candidates, dst_candidates, args.num_ops)
    return _build_strided_pairs(
        src_candidates,
        dst_candidates,
        args.num_ops,
        args.src_stride_blocks,
        args.dst_stride_blocks,
        chunk_blocks,
    )


def _build_scenario(
    args: argparse.Namespace,
    *,
    mode: str,
    seed: int,
) -> Scenario:
    rng = random.Random(seed)
    pairs = _build_pairs(args, rng=rng)
    size = args.copy_size_blocks * args.block_size
    if mode == "move":
        moves = [
            (layer_id, src * args.block_size, dst * args.block_size, size)
            for layer_id in range(args.num_layers)
            for src, dst in pairs
        ]
        swaps: list[tuple[int, int, int, int]] = []
    else:
        moves = []
        swaps = [
            (layer_id, src * args.block_size, dst * args.block_size, size)
            for layer_id in range(args.num_layers)
            for src, dst in pairs
        ]
    return Scenario(
        mode=mode,
        moves=moves,
        swaps=swaps,
        copy_size_blocks=args.copy_size_blocks,
        op_count=len(pairs) * args.num_layers,
        total_blocks=len(pairs) * args.copy_size_blocks * args.num_layers,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _prepare_executor_and_scenario(
    args: argparse.Namespace,
    *,
    mode: str,
    seed: int,
) -> tuple[CompactionExecutor, Scenario]:
    executor = CompactionExecutor(
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype=_resolve_dtype(args.dtype),
        device=torch.device("cuda"),
    )
    executor.fill_random(seed=seed)
    scenario = _build_scenario(args, mode=mode, seed=seed)
    return executor, scenario


def _measure_once(executor: CompactionExecutor, scenario: Scenario) -> float:
    t0 = time.perf_counter()
    executor.apply_semantic_segment_memory_ops(scenario.moves, scenario.swaps)
    executor.sync()
    return (time.perf_counter() - t0) * 1000


def _run_mode(args: argparse.Namespace, mode: str) -> TimingResult:
    if args.progress:
        print(f"[{mode}] preparing warmup state", flush=True)
    warmup_executor, warmup_scenario = _prepare_executor_and_scenario(
        args,
        mode=mode,
        seed=args.seed,
    )
    try:
        if args.progress:
            print(f"[{mode}] warmup start", flush=True)
        for _ in range(args.warmup_iterations):
            _measure_once(warmup_executor, warmup_scenario)
        op_count = warmup_scenario.op_count
        copy_size_blocks = warmup_scenario.copy_size_blocks
        total_blocks = warmup_scenario.total_blocks
    finally:
        del warmup_executor

    timings_ms: list[float] = []
    if args.rebuild_each_iteration:
        for index in range(args.iterations):
            if args.progress:
                print(f"[{mode}] iteration {index + 1}/{args.iterations}", flush=True)
            executor, scenario = _prepare_executor_and_scenario(
                args,
                mode=mode,
                seed=args.seed + index + 1,
            )
            try:
                timings_ms.append(_measure_once(executor, scenario))
            finally:
                del executor
    else:
        if args.progress:
            print(f"[{mode}] preparing benchmark state", flush=True)
        executor, scenario = _prepare_executor_and_scenario(
            args,
            mode=mode,
            seed=args.seed + 1,
        )
        try:
            for index in range(args.iterations):
                if args.progress:
                    print(f"[{mode}] iteration {index + 1}/{args.iterations}", flush=True)
                timings_ms.append(_measure_once(executor, scenario))
        finally:
            del executor

    return TimingResult(
        mode=mode,
        iterations=args.iterations,
        op_count=op_count,
        copy_size_blocks=copy_size_blocks,
        total_blocks=total_blocks,
        mean_ms=statistics.fmean(timings_ms),
        median_ms=statistics.median(timings_ms),
        p95_ms=_percentile(timings_ms, 0.95),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
    )


def _print_result(result: TimingResult) -> None:
    print(
        f"mode={result.mode} iterations={result.iterations} "
        f"ops={result.op_count} copy_size_blocks={result.copy_size_blocks} "
        f"total_blocks={result.total_blocks} "
        f"mean_ms={result.mean_ms:.3f} median_ms={result.median_ms:.3f} "
        f"p95_ms={result.p95_ms:.3f} min_ms={result.min_ms:.3f} "
        f"max_ms={result.max_ms:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0")
    if args.num_ops <= 0:
        raise ValueError("--num-ops must be > 0")
    if args.copy_size_blocks <= 0:
        raise ValueError("--copy-size-blocks must be > 0")
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be > 0")

    modes = [args.mode] if args.mode != "both" else ["move", "swap"]
    for mode in modes:
        result = _run_mode(args, mode)
        _print_result(result)


if __name__ == "__main__":
    main()