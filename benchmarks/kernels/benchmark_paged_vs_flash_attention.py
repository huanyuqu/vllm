# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark paged attention vs non-paged (contiguous KV) flash attention.

This script focuses on *decode-style* attention:
- Q: 1 token per sequence (shape [B, Hq, D])
- KV: full context per sequence (length = seq_len)

Paged attention reads KV through block tables (scattered blocks).
Non-paged attention uses contiguous K/V and calls flash-attn or torch SDPA.

Important semantic note for decode:
- In typical decode, K/V only contain past tokens. In that case, attention can
    be run with causal=False (there are no future tokens present).
- If you set causal=True while Q has length 1, many implementations interpret
    the single query token as position 0 and it will only attend to the first key
    token. That makes the runtime unrealistically fast and not comparable.

Note:
- This benchmark does NOT validate numerical equivalence between the two
  implementations, since their KV layouts differ.
- It is meant to estimate the performance delta and guide hybrid design.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.utils.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random,
)

logger = init_logger(__name__)


SUPPORTED_HEAD_SIZES = {32, 64, 80, 96, 112, 120, 128, 192, 256}
SUPPORTED_BLOCK_SIZES = {8, 16, 32}


@dataclass
class BenchResult:
    name: str
    us: float


def _cuda_time_us(fn, iters: int, profile: bool = False) -> float:
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return (end - start) * 1e6 / iters


def _parse_csv_ints(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def _parse_csv_strs(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


def _format_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    # Stable column order.
    columns = list(rows[0].keys())
    out_lines = [",".join(columns)]
    for r in rows:
        out_lines.append(",".join(str(r.get(c, "")) for c in columns))
    return "\n".join(out_lines) + "\n"


def _format_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _choose_num_blocks(*,
                       batch_size: int,
                       max_num_blocks_per_seq: int,
                       num_blocks: int | None,
                       num_blocks_multiplier: float,
                       num_blocks_min: int,
                       num_blocks_cap: int) -> int:
    if num_blocks is not None:
        return int(num_blocks)
    # Enough blocks to cover the sequences, plus a multiplier to avoid making
    # block_table lookups trivial while keeping memory bounded.
    needed = int(batch_size * max_num_blocks_per_seq)
    estimate = int(max(num_blocks_min, num_blocks_multiplier * needed))
    return int(min(num_blocks_cap, estimate))


def _maybe_clear_cuda_memory(no_clear_cache: bool) -> None:
    if no_clear_cache:
        return
    gc.collect()
    torch.cuda.empty_cache()


def _build_flash_segment_views(*,
                               k_contig: torch.Tensor,
                               v_contig: torch.Tensor,
                               batch_size: int,
                               seq_len: int,
                               num_kv_heads: int,
                               head_size: int,
                               flash_segments: int,
                               device: str):
    if flash_segments <= 1:
        raise ValueError("flash_segments must be > 1 to build segmented views")
    if seq_len % flash_segments != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by flash_segments ({flash_segments})"
        )
    seg_len = seq_len // flash_segments
    # k_contig is [B*L, Hkv, D]. Reshape to [B, L, Hkv, D] and slice along L.
    k_blhd = k_contig.view(batch_size, seq_len, num_kv_heads, head_size)
    v_blhd = v_contig.view(batch_size, seq_len, num_kv_heads, head_size)
    segments = []
    for i in range(flash_segments):
        s = i * seg_len
        e = (i + 1) * seg_len
        # Ensure contiguous layout for flash kernels.
        k_seg = k_blhd[:, s:e, :, :].contiguous().view(batch_size * seg_len,
                                                       num_kv_heads, head_size)
        v_seg = v_blhd[:, s:e, :, :].contiguous().view(batch_size * seg_len,
                                                       num_kv_heads, head_size)
        cu_seqlens_k_seg = torch.arange(
            0,
            batch_size * seg_len + 1,
            step=seg_len,
            dtype=torch.int32,
            device=device,
        )
        segments.append((k_seg, v_seg, cu_seqlens_k_seg, seg_len))
    return segments


@torch.inference_mode()
def main(
    *,
    paged_version: str,
    batch_size: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    profile: bool,
    kv_cache_dtype: str,
    device: str,
    iters: int,
    warmup: int,
    non_paged: str,
    num_blocks: int | None,
    num_blocks_multiplier: float,
    num_blocks_min: int,
    num_blocks_cap: int,
    causal: bool,
    flash_segments: int,
) -> None:
    if device != "cuda":
        raise ValueError("This benchmark currently supports CUDA only.")

    current_platform.seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    scale = float(1.0 / (head_size**0.5))

    # Q: decode-style (1 token per seq)
    q = torch.empty(batch_size, num_query_heads, head_size, dtype=dtype, device=device)
    q.uniform_(-scale, scale)

    # Sequence lengths (context length per seq).
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    max_seq_len = int(seq_len)

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # --- Paged attention inputs (blocked KV + block tables) ---
    # Default behavior: allocate only as many blocks as needed (times a multiplier)
    # to avoid OOM during sweeps. Use --num-blocks to force the legacy behavior.
    num_blocks = _choose_num_blocks(
        batch_size=batch_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        num_blocks=num_blocks,
        num_blocks_multiplier=num_blocks_multiplier,
        num_blocks_min=num_blocks_min,
        num_blocks_cap=num_blocks_cap,
    )
    # Random block tables: [B, max_num_blocks_per_seq]
    block_tables = torch.randint(
        low=0,
        high=num_blocks,
        size=(batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # --- Non-paged attention inputs (contiguous KV) ---
    # KV: [total_kv, Hkv, D]
    total_kv = batch_size * seq_len
    k_contig = torch.empty(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)
    v_contig = torch.empty_like(k_contig)
    k_contig.uniform_(-scale, scale)
    v_contig.uniform_(-scale, scale)

    # cu_seqlens for varlen
    cu_seqlens_q = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.arange(
        0, total_kv + 1, step=seq_len, dtype=torch.int32, device=device
    )

    # Output buffers
    out_paged = torch.empty_like(q)
    out_flash = torch.empty_like(q)

    # --- Build callables ---
    if paged_version not in ("v1", "v2"):
        raise ValueError("paged_version must be 'v1' or 'v2'")

    if paged_version == "v1":
        def run_paged() -> None:
            ops.paged_attention_v1(
                out_paged,
                q,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                None,  # alibi_slopes
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

    else:
        # V2 needs tmp buffers.
        partition_size = 512  # keep consistent with vllm.attention.ops.paged_attn
        num_partitions = (max_seq_len + partition_size - 1) // partition_size
        tmp_out = torch.empty(
            (batch_size, num_query_heads, num_partitions, head_size),
            dtype=dtype,
            device=device,
        )
        exp_sums = torch.empty(
            (batch_size, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=device,
        )
        max_logits = torch.empty_like(exp_sums)

        def run_paged() -> None:
            ops.paged_attention_v2(
                out_paged,
                exp_sums,
                max_logits,
                tmp_out,
                q,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                None,  # alibi_slopes
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

    def _make_run_non_paged():
        if non_paged not in ("auto", "flash", "sdpa"):
            raise ValueError("--non-paged must be one of: auto|flash|sdpa")

        want_flash = non_paged in ("auto", "flash")
        if want_flash and is_flash_attn_varlen_func_available():
            fa_version = get_flash_attn_version(requires_alibi=False)
            if fa_version is None:
                raise RuntimeError("Failed to resolve flash-attn version.")

            from vllm.attention.utils.fa_utils import flash_attn_varlen_func

            def run_flash() -> tuple[str, callable]:
                def _fn() -> None:
                    flash_attn_varlen_func(
                        q=q,
                        k=k_contig,
                        v=v_contig,
                        out=out_flash,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=1,
                        max_seqlen_k=max_seq_len,
                        softmax_scale=scale,
                        causal=causal,
                        alibi_slopes=None,
                        window_size=None,
                        softcap=0.0,
                        return_softmax_lse=False,
                        fa_version=fa_version,
                    )

                return f"flash_fa{fa_version}_contig", _fn

            return run_flash()

        # Fallback: PyTorch SDPA on contiguous KV.
        # This is still non-paged (contiguous), but may be slower than flash-attn.
        if non_paged == "flash":
            raise RuntimeError(
                "Requested --non-paged=flash, but flash-attn varlen is not available. "
                "Try --non-paged=sdpa or ensure vllm-flash-attn is built."
            )

        # Build K/V as [B, Hq, L, D] to match q [B, Hq, 1, D].
        # k_contig is [B*L, Hkv, D] => [B, L, Hkv, D] => [B, Hkv, L, D]
        k_bhld = k_contig.view(batch_size, seq_len, num_kv_heads, head_size).permute(0, 2, 1, 3)
        v_bhld = v_contig.view(batch_size, seq_len, num_kv_heads, head_size).permute(0, 2, 1, 3)
        if num_query_heads != num_kv_heads:
            repeat = num_query_heads // num_kv_heads
            k_bhld = k_bhld.repeat_interleave(repeat, dim=1)
            v_bhld = v_bhld.repeat_interleave(repeat, dim=1)

        q_bh1d = q.unsqueeze(2)  # [B, Hq, 1, D]

        def run_sdpa() -> None:
            # SDPA returns [B, Hq, 1, D]
            out = F.scaled_dot_product_attention(
                q_bh1d,
                k_bhld,
                v_bhld,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=scale,
            )
            out_flash.copy_(out.squeeze(2))

        return "torch_sdpa_contig", run_sdpa

    non_paged_name, run_flash = _make_run_non_paged()

    # Extra baseline: segmented flash-attn (sum of per-segment runtimes).
    # This is NOT mathematically equivalent to full attention (softmax spans segments),
    # but it approximates the cost of running flash-attn multiple times on shorter KV.
    seg_flash_name = None
    run_flash_segmented = None
    if non_paged_name.startswith("flash_fa") and flash_segments > 1:
        if seq_len % flash_segments != 0:
            raise ValueError(
                f"--flash-segments ({flash_segments}) must divide seq_len ({seq_len})"
            )
        fa_version = get_flash_attn_version(requires_alibi=False)
        if fa_version is None:
            raise RuntimeError("Failed to resolve flash-attn version.")
        from vllm.attention.utils.fa_utils import flash_attn_varlen_func

        segments = _build_flash_segment_views(
            k_contig=k_contig,
            v_contig=v_contig,
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            flash_segments=flash_segments,
            device=device,
        )

        def _fn_seg() -> None:
            for (k_seg, v_seg, cu_seqlens_k_seg, seg_len) in segments:
                flash_attn_varlen_func(
                    q=q,
                    k=k_seg,
                    v=v_seg,
                    out=out_flash,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_seg,
                    max_seqlen_q=1,
                    max_seqlen_k=seg_len,
                    softmax_scale=scale,
                    causal=causal,
                    alibi_slopes=None,
                    window_size=None,
                    softcap=0.0,
                    return_softmax_lse=False,
                    fa_version=fa_version,
                )

        seg_flash_name = f"flash_fa{fa_version}_seg{flash_segments}x{seq_len // flash_segments}"
        run_flash_segmented = _fn_seg

    # --- Warmup ---
    print("Warming up...")
    for _ in range(warmup):
        run_paged()
        run_flash()
        if run_flash_segmented is not None:
            run_flash_segmented()
    torch.cuda.synchronize()

    # --- Benchmark ---
    results: list[BenchResult] = []
    results.append(BenchResult(f"paged_{paged_version}", _cuda_time_us(run_paged, iters, profile)))
    results.append(BenchResult(non_paged_name, _cuda_time_us(run_flash, iters, profile)))
    if run_flash_segmented is not None and seg_flash_name is not None:
        results.append(BenchResult(seg_flash_name, _cuda_time_us(run_flash_segmented, iters, profile)))

    print("\nResults (decode-style):")
    for r in results:
        print(f"  {r.name:24s}: {r.us:9.3f} us")

    paged_us = results[0].us
    flash_us = results[1].us
    print("\nSpeedup:")
    print(f"  flash / paged: {paged_us / flash_us:.3f}x")
    print(f"  paged / flash: {flash_us / paged_us:.3f}x")
    if run_flash_segmented is not None and seg_flash_name is not None:
        seg_us = next(r.us for r in results if r.name == seg_flash_name)
        print(f"  segmented_flash / paged: {paged_us / seg_us:.3f}x")
        print(f"  segmented_flash / flash: {flash_us / seg_us:.3f}x")


@torch.inference_mode()
def sweep(
    *,
    batch_sizes: list[int],
    seq_lens: list[int],
    head_sizes: list[int],
    block_sizes: list[int],
    flash_segments_list: list[int],
    paged_versions: list[str],
    non_paged_impls: list[str],
    num_query_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    seed: int,
    profile: bool,
    kv_cache_dtype: str,
    device: str,
    iters: int,
    warmup: int,
    num_blocks: int | None,
    num_blocks_multiplier: float,
    num_blocks_min: int,
    num_blocks_cap: int,
    skip_unsupported: bool,
    no_clear_cache: bool,
    causal: bool,
    out_csv: str | None,
    out_md: str | None,
) -> None:
    rows: list[dict[str, Any]] = []
    total = 0
    for _ in product(batch_sizes, seq_lens, head_sizes, block_sizes,
                     flash_segments_list):
        total += 1
    print(f"Sweep runs: {total}")

    # Normalize/validate impl lists.
    paged_versions = [v.strip() for v in paged_versions if v.strip()]
    non_paged_impls = [v.strip() for v in non_paged_impls if v.strip()]
    for v in paged_versions:
        if v not in ("v1", "v2"):
            raise ValueError("--sweep-paged-versions must be a subset of: v1,v2")
    for v in non_paged_impls:
        if v not in ("flash", "sdpa"):
            raise ValueError("--sweep-non-paged must be a subset of: flash,sdpa")

    for (batch_size, seq_len, head_size, block_size, flash_segments) in product(
        batch_sizes, seq_lens, head_sizes, block_sizes, flash_segments_list
    ):
        if flash_segments < 1:
            raise ValueError(f"flash_segments must be >= 1, got {flash_segments}")
        if flash_segments > 1 and (seq_len % flash_segments != 0):
            raise ValueError(
                f"In sweep: seq_len ({seq_len}) must be divisible by flash_segments ({flash_segments})"
            )
        if ("v1" in paged_versions or "v2" in paged_versions):
            if head_size not in SUPPORTED_HEAD_SIZES or block_size not in SUPPORTED_BLOCK_SIZES:
                msg = (
                    f"Unsupported paged params: head_size={head_size} block_size={block_size}. "
                    f"Supported head_size={sorted(SUPPORTED_HEAD_SIZES)}, "
                    f"block_size={sorted(SUPPORTED_BLOCK_SIZES)}"
                )
                if skip_unsupported:
                    print(f"\n[skip] {msg}")
                    continue
                raise ValueError(msg)

        print(
            f"\n=== B={batch_size} L={seq_len} "
            f"Hq={num_query_heads} Hkv={num_kv_heads} D={head_size} "
            f"block={block_size} seg={flash_segments} dtype={dtype} "
            f"paged={','.join(paged_versions)} non_paged={','.join(non_paged_impls)} ==="
        )

        # We rebuild tensors per point to avoid cross-point caching artifacts.
        current_platform.seed_everything(seed)

        # Ensure we release memory even if a kernel errors.
        q = seq_lens_t = block_tables = None
        key_caches = value_caches = key_cache = value_cache = None
        k_scale = v_scale = None
        k_contig = v_contig = cu_seqlens_q = cu_seqlens_k = None
        out_paged = out_np = None
        tmp_out = exp_sums = max_logits = None
        k_bhld = v_bhld = q_bh1d = None
        fns = None

        try:
            scale = float(1.0 / (head_size**0.5))
            q = torch.empty(batch_size, num_query_heads, head_size, dtype=dtype, device=device)
            q.uniform_(-scale, scale)
            seq_lens_t = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            max_seq_len = int(seq_len)

            max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
            num_blocks_eff = _choose_num_blocks(
                batch_size=batch_size,
                max_num_blocks_per_seq=max_num_blocks_per_seq,
                num_blocks=num_blocks,
                num_blocks_multiplier=num_blocks_multiplier,
                num_blocks_min=num_blocks_min,
                num_blocks_cap=num_blocks_cap,
            )
            block_tables = torch.randint(
                low=0,
                high=num_blocks_eff,
                size=(batch_size, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=device,
            )
            key_caches, value_caches = create_kv_caches_with_random(
                num_blocks_eff,
                block_size,
                1,
                num_kv_heads,
                head_size,
                kv_cache_dtype,
                dtype,
                device=device,
            )
            key_cache, value_cache = key_caches[0], value_caches[0]
            k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            total_kv = batch_size * seq_len
            k_contig = torch.empty(total_kv, num_kv_heads, head_size, dtype=dtype, device=device)
            v_contig = torch.empty_like(k_contig)
            k_contig.uniform_(-scale, scale)
            v_contig.uniform_(-scale, scale)
            cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.arange(0, total_kv + 1, step=seq_len, dtype=torch.int32, device=device)

            out_paged = torch.empty_like(q)

            # Build callables.
            fns = []

            if "v1" in paged_versions:
                def run_paged_v1() -> None:
                    ops.paged_attention_v1(
                        out_paged,
                        q,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        seq_lens_t,
                        block_size,
                        max_seq_len,
                        None,
                        kv_cache_dtype,
                        k_scale,
                        v_scale,
                    )

                fns.append(("paged_v1", run_paged_v1))

            if "v2" in paged_versions:
                partition_size = 512
                num_partitions = (max_seq_len + partition_size - 1) // partition_size
                tmp_out = torch.empty(
                    (batch_size, num_query_heads, num_partitions, head_size),
                    dtype=dtype,
                    device=device,
                )
                exp_sums = torch.empty(
                    (batch_size, num_query_heads, num_partitions),
                    dtype=torch.float32,
                    device=device,
                )
                max_logits = torch.empty_like(exp_sums)

                def run_paged_v2() -> None:
                    ops.paged_attention_v2(
                        out_paged,
                        exp_sums,
                        max_logits,
                        tmp_out,
                        q,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        seq_lens_t,
                        block_size,
                        max_seq_len,
                        None,
                        kv_cache_dtype,
                        k_scale,
                        v_scale,
                    )

                fns.append(("paged_v2", run_paged_v2))

            out_np = torch.empty_like(q)

            if "flash" in non_paged_impls:
                if is_flash_attn_varlen_func_available():
                    fa_version = get_flash_attn_version(requires_alibi=False)
                    if fa_version is None:
                        raise RuntimeError("Failed to resolve flash-attn version.")
                    from vllm.attention.utils.fa_utils import flash_attn_varlen_func

                    def run_flash() -> None:
                        flash_attn_varlen_func(
                            q=q,
                            k=k_contig,
                            v=v_contig,
                            out=out_np,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_k=cu_seqlens_k,
                            max_seqlen_q=1,
                            max_seqlen_k=max_seq_len,
                            softmax_scale=scale,
                            causal=causal,
                            alibi_slopes=None,
                            window_size=None,
                            softcap=0.0,
                            return_softmax_lse=False,
                            fa_version=fa_version,
                        )

                    fns.append((f"flash_fa{fa_version}", run_flash))

                    # Optional segmented flash baseline.
                    if flash_segments > 1:
                        segments = _build_flash_segment_views(
                            k_contig=k_contig,
                            v_contig=v_contig,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            flash_segments=flash_segments,
                            device=device,
                        )

                        def run_flash_segmented() -> None:
                            for (k_seg, v_seg, cu_seqlens_k_seg, seg_len) in segments:
                                flash_attn_varlen_func(
                                    q=q,
                                    k=k_seg,
                                    v=v_seg,
                                    out=out_np,
                                    cu_seqlens_q=cu_seqlens_q,
                                    cu_seqlens_k=cu_seqlens_k_seg,
                                    max_seqlen_q=1,
                                    max_seqlen_k=seg_len,
                                    softmax_scale=scale,
                                    causal=causal,
                                    alibi_slopes=None,
                                    window_size=None,
                                    softcap=0.0,
                                    return_softmax_lse=False,
                                    fa_version=fa_version,
                                )

                        fns.append((f"flash_fa{fa_version}_seg{flash_segments}x{seq_len // flash_segments}", run_flash_segmented))
                else:
                    print("flash-attn varlen not available; will record flash as NA")

            if "sdpa" in non_paged_impls:
                k_bhld = k_contig.view(batch_size, seq_len, num_kv_heads, head_size).permute(0, 2, 1, 3)
                v_bhld = v_contig.view(batch_size, seq_len, num_kv_heads, head_size).permute(0, 2, 1, 3)
                if num_query_heads != num_kv_heads:
                    repeat = num_query_heads // num_kv_heads
                    k_bhld = k_bhld.repeat_interleave(repeat, dim=1)
                    v_bhld = v_bhld.repeat_interleave(repeat, dim=1)
                q_bh1d = q.unsqueeze(2)

                def run_sdpa() -> None:
                    out = F.scaled_dot_product_attention(
                        q_bh1d,
                        k_bhld,
                        v_bhld,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=scale,
                    )
                    out_np.copy_(out.squeeze(2))

                fns.append(("torch_sdpa", run_sdpa))

            if not fns:
                raise RuntimeError("No benchmark functions selected in sweep")

            print("Warming up...")
            for _ in range(warmup):
                for _, fn in fns:
                    fn()
            torch.cuda.synchronize()

            times: dict[str, float] = {}
            for name, fn in fns:
                us = _cuda_time_us(fn, iters, profile)
                times[name] = us
                print(f"{name}: {us:.3f} us")

            paged_v1_us = times.get("paged_v1")
            paged_v2_us = times.get("paged_v2")
            flash_key = next((k for k in times.keys() if k.startswith("flash_fa")), None)
            flash_us = times.get(flash_key) if flash_key else None
            seg_flash_key = next((k for k in times.keys() if "_seg" in k and k.startswith("flash_fa")), None)
            seg_flash_us = times.get(seg_flash_key) if seg_flash_key else None
            sdpa_us = times.get("torch_sdpa")

            def fmt(x: float | None) -> str:
                return "" if x is None else f"{x:.3f}"

            def ratio(num: float | None, den: float | None) -> str:
                if num is None or den is None:
                    return ""
                return f"{num / den:.3f}"

            row: dict[str, Any] = {
                "dtype": str(dtype).replace("torch.", ""),
                "kv_cache_dtype": kv_cache_dtype,
                "B": batch_size,
                "L": seq_len,
                "Hq": num_query_heads,
                "Hkv": num_kv_heads,
                "D": head_size,
                "block": block_size,
                "flash_segments": flash_segments,
                "paged_v1_us": fmt(paged_v1_us),
                "paged_v2_us": fmt(paged_v2_us),
                "flash_us": fmt(flash_us),
                "flash_seg_us": fmt(seg_flash_us),
                "sdpa_us": fmt(sdpa_us),
                "flash_over_paged_v1": ratio(paged_v1_us, flash_us),
                "flash_over_paged_v2": ratio(paged_v2_us, flash_us),
                "flash_seg_over_paged_v1": ratio(paged_v1_us, seg_flash_us),
                "flash_seg_over_paged_v2": ratio(paged_v2_us, seg_flash_us),
                "sdpa_over_paged_v1": ratio(paged_v1_us, sdpa_us),
                "sdpa_over_paged_v2": ratio(paged_v2_us, sdpa_us),
                "paged_v2_over_v1": ratio(paged_v1_us, paged_v2_us),
            }
            if flash_key:
                row["flash_impl"] = flash_key
            if seg_flash_key:
                row["flash_seg_impl"] = seg_flash_key
            rows.append(row)
        finally:
            # Drop references to large CUDA tensors before the next sweep point.
            q = seq_lens_t = block_tables = None
            key_caches = value_caches = key_cache = value_cache = None
            k_scale = v_scale = None
            k_contig = v_contig = cu_seqlens_q = cu_seqlens_k = None
            out_paged = out_np = None
            tmp_out = exp_sums = max_logits = None
            k_bhld = v_bhld = q_bh1d = None
            fns = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _maybe_clear_cuda_memory(no_clear_cache)

    csv_text = _format_csv(rows)
    md_columns = [
        "B", "L", "Hq", "Hkv", "D", "block", "flash_segments",
        "paged_v1_us", "paged_v2_us", "flash_us", "flash_seg_us", "sdpa_us",
        "paged_v2_over_v1", "flash_over_paged_v2", "flash_seg_over_paged_v2", "sdpa_over_paged_v2",
    ]
    md_text = _format_markdown(rows, md_columns)

    if out_csv:
        _write_text(out_csv, csv_text)
        print(f"\nWrote CSV: {out_csv}")
    else:
        print("\nCSV:\n" + csv_text)

    if out_md:
        _write_text(out_md, md_text)
        print(f"Wrote Markdown: {out_md}")
    else:
        print("Markdown:\n" + md_text)


if __name__ == "__main__":
    # python benchmarks/kernels/benchmark_paged_vs_flash_attention.py --sweep --sweep-paged-versions v1,v2 --sweep-non-paged flash,sdpa --sweep-batch-sizes 8 --sweep-seq-lens 1024 --sweep-head-sizes 128 --sweep-block-sizes 16 --num-query-heads 64 --num-kv-heads 8 --dtype half --iters 50 --warmup 5 --out-csv /tmp/paged_vs_nonpaged_all.csv --out-md /tmp/paged_vs_nonpaged_all.md

    parser = FlexibleArgumentParser(
        description="Benchmark paged attention vs non-paged (contiguous KV) flash attention."
    )
    parser.add_argument("--paged-version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[32, 64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="half",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="Run a parameter sweep and emit CSV/Markdown.")
    parser.add_argument("--sweep-paged-versions", type=str, default="v2",
                        help="Comma-separated list, e.g. 'v1,v2'.")
    parser.add_argument("--sweep-non-paged", type=str, default="flash,sdpa",
                        help="Comma-separated list: 'flash,sdpa'.")
    parser.add_argument("--sweep-batch-sizes", type=str, default="8",
                        help="Comma-separated list, e.g. '1,2,4,8,16'.")
    parser.add_argument("--sweep-seq-lens", type=str, default="512,1024,2048,4096,8192",
                        help="Comma-separated list.")
    parser.add_argument("--sweep-head-sizes", type=str, default="128",
                        help="Comma-separated list, e.g. '64,128'.")
    parser.add_argument("--sweep-block-sizes", type=str, default="16",
                        help="Comma-separated list, e.g. '8,16,32'.")
    parser.add_argument(
        "--sweep-flash-segments",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of flash segment counts to sweep. "
            "If omitted/empty, uses --flash-segments as a single value."
        ),
    )
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Write CSV to this path (default: print to stdout).")
    parser.add_argument("--out-md", type=str, default=None,
                        help="Write Markdown table to this path (default: print to stdout).")
    parser.add_argument(
        "--non-paged",
        type=str,
        choices=["auto", "flash", "sdpa"],
        default="auto",
        help="Non-paged baseline: auto prefers flash-attn then falls back to torch SDPA.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help=(
            "Use causal masking for non-paged baselines. Default is non-causal "
            "(recommended for decode since K/V are past-only)."
        ),
    )
    parser.add_argument(
        "--flash-segments",
        type=int,
        default=1,
        help=(
            "If > 1, additionally benchmark segmented FlashAttention by splitting the KV "
            "sequence into N equal segments and summing the time of N flash-attn calls. "
            "(Requires seq_len divisible by N.)"
        ),
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help=(
            "Number of KV-cache blocks for paged attention. If unset, a safe "
            "value is derived from B and seq_len to reduce OOM risk."
        ),
    )
    parser.add_argument(
        "--num-blocks-multiplier",
        type=float,
        default=4.0,
        help="When --num-blocks is unset: num_blocks ~= multiplier * B * ceil(L/block_size).",
    )
    parser.add_argument(
        "--num-blocks-min",
        type=int,
        default=1024,
        help="Lower bound for derived num_blocks when --num-blocks is unset.",
    )
    parser.add_argument(
        "--num-blocks-cap",
        type=int,
        default=128 * 1024,
        help="Upper bound for derived num_blocks when --num-blocks is unset.",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="In sweep: skip points with unsupported paged head_size/block_size instead of failing.",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="In sweep: do not call gc.collect()/torch.cuda.empty_cache() between points.",
    )
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    if args.sweep:
        paged_versions = _parse_csv_strs(args.sweep_paged_versions)
        non_paged_impls = _parse_csv_strs(args.sweep_non_paged)
        batch_sizes = _parse_csv_ints(args.sweep_batch_sizes)
        seq_lens = _parse_csv_ints(args.sweep_seq_lens)
        head_sizes = _parse_csv_ints(args.sweep_head_sizes)
        block_sizes = _parse_csv_ints(args.sweep_block_sizes)
        flash_segments_list = _parse_csv_ints(args.sweep_flash_segments) if args.sweep_flash_segments.strip() else [int(args.flash_segments)]
        if not paged_versions:
            raise ValueError("--sweep-paged-versions is empty")
        if not non_paged_impls:
            raise ValueError("--sweep-non-paged is empty")
        if not batch_sizes or not seq_lens or not head_sizes or not block_sizes or not flash_segments_list:
            raise ValueError("Sweep lists must be non-empty")
        sweep(
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            head_sizes=head_sizes,
            block_sizes=block_sizes,
            flash_segments_list=flash_segments_list,
            paged_versions=paged_versions,
            non_paged_impls=non_paged_impls,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            profile=args.profile,
            kv_cache_dtype=args.kv_cache_dtype,
            device="cuda",
            iters=args.iters,
            warmup=args.warmup,
            num_blocks=args.num_blocks,
            num_blocks_multiplier=args.num_blocks_multiplier,
            num_blocks_min=args.num_blocks_min,
            num_blocks_cap=args.num_blocks_cap,
            skip_unsupported=args.skip_unsupported,
            no_clear_cache=args.no_clear_cache,
            causal=args.causal,
            out_csv=args.out_csv,
            out_md=args.out_md,
        )
    else:
        main(
            paged_version=args.paged_version,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            profile=args.profile,
            kv_cache_dtype=args.kv_cache_dtype,
            device="cuda",
            iters=args.iters,
            warmup=args.warmup,
            non_paged=args.non_paged,
            num_blocks=args.num_blocks,
            num_blocks_multiplier=args.num_blocks_multiplier,
            num_blocks_min=args.num_blocks_min,
            num_blocks_cap=args.num_blocks_cap,
            causal=args.causal,
            flash_segments=args.flash_segments,
        )
