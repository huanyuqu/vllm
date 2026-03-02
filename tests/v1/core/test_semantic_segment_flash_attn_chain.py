# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import pytest
import torch

from vllm.platforms import current_platform
from vllm.vllm_flash_attn import (
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    is_fa_version_supported,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from tests.kernels.attention.test_flash_attn import ref_paged_attn
from tests.kernels.attention.test_segmented_flash_attn import (
    _fa2_varlen_fwd_supports_segmented,
)
from tests.v1.core.utils import create_requests


def _expand_to_atomic_page_ids(
    *,
    block_id: int,
    relative_id: int,
    size: int,
    atomic_block_size: int,
    max_block_size: int,
) -> list[int]:
    start_addr = block_id * max_block_size + relative_id * size
    assert start_addr % atomic_block_size == 0
    start_page = start_addr // atomic_block_size
    num_pages = size // atomic_block_size
    return [start_page + i for i in range(num_pages)]


def _logical_page_ids_for_request(
    *,
    segments,
    atomic_block_size: int,
    max_block_size: int,
) -> list[int]:
    page_ids: list[int] = []
    for seg in segments:
        for blk in seg.blocks:
            page_ids.extend(
                _expand_to_atomic_page_ids(
                    block_id=blk.block_id,
                    relative_id=blk.relative_id,
                    size=blk.size,
                    atomic_block_size=atomic_block_size,
                    max_block_size=max_block_size,
                ))
    return page_ids


def _fill_pages_in_order(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_ids: list[int],
    logical_k: torch.Tensor,
    logical_v: torch.Tensor,
    start_token: int,
) -> None:
    block_size = key_cache.shape[1]
    for i, page_id in enumerate(page_ids):
        s = start_token + i * block_size
        e = s + block_size
        key_cache[page_id].copy_(logical_k[s:e])
        value_cache[page_id].copy_(logical_v[s:e])


def _apply_moves_swaps(
    *,
    kv_cache: torch.Tensor,
    moves: list[tuple[int, int, int, int]],
    swaps: list[tuple[int, int, int, int]],
) -> None:
    # kv_cache: [2, num_pages, block_size, num_kv_heads, head_size]
    key_cache, value_cache = kv_cache.unbind(0)
    k_flat = key_cache.reshape(-1, *key_cache.shape[2:])
    v_flat = value_cache.reshape(-1, *value_cache.shape[2:])
    for group_id, src_addr, dst_addr, size in moves:
        assert group_id == 0
        k_flat[dst_addr : dst_addr + size].copy_(k_flat[src_addr : src_addr + size])
        v_flat[dst_addr : dst_addr + size].copy_(v_flat[src_addr : src_addr + size])
    for group_id, addr1, addr2, size in swaps:
        assert group_id == 0
        temp_k = k_flat[addr1 : addr1 + size].clone()
        temp_v = v_flat[addr1 : addr1 + size].clone()
        k_flat[addr1 : addr1 + size].copy_(k_flat[addr2 : addr2 + size])
        v_flat[addr1 : addr1 + size].copy_(v_flat[addr2 : addr2 + size])
        k_flat[addr2 : addr2 + size].copy_(temp_k)
        v_flat[addr2 : addr2 + size].copy_(temp_v)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@torch.inference_mode()
def test_semantic_segment_seal_consolidate_drives_segmented_flashattn() -> None:
    torch.set_default_device("cuda")

    fa_version = 2
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    sig = inspect.signature(flash_attn_varlen_func)
    if "segment_num" not in sig.parameters:
        pytest.skip("flash_attn_varlen_func does not support segmented attention")
    if not _fa2_varlen_fwd_supports_segmented():
        pytest.skip(
            "Loaded torch.ops._vllm_fa2_C.varlen_fwd schema does not expose "
            "segment_* arguments; likely still using an old _vllm_fa2_C.abi3.so"
        )

    # Import backend lazily to avoid importing CUDA-heavy modules on skip.
    from vllm.attention.backends.abstract import AttentionType
    from vllm.v1.attention.backends import flash_attn as fa_backend

    current_platform.seed_everything(0)

    # Small, deterministic shapes.
    query_len = 4
    # FIXME: Use lengths that avoid triggering the segmented-hybrid block-internal
    # segment-boundary behavior (segment lengths aligned to common kBlockN).
    prefix_len = 256
    tail_len = 128
    kv_len = prefix_len + tail_len
    atomic_block_size = 16
    supported_block_sizes = [atomic_block_size]
    max_block_size = max(supported_block_sizes)
    num_max_blocks = 64

    num_query_heads = 4
    num_kv_heads = 4
    head_size = 64
    dtype = torch.bfloat16
    scale = head_size**-0.5

    # KV cache expected by FlashAttentionImpl: [2, num_pages, block_size, Hkv, D]
    kv_cache_for_impl = torch.empty(
        (2, num_max_blocks, atomic_block_size, num_kv_heads, head_size),
        dtype=dtype,
    )
    key_cache, value_cache = kv_cache_for_impl.unbind(0)

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
    logical_k = torch.randn(kv_len, num_kv_heads, head_size, dtype=dtype)
    logical_v = torch.randn_like(logical_k)

    # Semantic segment manager wiring via KVCacheManager.
    kv_cache_config = KVCacheConfig(
        num_blocks=num_max_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.0"],
                FullAttentionSpec(
                    block_size=atomic_block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )
    kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=kv_len,
        enable_caching=False,
        enable_semantic_segment=True,
        supported_block_sizes=supported_block_sizes,
    )

    # Create two requests. Request B is a short-lived allocator to fragment
    # memory so consolidate triggers real moves.
    req_a, req_b = create_requests(
        num_requests=2,
        num_tokens=kv_len,
        block_size=atomic_block_size,
        req_ids=["req_a", "req_b"],
    )

    # Allocate A for a prefix in 2 parts with B in-between to create a hole.
    # This makes A's sealed segment non-contiguous and forces consolidate to
    # generate move ops.
    a_first = 160
    b_alloc = 32
    a_second = prefix_len - a_first
    assert a_second > 0
    blocks_a_first = kv_cache_manager.allocate_segment(req_a, num_new_tokens=a_first)
    assert blocks_a_first is not None
    assert len(blocks_a_first) == a_first // atomic_block_size

    blocks_b = kv_cache_manager.allocate_segment(req_b, num_new_tokens=b_alloc)
    assert blocks_b is not None
    assert len(blocks_b) == b_alloc // atomic_block_size

    blocks_a_second = kv_cache_manager.allocate_segment(req_a, num_new_tokens=a_second)
    assert blocks_a_second is not None
    assert len(blocks_a_second) == a_second // atomic_block_size

    # Fill A's first 48 tokens into the KV cache according to its *current*
    # logical block order (pre-consolidation).
    segs_a_pre = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    page_ids_a_pre = _logical_page_ids_for_request(
        segments=segs_a_pre,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    assert len(page_ids_a_pre) == (prefix_len // atomic_block_size)
    _fill_pages_in_order(
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids_a_pre,
        logical_k=logical_k,
        logical_v=logical_v,
        start_token=0,
    )

    # Make sealing succeed without requiring the full prefix-cache hashing path.
    # SemanticSegmentManager.seal_segment only requires the tail block hash.
    segs_a_obj = kv_cache_manager.coordinator.single_type_managers[0].req_to_segments[
        req_a.request_id
    ]
    tail = segs_a_obj.unsealed_segment.tail
    assert tail is not None
    tail.block_hash = make_block_hash_with_group_id(BlockHash(b"x" * 32), 0)

    kv_cache_manager.coordinator.seal_segment(req_a.request_id)

    # Freeing a request seals its unsealed segment; ensure the tail has a hash
    # to satisfy SemanticSegmentManager.seal_segment.
    segs_b_obj = kv_cache_manager.coordinator.single_type_managers[0].req_to_segments[
        req_b.request_id
    ]
    tail_b = segs_b_obj.unsealed_segment.tail
    assert tail_b is not None
    tail_b.block_hash = make_block_hash_with_group_id(BlockHash(b"y" * 32), 0)

    # Free B to create space for consolidation to pack A.
    kv_cache_manager.free(req_b)

    kv_cache_manager.coordinator.consolidate_segment_memory(req_a.request_id)
    moves, swaps = kv_cache_manager.get_pending_moves()
    assert moves or swaps, "Expected consolidate to generate move/swap ops"

    _apply_moves_swaps(kv_cache=kv_cache_for_impl, moves=moves, swaps=swaps)
    # Simulate next-step bookkeeping (free moved-out blocks).
    kv_cache_manager.get_pending_moves()

    # Sanity: after consolidation ops, the prefix tokens should be readable
    # in the post-consolidation logical block order and match what we filled.
    # This isolates layout/move/swap issues from attention-kernel issues.
    segs_a_mid = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    page_ids_a_mid = _logical_page_ids_for_request(
        segments=segs_a_mid,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    # At this point we only have prefix_len tokens allocated for A.
    assert len(page_ids_a_mid) == (prefix_len // atomic_block_size)
    gathered_k_prefix = key_cache[page_ids_a_mid].view(-1, num_kv_heads, head_size)[:prefix_len]
    gathered_v_prefix = value_cache[page_ids_a_mid].view(-1, num_kv_heads, head_size)[:prefix_len]
    assert torch.equal(gathered_k_prefix, logical_k[:prefix_len])
    assert torch.equal(gathered_v_prefix, logical_v[:prefix_len])

    # Allocate the trailing (paged) tail_len tokens for A as a new, unsealed segment.
    assert kv_cache_manager.allocate_segment(req_a, num_new_tokens=tail_len) is not None

    # Fill tokens [prefix_len:kv_len] according to *post-consolidation* + tail allocation order.
    segs_a_post = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    page_ids_a_post = _logical_page_ids_for_request(
        segments=segs_a_post,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    assert len(page_ids_a_post) == (kv_len // atomic_block_size)
    _fill_pages_in_order(
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids_a_post[(prefix_len // atomic_block_size):],
        logical_k=logical_k,
        logical_v=logical_v,
        start_token=prefix_len,
    )

    # Sanity: full gather through block_table matches logical KV.
    gathered_k = key_cache[page_ids_a_post].view(-1, num_kv_heads, head_size)[:kv_len]
    gathered_v = value_cache[page_ids_a_post].view(-1, num_kv_heads, head_size)[:kv_len]
    assert torch.equal(gathered_k, logical_k)
    assert torch.equal(gathered_v, logical_v)

    # Build segment metadata like GPUModelRunner: only sealed+consolidated.
    sealed = [s for s in segs_a_post if s.is_sealed and s.is_consolidated]
    assert len(sealed) == 1
    sealed_seg = sealed[0]
    assert sealed_seg.head is not None

    start_token_idx = sealed_seg.head.block_id * max_block_size
    start_token_idx += sealed_seg.head.relative_id * sealed_seg.head.size
    seg_len = sealed_seg.capacity
    assert seg_len == prefix_len

    # Sanity: the sealed prefix is physically contiguous starting at
    # start_token_idx in the flattened cache, and matches logical KV.
    k_flat_dbg = key_cache.view(-1, num_kv_heads, head_size)
    v_flat_dbg = value_cache.view(-1, num_kv_heads, head_size)
    assert torch.equal(k_flat_dbg[start_token_idx : start_token_idx + seg_len], logical_k[:seg_len])
    assert torch.equal(v_flat_dbg[start_token_idx : start_token_idx + seg_len], logical_v[:seg_len])

    full_block_table = torch.tensor([page_ids_a_post], dtype=torch.int32)
    num_segments = torch.tensor([2], dtype=torch.int32)
    segment_start_indices = torch.tensor(
        [[start_token_idx, 0]], dtype=torch.int64
    )
    segment_lens = torch.tensor(
        [[seg_len, kv_len - seg_len]], dtype=torch.int32
    )
    segment_block_table = torch.tensor(
        [[full_block_table[0, -1].item()]], dtype=torch.int32
    )

    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32)
    slot_mapping = torch.empty((query_len,), dtype=torch.int32)

    attn_metadata = fa_backend.FlashAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=query_start_loc,
        max_seq_len=kv_len,
        seq_lens=seq_lens,
        block_table=full_block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        segment_lens=segment_lens,
        segment_block_table=segment_block_table,
        segment_start_indices=segment_start_indices,
        num_segments=num_segments,
        causal=True,
    )

    class _DummyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale_shape = (1, num_kv_heads)
            self._q_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._k_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._v_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._share_kv_cache = True

    layer = _DummyLayer()

    called: dict[str, bool] = {"segmented": False}
    orig_func = fa_backend.flash_attn_varlen_func

    def _wrapped_flash_attn_varlen_func(*args, **kwargs):
        if "segment_num" in kwargs:
            called["segmented"] = True
        return orig_func(*args, **kwargs)

    fa_backend.flash_attn_varlen_func = _wrapped_flash_attn_varlen_func  # type: ignore[assignment]
    try:
        impl = fa_backend.FlashAttentionImpl(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            sinks=None,
        )

        out = torch.empty_like(query)
        impl.forward(
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=kv_cache_for_impl,
            attn_metadata=attn_metadata,
            output=out,
        )
    finally:
        fa_backend.flash_attn_varlen_func = orig_func  # type: ignore[assignment]

    assert called["segmented"], "Expected FlashAttentionImpl to pass segment_* args"

    # NOTE: ref_paged_attn scales `query` in-place; clone to avoid affecting
    # subsequent kernel calls and comparisons.
    query_ref = query.clone()
    ref_out = ref_paged_attn(
        query=query_ref,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=full_block_table,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
    )

    # Also compare against the non-segmented FA2 paged kernel to separate
    # potential kernel-vs-python-reference drift from segmented-hybrid issues.
    out_paged = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out_paged,
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=full_block_table,
        softcap=0,
        fa_version=fa_version,
    )

    torch.testing.assert_close(out_paged, ref_out, atol=2e-2, rtol=2e-2)

    # Direct segmented-hybrid call (bypassing FlashAttentionImpl's metadata
    # packing) to isolate whether the mismatch is in kernel usage.
    k_flat_direct = key_cache.view(-1, num_kv_heads, head_size)
    v_flat_direct = value_cache.view(-1, num_kv_heads, head_size)
    sealed_k_ptr = k_flat_direct[start_token_idx].data_ptr()
    sealed_v_ptr = v_flat_direct[start_token_idx].data_ptr()
    tail_page_ids = page_ids_a_post[(prefix_len // atomic_block_size):]
    block_table_paged_direct = torch.tensor([tail_page_ids], dtype=torch.int32)
    out_segmented_direct = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out_segmented_direct,
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table_paged_direct,
        softcap=0,
        segment_num=torch.tensor([2], dtype=torch.int32),
        segment_lens=torch.tensor([[seg_len, kv_len - seg_len]], dtype=torch.int32),
        segment_k_ptrs=torch.tensor([[sealed_k_ptr, 0]], dtype=torch.int64),
        segment_v_ptrs=torch.tensor([[sealed_v_ptr, 0]], dtype=torch.int64),
        fa_version=fa_version,
    )

    torch.testing.assert_close(
        out_segmented_direct, out_paged, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(out, out_paged, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@torch.inference_mode()
def test_semantic_segment_consolidate_with_live_request_triggers_swaps() -> None:
    torch.set_default_device("cuda")
    fa_version = 2
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    sig = inspect.signature(flash_attn_varlen_func)
    if "segment_num" not in sig.parameters:
        pytest.skip("flash_attn_varlen_func does not support segmented attention")
    if not _fa2_varlen_fwd_supports_segmented():
        pytest.skip(
            "Loaded torch.ops._vllm_fa2_C.varlen_fwd schema does not expose "
            "segment_* arguments; likely still using an old _vllm_fa2_C.abi3.so"
        )

    from vllm.attention.backends.abstract import AttentionType
    from vllm.v1.attention.backends import flash_attn as fa_backend

    current_platform.seed_everything(0)

    query_len = 4
    atomic_block_size = 16
    supported_block_sizes = [atomic_block_size]
    max_block_size = max(supported_block_sizes)
    num_max_blocks = 64

    prefix_len = 256
    tail_len = 128
    kv_len = prefix_len + tail_len
    b_alloc = 32
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 64
    dtype = torch.bfloat16
    scale = head_size**-0.5

    kv_cache = torch.empty(
        (2, num_max_blocks, atomic_block_size, num_kv_heads, head_size),
        dtype=dtype,
    )
    key_cache, value_cache = kv_cache.unbind(0)

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)

    # Request-local logical KV payloads used to verify post-swap correctness.
    logical_k_a = torch.randn(kv_len, num_kv_heads, head_size, dtype=dtype)
    logical_v_a = torch.randn_like(logical_k_a)
    logical_k_b = torch.randn(b_alloc, num_kv_heads, head_size, dtype=dtype)
    logical_v_b = torch.randn_like(logical_k_b)

    kv_cache_config = KVCacheConfig(
        num_blocks=num_max_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer.0"],
                FullAttentionSpec(
                    block_size=atomic_block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )
    kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=kv_len,
        enable_caching=False,
        enable_semantic_segment=True,
        supported_block_sizes=supported_block_sizes,
    )

    req_a, req_b = create_requests(
        num_requests=2,
        num_tokens=kv_len,
        block_size=atomic_block_size,
        req_ids=["req_a_swap", "req_b_swap"],
    )

    # Create fragmented A layout by placing B in the middle.
    a_first = 160
    a_second = prefix_len - a_first
    assert a_second > 0
    assert kv_cache_manager.allocate_segment(req_a, num_new_tokens=a_first) is not None
    assert kv_cache_manager.allocate_segment(req_b, num_new_tokens=b_alloc) is not None
    assert kv_cache_manager.allocate_segment(req_a, num_new_tokens=a_second) is not None

    segs_a_pre = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    segs_b_pre = kv_cache_manager.get_segments(req_b.request_id).multi_group_segments[0]
    page_ids_a_pre = _logical_page_ids_for_request(
        segments=segs_a_pre,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    page_ids_b_pre = _logical_page_ids_for_request(
        segments=segs_b_pre,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    assert len(page_ids_a_pre) == (prefix_len // atomic_block_size)
    assert len(page_ids_b_pre) == (b_alloc // atomic_block_size)

    _fill_pages_in_order(
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids_a_pre,
        logical_k=logical_k_a,
        logical_v=logical_v_a,
        start_token=0,
    )
    _fill_pages_in_order(
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids_b_pre,
        logical_k=logical_k_b,
        logical_v=logical_v_b,
        start_token=0,
    )

    # Seal A only; keep B alive so A's consolidation needs swaps.
    segs_a_obj = kv_cache_manager.coordinator.single_type_managers[0].req_to_segments[
        req_a.request_id
    ]
    tail_a = segs_a_obj.unsealed_segment.tail
    assert tail_a is not None
    tail_a.block_hash = make_block_hash_with_group_id(BlockHash(b"z" * 32), 0)
    kv_cache_manager.coordinator.seal_segment(req_a.request_id)

    kv_cache_manager.coordinator.consolidate_segment_memory(req_a.request_id)
    moves, swaps = kv_cache_manager.get_pending_moves()

    # With B still occupying the middle hole, consolidation should require swaps.
    assert swaps, "Expected consolidate to generate swap ops with live request B"
    assert not moves, "Did not expect consolidate to generate move ops with live request B"

    _apply_moves_swaps(kv_cache=kv_cache, moves=moves, swaps=swaps)
    kv_cache_manager.get_pending_moves()

    segs_a_post = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    segs_b_post = kv_cache_manager.get_segments(req_b.request_id).multi_group_segments[0]
    page_ids_a_post = _logical_page_ids_for_request(
        segments=segs_a_post,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    page_ids_b_post = _logical_page_ids_for_request(
        segments=segs_b_post,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )

    gathered_k_a = key_cache[page_ids_a_post].view(-1, num_kv_heads, head_size)[:prefix_len]
    gathered_v_a = value_cache[page_ids_a_post].view(-1, num_kv_heads, head_size)[:prefix_len]
    gathered_k_b = key_cache[page_ids_b_post].view(-1, num_kv_heads, head_size)[:b_alloc]
    gathered_v_b = value_cache[page_ids_b_post].view(-1, num_kv_heads, head_size)[:b_alloc]
    assert torch.equal(gathered_k_a, logical_k_a[:prefix_len])
    assert torch.equal(gathered_v_a, logical_v_a[:prefix_len])
    assert torch.equal(gathered_k_b, logical_k_b)
    assert torch.equal(gathered_v_b, logical_v_b)

    # Add an unsealed tail for A and verify full A gather still matches.
    assert kv_cache_manager.allocate_segment(req_a, num_new_tokens=tail_len) is not None

    segs_a_final = kv_cache_manager.get_segments(req_a.request_id).multi_group_segments[0]
    page_ids_a_final = _logical_page_ids_for_request(
        segments=segs_a_final,
        atomic_block_size=atomic_block_size,
        max_block_size=max_block_size,
    )
    assert len(page_ids_a_final) == (kv_len // atomic_block_size)
    _fill_pages_in_order(
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids_a_final[(prefix_len // atomic_block_size):],
        logical_k=logical_k_a,
        logical_v=logical_v_a,
        start_token=prefix_len,
    )

    gathered_k_a_full = key_cache[page_ids_a_final].view(-1, num_kv_heads, head_size)[:kv_len]
    gathered_v_a_full = value_cache[page_ids_a_final].view(-1, num_kv_heads, head_size)[:kv_len]
    assert torch.equal(gathered_k_a_full, logical_k_a)
    assert torch.equal(gathered_v_a_full, logical_v_a)

    # Build segmented attention metadata and verify final outputs match.
    sealed = [s for s in segs_a_final if s.is_sealed and s.is_consolidated]
    assert len(sealed) == 1
    sealed_seg = sealed[0]
    assert sealed_seg.head is not None

    start_token_idx = sealed_seg.head.block_id * max_block_size
    start_token_idx += sealed_seg.head.relative_id * sealed_seg.head.size
    seg_len = sealed_seg.capacity
    assert seg_len == prefix_len

    full_block_table = torch.tensor([page_ids_a_final], dtype=torch.int32)
    num_segments = torch.tensor([2], dtype=torch.int32)
    segment_start_indices = torch.tensor(
        [[start_token_idx, 0]], dtype=torch.int64
    )
    segment_lens = torch.tensor(
        [[seg_len, kv_len - seg_len]], dtype=torch.int32
    )
    segment_block_table = torch.tensor(
        [[full_block_table[0, -1].item()]], dtype=torch.int32
    )

    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32)
    slot_mapping = torch.empty((query_len,), dtype=torch.int32)

    attn_metadata = fa_backend.FlashAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        query_start_loc=query_start_loc,
        max_seq_len=kv_len,
        seq_lens=seq_lens,
        block_table=full_block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        segment_lens=segment_lens,
        segment_block_table=segment_block_table,
        segment_start_indices=segment_start_indices,
        num_segments=num_segments,
        causal=True,
    )

    class _DummyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale_shape = (1, num_kv_heads)
            self._q_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._k_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._v_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._share_kv_cache = True

    layer = _DummyLayer()

    called: dict[str, bool] = {"segmented": False}
    orig_func = fa_backend.flash_attn_varlen_func

    def _wrapped_flash_attn_varlen_func(*args, **kwargs):
        if "segment_num" in kwargs:
            called["segmented"] = True
        return orig_func(*args, **kwargs)

    fa_backend.flash_attn_varlen_func = _wrapped_flash_attn_varlen_func  # type: ignore[assignment]
    try:
        impl = fa_backend.FlashAttentionImpl(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            sinks=None,
        )

        out = torch.empty_like(query)
        impl.forward(
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=out,
        )
    finally:
        fa_backend.flash_attn_varlen_func = orig_func  # type: ignore[assignment]

    assert called["segmented"], "Expected FlashAttentionImpl to pass segment_* args"

    query_ref = query.clone()
    ref_out = ref_paged_attn(
        query=query_ref,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=full_block_table,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
    )

    out_paged = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out_paged,
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=full_block_table,
        softcap=0,
        fa_version=fa_version,
    )
    torch.testing.assert_close(out_paged, ref_out, atol=2e-2, rtol=2e-2)

    k_flat_direct = key_cache.view(-1, num_kv_heads, head_size)
    v_flat_direct = value_cache.view(-1, num_kv_heads, head_size)
    sealed_k_ptr = k_flat_direct[start_token_idx].data_ptr()
    sealed_v_ptr = v_flat_direct[start_token_idx].data_ptr()
    tail_page_ids = page_ids_a_final[(prefix_len // atomic_block_size):]
    block_table_paged_direct = torch.tensor([tail_page_ids], dtype=torch.int32)
    out_segmented_direct = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out_segmented_direct,
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table_paged_direct,
        softcap=0,
        segment_num=torch.tensor([2], dtype=torch.int32),
        segment_lens=torch.tensor([[seg_len, kv_len - seg_len]], dtype=torch.int32),
        segment_k_ptrs=torch.tensor([[sealed_k_ptr, 0]], dtype=torch.int64),
        segment_v_ptrs=torch.tensor([[sealed_v_ptr, 0]], dtype=torch.int64),
        fa_version=fa_version,
    )

    torch.testing.assert_close(out_segmented_direct, out_paged, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out, out_paged, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
