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

from tests.kernels.attention.test_flash_attn import ref_paged_attn


def _fa2_varlen_fwd_supports_segmented() -> bool:
    try:
        # The op schema is registered when the C++ extension is loaded.
        # Try querying first; if the extension hasn't been imported yet, import
        # it and retry to avoid false negatives.
        try:
            schema = torch._C._get_schema("_vllm_fa2_C::varlen_fwd", "")
        except Exception:
            import vllm.vllm_flash_attn.flash_attn_interface  # noqa: F401
            schema = torch._C._get_schema("_vllm_fa2_C::varlen_fwd", "")
        schema_str = str(schema)
        return "segment_num" in schema_str and "segment_lens" in schema_str
    except Exception:
        return False


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@torch.inference_mode()
def test_segmented_hybrid_varlen_matches_paged_reference() -> None:
    """Smoke test for vLLM segmented-hybrid flash-attn call.

    We construct a single sequence whose KV is split into:
    - one sealed (contiguous) segment
    - one trailing paged segment (specified via block_table)

    The segmented-hybrid kernel output should match the regular paged attention
    reference computed over the full KV sequence.
    """

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

    current_platform.seed_everything(0)

    # Minimal shapes.
    query_len = 4
    kv_len = 32
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 64
    dtype = torch.bfloat16

    block_size = 16
    num_blocks = 4

    scale = head_size**-0.5
    window_size = (-1, -1)

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)

    # Full block table for the reference paged attention: use block 0 then 1.
    full_block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    # Sealed segment covers the first block (16 tokens) contiguously.
    sealed_len = 16
    paged_len = kv_len - sealed_len
    assert paged_len == 16

    k_flat = key_cache.view(-1, num_kv_heads, head_size)
    v_flat = value_cache.view(-1, num_kv_heads, head_size)
    sealed_k_ptr = k_flat[0].data_ptr()
    sealed_v_ptr = v_flat[0].data_ptr()

    # Segmented-hybrid metadata: 1 sealed + 1 trailing paged.
    segment_num = torch.tensor([2], dtype=torch.int32)
    segment_lens = torch.tensor([[sealed_len, paged_len]], dtype=torch.int32)
    segment_k_ptrs = torch.tensor([[sealed_k_ptr, 0]], dtype=torch.int64)
    segment_v_ptrs = torch.tensor([[sealed_v_ptr, 0]], dtype=torch.int64)

    # Block table only for the trailing paged segment: last block is block 1.
    block_table_paged = torch.tensor([[1]], dtype=torch.int32)

    out = torch.empty_like(query)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_table_paged,
        softcap=0,
        segment_num=segment_num,
        segment_lens=segment_lens,
        segment_k_ptrs=segment_k_ptrs,
        segment_v_ptrs=segment_v_ptrs,
        fa_version=fa_version,
    )

    ref_out = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=full_block_table,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
    )

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@torch.inference_mode()
def test_v1_flashattn_impl_forwards_segment_metadata() -> None:
    """End-to-end vLLM v1 path smoke test.

    This exercises the vLLM chain:
    FlashAttentionImpl.forward -> (segmented branch) -> flash_attn_varlen_func

    The test asserts that the backend forwards `segment_num` into the
    flash-attn call, and that the numerical result matches the paged reference.
    """

    torch.set_default_device("cuda")

    fa_version = 2
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    # Ensure the wrapper exposes segmented args (otherwise backend can't pass).
    sig = inspect.signature(flash_attn_varlen_func)
    if "segment_num" not in sig.parameters:
        pytest.skip("flash_attn_varlen_func does not support segmented attention")

    if not _fa2_varlen_fwd_supports_segmented():
        pytest.skip(
            "Loaded torch.ops._vllm_fa2_C.varlen_fwd schema does not expose "
            "segment_* arguments; likely still using an old _vllm_fa2_C.abi3.so"
        )

    current_platform.seed_everything(0)

    # Import backend lazily to avoid importing CUDA-heavy modules on skip.
    from vllm.attention.backends.abstract import AttentionType
    from vllm.v1.attention.backends import flash_attn as fa_backend

    # Minimal shapes.
    query_len = 4
    kv_len = 32
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 64
    dtype = torch.bfloat16
    block_size = 16
    num_blocks = 4

    scale = head_size**-0.5

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    kv_cache = torch.stack([key_cache, value_cache], dim=0)

    # Full block table for the reference paged attention: use block 0 then 1.
    full_block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    # Semantic segment metadata: one sealed segment of length 16 at token index 0.
    # The remaining 16 tokens are the paged tail.
    num_segments = torch.tensor([1], dtype=torch.int32)
    segment_pointers = torch.tensor([0], dtype=torch.int32)
    segment_lens = torch.tensor([16], dtype=torch.int32)

    # Build FlashAttentionMetadata required by FlashAttentionImpl.forward.
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
        segment_pointers=segment_pointers,
        segment_lens=segment_lens,
        num_segments=num_segments,
        causal=True,
    )

    # Dummy layer object providing the scale tensors used by the backend.
    class _DummyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale_shape = (1, num_kv_heads)
            self._q_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._k_scale = torch.ones(scale_shape, dtype=torch.float32)
            self._v_scale = torch.ones(scale_shape, dtype=torch.float32)

    layer = _DummyLayer()

    # Monkeypatch the backend's local symbol to confirm it forwards segment args.
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

    assert called["segmented"], (
        "FlashAttentionImpl.forward did not forward segment_* args; "
        "vLLM segmented-hybrid path is likely not wired up."
    )

    ref_out = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=full_block_table,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
    )
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)