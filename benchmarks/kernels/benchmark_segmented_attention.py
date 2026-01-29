import torch
import time
import pytest
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, CompilationConfig, LoadConfig, SpeculativeConfig, ObservabilityConfig
from vllm.v1.kv_cache_interface import AttentionSpec


class MockLayer:
    _k_scale = torch.tensor(1.0, device="cuda")
    _v_scale = torch.tensor(1.0, device="cuda")
    _q_scale = torch.tensor(1.0, device="cuda")

def benchmark_flash_attn_segments():
    layer = MockLayer()
    # Setup
    num_reqs = 32
    head_size = 128
    num_heads = 32
    num_kv_heads = 32
    block_size = 16
    dtype = torch.float16
    device = "cuda"
    
    # 1. Simulate Paged Attention (Baseline)
    # 100 blocks per request, all paged
    num_blocks_per_req = 100
    total_blocks = num_reqs * num_blocks_per_req
    
    kv_cache = torch.randn(2, total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).view(num_reqs, num_blocks_per_req)
    
    # Query: 1 token per request
    query = torch.randn(num_reqs, num_heads, head_size, dtype=dtype, device=device)
    
    # Metadata for Paged
    common_meta_paged = CommonAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs, dtype=torch.int32, device="cpu"), # Not used for decode
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables,
        slot_mapping=torch.full((num_reqs, 1), -1, dtype=torch.long, device=device), # Dummy
    )
    
    # Mock VllmConfig
    model_config = ModelConfig("gpt2", "generate", tokenizer_mode="auto", trust_remote_code=False, dtype=dtype, seed=0)
    model_config.num_attention_heads = num_heads
    model_config.num_kv_heads = num_kv_heads
    model_config.head_dim = head_size
    model_config.dtype = dtype
    
    cache_config = CacheConfig(block_size, 0.9, 1, "auto")
    parallel_config = ParallelConfig(1, 1, False)
    scheduler_config = SchedulerConfig("generate", 2048, num_reqs, 2048)
    device_config = DeviceConfig(device)
    compilation_config = CompilationConfig()
    load_config = LoadConfig()
    spec_config = None
    obs_config = ObservabilityConfig()
    
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        compilation_config=compilation_config,
        load_config=load_config,
        speculative_config=spec_config,
        observability_config=obs_config
    )
    
    kv_spec = AttentionSpec(block_size, num_kv_heads, head_size, dtype)
    
    backend = FlashAttentionBackend
    impl = backend.get_impl_cls()(num_heads, head_size, 1.0, num_kv_heads, None, None, "auto")
    impl.device = torch.device(device)
    builder = backend.get_builder_cls()(kv_spec, ["layer1"], vllm_config, torch.device(device))
    
    meta_paged = builder.build(0, common_meta_paged)
    for _ in range(10):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_paged, output=torch.empty_like(query))
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_paged, output=torch.empty_like(query))
    torch.cuda.synchronize()
    paged_time = (time.time() - start) / 100 * 1000
    print(f"Paged Attention Time: {paged_time:.3f} ms")
    
    # 2. Simulate Segmented Attention
    # 5 segments of 20 blocks (320 tokens) each
    # Segments are contiguous in kv_cache (lucky allocation)
    # We construct segment pointers
    num_segments_per_req = 5
    segment_len = 20 * block_size
    
    segment_pointers = []
    segment_lens = []
    num_segments_list = []
    
    for i in range(num_reqs):
        base = i * num_blocks_per_req * block_size # Flattened index
        req_ptrs = []
        req_lens = []
        for s in range(num_segments_per_req):
            req_ptrs.append(base + s * segment_len)
            req_lens.append(segment_len)
        segment_pointers.extend(req_ptrs)
        segment_lens.extend(req_lens)
        num_segments_list.append(num_segments_per_req)
        
    common_meta_seg = CommonAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs, dtype=torch.int32, device="cpu"),
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables, # Still needed for fallback/unsealed
        slot_mapping=torch.full((num_reqs, 1), -1, dtype=torch.long, device=device),
        segment_pointers=torch.tensor(segment_pointers, dtype=torch.long, device=device),
        segment_lens=torch.tensor(segment_lens, dtype=torch.int32, device=device),
        num_segments=torch.tensor(num_segments_list, dtype=torch.int32, device="cpu") # On CPU for iterator
    )
    
    meta_seg = builder.build(0, common_meta_seg)
    
    # Warmup Segmented
    for _ in range(10):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_seg, output=torch.empty_like(query))
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_seg, output=torch.empty_like(query))
    torch.cuda.synchronize()
    seg_time = (time.time() - start) / 100 * 1000
    print(f"Segmented Attention Time: {seg_time:.3f} ms")

    # 3. Simulate Single Segment Attention
    # 1 segment covering the whole sequence
    num_segments_per_req_single = 1
    segment_len_single = num_blocks_per_req * block_size
    
    segment_pointers_single = []
    segment_lens_single = []
    num_segments_list_single = []
    
    for i in range(num_reqs):
        base = i * num_blocks_per_req * block_size 
        req_ptrs = [base]
        req_lens = [segment_len_single]
        segment_pointers_single.extend(req_ptrs)
        segment_lens_single.extend(req_lens)
        num_segments_list_single.append(num_segments_per_req_single)

    common_meta_seg_single = CommonAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs, dtype=torch.int32, device="cpu"),
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables,
        slot_mapping=torch.full((num_reqs, 1), -1, dtype=torch.long, device=device),
        segment_pointers=torch.tensor(segment_pointers_single, dtype=torch.long, device=device),
        segment_lens=torch.tensor(segment_lens_single, dtype=torch.int32, device=device),
        num_segments=torch.tensor(num_segments_list_single, dtype=torch.int32, device="cpu")
    )
    
    meta_seg_single = builder.build(0, common_meta_seg_single)
    
    # Warmup Single Segment
    for _ in range(10):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_seg_single, output=torch.empty_like(query))
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query, kv_cache, kv_cache, kv_cache, meta_seg_single, output=torch.empty_like(query))
    torch.cuda.synchronize()
    seg_single_time = (time.time() - start) / 100 * 1000
    print(f"Single Segment Attention Time: {seg_single_time:.3f} ms")
    # 4. Simulate Single Request Attention (Batch Size = 1)
    # This isolates the overhead of Python loop vs Kernel overhead per request
    num_reqs_1 = 1
    # We need new metadata for BS=1
    query_1 = query[:1]
    
    # 4.1 Paged (BS=1)
    common_meta_paged_1 = CommonAttentionMetadata(
        num_reqs=num_reqs_1,
        num_actual_tokens=num_reqs_1,
        query_start_loc=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs_1, dtype=torch.int32, device="cpu"),
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables[:1],
        slot_mapping=torch.full((num_reqs_1, 1), -1, dtype=torch.long, device=device),
    )
    meta_paged_1 = builder.build(0, common_meta_paged_1)
    
    # Warmup Paged BS=1
    for _ in range(10):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_paged_1, output=torch.empty_like(query_1))
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_paged_1, output=torch.empty_like(query_1))
    torch.cuda.synchronize()
    paged_1_time = (time.time() - start) / 100 * 1000
    
    # 4.2 Segmented (BS=1, 5 Segments)
    segment_pointers_1 = segment_pointers[:num_segments_per_req]
    segment_lens_1 = segment_lens[:num_segments_per_req]
    num_segments_list_1 = [num_segments_per_req]
    
    common_meta_seg_1 = CommonAttentionMetadata(
        num_reqs=num_reqs_1,
        num_actual_tokens=num_reqs_1,
        query_start_loc=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs_1, dtype=torch.int32, device="cpu"),
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables[:1],
        slot_mapping=torch.full((num_reqs_1, 1), -1, dtype=torch.long, device=device),
        segment_pointers=torch.tensor(segment_pointers_1, dtype=torch.long, device=device),
        segment_lens=torch.tensor(segment_lens_1, dtype=torch.int32, device=device),
        num_segments=torch.tensor(num_segments_list_1, dtype=torch.int32, device="cpu")
    )
    meta_seg_1 = builder.build(0, common_meta_seg_1)
    
    # Warmup Segmented BS=1
    for _ in range(10):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_seg_1, output=torch.empty_like(query_1))
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_seg_1, output=torch.empty_like(query_1))
    torch.cuda.synchronize()
    seg_1_time = (time.time() - start) / 100 * 1000
    
    print(f"\n--- Batch Size = 1 Analysis ---")
    print(f"Paged Attention Time (BS=1): {paged_1_time:.3f} ms")
    print(f"Segmented Attention Time (BS=1, 5 Segs): {seg_1_time:.3f} ms")
    print(f"Speedup (BS=1): {paged_1_time / seg_1_time:.2f}x")

    # 5. Simulate Single Request & Single Segment (BS=1, Segs=1)
    # Ideally this should be fastest if implementation is optimal
    num_segments_per_req_single_1 = 1
    segment_len_single_1 = num_blocks_per_req * block_size
    
    segment_pointers_single_1 = [0] # Assuming first request, first block
    segment_lens_single_1 = [segment_len_single_1]
    num_segments_list_single_1 = [1]
    
    common_meta_seg_single_1 = CommonAttentionMetadata(
        num_reqs=num_reqs_1,
        num_actual_tokens=num_reqs_1,
        query_start_loc=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.arange(num_reqs_1 + 1, dtype=torch.int32, device="cpu"),
        seq_lens=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.full((num_reqs_1,), num_blocks_per_req * block_size, dtype=torch.int32, device="cpu"),
        num_computed_tokens_cpu=torch.zeros(num_reqs_1, dtype=torch.int32, device="cpu"),
        max_query_len=1,
        max_seq_len=num_blocks_per_req * block_size,
        block_table_tensor=block_tables[:1],
        slot_mapping=torch.full((num_reqs_1, 1), -1, dtype=torch.long, device=device),
        segment_pointers=torch.tensor(segment_pointers_single_1, dtype=torch.long, device=device),
        segment_lens=torch.tensor(segment_lens_single_1, dtype=torch.int32, device=device),
        num_segments=torch.tensor(num_segments_list_single_1, dtype=torch.int32, device="cpu")
    )
    meta_seg_single_1 = builder.build(0, common_meta_seg_single_1)
    
    # Warmup BS=1 Segs=1
    for _ in range(10):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_seg_single_1, output=torch.empty_like(query_1))
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        impl.forward(layer, query_1, kv_cache, kv_cache, kv_cache, meta_seg_single_1, output=torch.empty_like(query_1))
    torch.cuda.synchronize()
    seg_single_1_time = (time.time() - start) / 100 * 1000
    
    print(f"Segmented Attention Time (BS=1, 1 Seg): {seg_single_1_time:.3f} ms")
    print(f"Speedup (BS=1, 1 Seg vs Paged): {paged_1_time / seg_single_1_time:.2f}x")

if __name__ == "__main__":
    benchmark_flash_attn_segments()
