import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from tests.v1.core.utils import create_requests
from vllm.v1.structured_output import StructuredOutputManager

def test_schedule_semantic_segments():
    block_size = 16
    supported_block_sizes = [16, 32, 64]
    
    # Setup configs
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    
    # Enable semantic segments
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_semantic_segment=True,
        semantic_supported_block_sizes=supported_block_sizes,
        semantic_eviction_policy="lru",
    )
    cache_config.num_gpu_blocks = 1000

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    
    # KV Cache Config needs atomic block size
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32, False)
            )
        ],
    )
    
    # Init Scheduler
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    # Create a request that requires multiple blocks
    # Length=100. supported_sizes=[16, 32, 64].
    # Decomposition depends on allocator implementation, but likely 64 + 32 + 4 (round up to 16) -> 64+32+16?
    # Or 64 + 16 + 16 + 4?
    # Regardless, we expect `block_ids` to be fully populated with atomic IDs.
    # 100 tokens. min block size 16. ceil(100/16) * 16 = 7 * 16 = 112 tokens capacity minimum.
    # So we expect 7 atomic blocks (IDs).
    
    num_tokens = 100
    expected_num_atomic_blocks = (num_tokens + block_size - 1) // block_size # 7
    
    reqs = create_requests(num_requests=1, num_tokens=num_tokens, block_size=block_size)
    for req in reqs:
        scheduler.add_request(req)
        
    output = scheduler.schedule()
    
    assert len(output.scheduled_new_reqs) == 1
    new_req = output.scheduled_new_reqs[0]
    
    # Check block_ids
    block_ids = new_req.block_ids
    print(f"DEBUG: block_ids={block_ids}")
    
    assert isinstance(block_ids, list)
    assert len(block_ids) > 0
    assert all(isinstance(id, int) for id in block_ids)
    
    # We expect integer IDs, not objects
    # With standard allocation (slot based), we get `expected_num_atomic_blocks`
    # With semantic allocation, constructing those "variable blocks" should break down into at least that many atomic blocks
    # Actually it should be exactly that many atomic blocks because `get_block_ids` flattens them.
    # Assume 1 chunk of 64 (4 atoms), 1 chunk of 32 (2 atoms), 1 chunk of 16 (1 atom) = 7 atoms.
    
    assert len(block_ids) == expected_num_atomic_blocks
    
    # Verify they are unique (basic check)
    assert len(set(block_ids)) == len(block_ids)

