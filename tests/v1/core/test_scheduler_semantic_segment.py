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

pytestmark = pytest.mark.cpu_test


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
        semantic_eviction_policy="tight",
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
    
    num_tokens = 100
    # 100 tokens: buddy pool allocates two 64-token blocks, then split into 16-token atomic blocks
    # 64-token blocks: 100 / 64 -> ceil(1.5625) = 2 blocks
    # Each 64-token block splits into 64 / 16 = 4 atomic blocks
    # Total atomic blocks: 2 * 4 = 8
    expected_num_atomic_blocks = 8
    
    reqs = create_requests(num_requests=1, num_tokens=num_tokens, block_size=block_size)
    for req in reqs:
        scheduler.add_request(req)
        
    output = scheduler.schedule()
    
    assert len(output.scheduled_new_reqs) == 1
    new_req = output.scheduled_new_reqs[0]
    
    # Check block_ids
    block_ids = new_req.block_ids
    
    assert len(block_ids) == 1  # Only one KV cache group
    assert len(block_ids[0]) == expected_num_atomic_blocks
    assert block_ids[0] == [0, 1, 2, 3, 4, 5, 6, 7]
