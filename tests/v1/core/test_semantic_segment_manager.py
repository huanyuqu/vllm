import pytest
from unittest.mock import MagicMock
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.semantic_segment_manager import SemanticSegmentManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_segment_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request


@pytest.fixture
def block_pool():
    return BuddyBlockPool(
        num_max_gpu_blocks=1024,
        supported_sizes=[8, 32, 128],
        enable_caching=True
    )


@pytest.fixture
def segment_manager(block_pool):
    return SemanticSegmentManager(block_pool)


def test_supported_sizes_completion(block_pool):
    assert block_pool.supported_sizes == (128, 64, 32, 16, 8)
    assert block_pool.max_block_size == 128
    assert block_pool.min_block_size == 8


def test_allocate_new_blocks(segment_manager):
    request_id = "req1"
    blocks = segment_manager.allocate_new_blocks(request_id, 256)
    assert len(blocks) == 2
    assert sum(b.size for b in blocks) == 256
    assert all(not b.is_sealed for b in blocks)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 1
    assert segments.unsealed_segment is not None
    assert len(segments.unsealed_segment.blocks) == len(blocks)
    
    
def test_allocate_block_with_excessive_memory(segment_manager):
    request_id = "req1"
    blocks = segment_manager.allocate_new_blocks(request_id, 64)
    assert len(blocks) == 1
    assert sum(b.size for b in blocks) == 128


def test_seal_segment(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id
    
    # Allocate blocks
    blocks = segment_manager.allocate_new_blocks(request_id, 256)
    
    # Manually set block hashes as if they were computed and cached
    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        block.ref_cnt = 1  # Simulate usage
        
    # Seal
    segment_manager.seal_segment(request, group_id)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 1
    assert segments.unsealed_segment is None
    assert segments.last_segment.is_sealed
    assert segments.last_segment.segment_hash is not None
    assert segments.last_segment.ref_cnt == 1
    
    
def test_seal_segment_different_ref_counts(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id
    
    # Allocate blocks
    blocks = segment_manager.allocate_new_blocks(request_id, 512)
    
    # Manually set block hashes as if they were computed and cached
    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        if i == 0:
            block.ref_cnt = 3
        elif i == 1:
            block.ref_cnt = 2
        else:
            block.ref_cnt = 1
        
    # Seal
    segment_manager.seal_segment(request, group_id)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 3
    assert segments[0].ref_cnt == 3
    assert len(segments[0]) == 1
    assert segments[1].ref_cnt == 2
    assert len(segments[1]) == 1
    assert segments[2].ref_cnt == 1
    assert len(segments[2]) == 2
    
    
def test_seal_multiple_segments(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id
    
    # First allocation and seal
    blocks1 = segment_manager.allocate_new_blocks(request_id, 256)
    group_id = 0
    for i, block in enumerate(blocks1):
        block_hash = BlockHash(f"hash1_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        block.ref_cnt = 1
    
    segment_manager.seal_segment(request, group_id)
    
    # Second allocation and seal
    blocks2 = segment_manager.allocate_new_blocks(request_id, 64)
    for i, block in enumerate(blocks2):
        block_hash = BlockHash(f"hash2_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        block.ref_cnt = 1
    
    segment_manager.seal_segment(request, group_id)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 2
    assert segments.unsealed_segment is None
    assert segments[0].segment_hash is not None
    assert segments[1].segment_hash is not None
    assert segments[0].segment_hash != segments[1].segment_hash
    
    
def test_segment_sharing(segment_manager):
    req1 = MagicMock(spec=Request)
    req1.request_id = "req1"
    req2 = MagicMock(spec=Request)
    req2.request_id = "req2"
    
    # Allocate blocks for req1 (2 blocks)
    blocks1 = segment_manager.allocate_new_blocks("req1", 256)
    
    # Allocate blocks for req2 (2 blocks)
    blocks2 = segment_manager.allocate_new_blocks("req2", 129)
    
    # Simulate sharing: req2 has [blocks1, blocks2]
    req2_segments = segment_manager.req_to_segments["req2"]
    req2_unsealed = req2_segments.unsealed_segment
    req2_unsealed.blocks = blocks1 + req2_unsealed.blocks
    
    # Update ref counts
    for b in blocks1:
        b.ref_cnt = 2
    for b in blocks2:
        b.ref_cnt = 1
        
    # Set hashes
    group_id = 0
    all_blocks = blocks1 + blocks2
    for i, block in enumerate(all_blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
    # Seal req1
    segment_manager.seal_segment(req1, group_id)
    
    # Verify req1
    req1_segs = segment_manager.req_to_segments["req1"]
    assert len(req1_segs) == 1
    assert req1_segs[0].is_sealed
    shared_segment = req1_segs[0]
    
    # Seal req2
    segment_manager.seal_segment(req2, group_id)
    
    # Verify req2
    req2_segs = segment_manager.req_to_segments["req2"]
    assert len(req2_segs) == 2
    assert req2_segs[0] is shared_segment
    assert req2_segs[1].is_sealed
    assert req2_segs[1].blocks == blocks2

    
def test_free_request_and_eviction():
    # Create a very small pool: 2 blocks of size 64
    pool = BuddyBlockPool(num_gpu_blocks=2, supported_sizes=[64], enable_caching=True)
    manager = SemanticSegmentManager(pool)
    
    req1 = MagicMock(spec=Request)
    req1.request_id = "req1"
    
    # Allocate all memory for req1 (128 tokens)
    manager.allocate_new_blocks("req1", 128)
    
    # Seal and free req1
    group_id = 0
    segments = manager.req_to_segments["req1"]
    # We need to set block hashes for seal to work
    for block in segments.unsealed_segment.blocks:
        block.block_hash = make_block_hash_with_group_id(BlockHash(b"h"), group_id)
        block.ref_cnt = 1
        
    manager.free(req1, group_id)
    
    # Check that segments are in free queue
    assert manager.free_segment_queue.num_free_blocks > 0
    
    # Now allocate for req2, should trigger eviction
    req2 = MagicMock(spec=Request)
    req2.request_id = "req2"
    
    # This should succeed by evicting req1's segments
    # We request 64 tokens, which requires 1 block.
    # The pool is full (used by req1's freed segments).
    # Eviction should free up space.
    blocks = manager.allocate_new_blocks("req2", 64)
    assert len(blocks) > 0
    
    # Check that free queue is reduced (one segment evicted)
    # Note: exact behavior depends on how many segments were created for req1.
    # If 128 tokens were 2 blocks of 64, and they were sealed into 1 segment (if hashes match/logic allows) or 2 segments.
    # seal_segment groups consecutive blocks with same ref_cnt.
    # Here all have ref_cnt=1. So they should be 1 segment if logic allows.
    # But wait, seal_segment logic:
    # "We group consecutive blocks with the same ref_cnt into one segment."
    # So likely 1 segment of 2 blocks.
    # If we evict that segment, we free 2 blocks.
    # Then we allocate 1 block.
    # So we have 1 free block left in pool, and 0 segments in free queue.
    
    assert manager.free_segment_queue.num_free_blocks == 0


def test_reclaim_from_allocated_blocks():
    # Test the 3rd stage of allocation: reclaiming from allocated blocks
    # Pool: 1 block of 64.
    pool = BuddyBlockPool(num_gpu_blocks=1, supported_sizes=[64, 32], enable_caching=True)
    manager = SemanticSegmentManager(pool)
    
    req1 = MagicMock(spec=Request)
    req1.request_id = "req1"
    
    # Allocate 32 tokens for req1. This splits the 64 block into 32 (allocated) and 32 (free).
    manager.allocate_new_blocks("req1", 32)
    
    # Now we have 32 free.
    
    # Allocate another 32 for req2.
    manager.allocate_new_blocks("req2", 32)
    
    # Now pool is full (in terms of 32-blocks). 
    # Actually, 64 -> 32(req1) + 32(req2). Both allocated.
    
    # Now req1 is done but NOT freed (simulating fragmentation or just usage).
    # Wait, if req1 is not freed, we can't reclaim from it unless we implement partial reclamation which BuddyBlockPool supports?
    # BuddyBlockPool.reclaim_from_allocated_blocks tries to split blocks that are larger than needed?
    # No, it reclaims from blocks that are *allocated* but have *unused* space?
    # Let's check BuddyBlockPool._reclaim_one_block logic.
    # It checks `block.num_tokens <= size - self.min_block_size`.
    # If we allocated 32 tokens, `num_tokens` is likely 32 (capacity).
    # But `update_block_usage` updates `num_tokens` (used).
    # If we didn't call `update_block_usage`, `num_tokens` might be 0 or capacity?
    # BuddyTreeBlock `num_tokens` defaults to 0? No, `size` is capacity. `num_tokens` is usage?
    # In `_split_block`: `left_child.num_tokens = min(child_size, total_tokens)`.
    # When allocating, `get_new_blocks` calls `_allocate_largest_blocks`.
    # It doesn't seem to set `num_tokens` (usage) on the block?
    # Ah, `BuddyTreeBlock` has `_num_tokens`.
    
    # Let's look at `BuddyBlockPool` again.
    # `_allocate_block`: `self.allocated_blocks[size].add(block)`.
    # It doesn't set `num_tokens`.
    # So `num_tokens` is 0 initially?
    # `BuddyTreeBlock` definition: `_num_tokens: int = field(default=0, init=False)`.
    # So yes, 0.
    
    # So if we allocate a 64 block, `num_tokens` is 0.
    # `_reclaim_one_block` checks `block.num_tokens <= size - self.min_block_size`.
    # 0 <= 64 - 32 (if min is 32). True.
    # So it can reclaim.
    
    # So if we allocate a 64 block for req1, but only use 32 tokens (conceptually),
    # we can reclaim the other 32.
    
    # Let's try:
    # Pool: 1 block of 64.
    pool = BuddyBlockPool(num_gpu_blocks=1, supported_sizes=[64, 32], enable_caching=True)
    manager = SemanticSegmentManager(pool)
    
    # Allocate 64 tokens for req1. This takes the whole 64 block.
    # But we only "use" 32 tokens.
    # Wait, `allocate_new_blocks` takes `num_tokens`.
    # If we ask for 32, it splits and gives us 32.
    # If we ask for 64, it gives us 64.
    
    # To test reclamation, we need a block that is allocated as LARGE, but used SMALL.
    # E.g. allocate 64.
    blocks = manager.allocate_new_blocks("req1", 64)
    block = blocks[0]
    # Simulate usage: only 32 tokens used.
    pool.update_block_usage(block.block_id, block.size, block.relative_id, 32)
    
    # Now try to allocate another 32 tokens for req2.
    # The pool has no free blocks.
    # But it should be able to reclaim from req1's block (split 64 -> 32 used + 32 free).
    
    blocks2 = manager.allocate_new_blocks("req2", 32)
    assert len(blocks2) > 0
    assert blocks2[0].size == 32


def test_get_cached_segment_hit(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id

    blocks = segment_manager.allocate_new_blocks(request_id, 64)

    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        block.ref_cnt = 1

    segment_manager.seal_segment(request, group_id)
    segments = segment_manager.req_to_segments[request_id]
    sealed = segments.last_segment
    assert sealed is not None
    assert sealed.segment_hash is not None

    # Lookup uses the group-agnostic SegmentHash.
    base_segment_hash = get_segment_hash(sealed.segment_hash)
    cached = segment_manager.get_cached_segment(base_segment_hash, [group_id])
    assert cached is not None
    assert cached[0].segment_id == sealed.segment_id


def test_find_longest_cache_hit_segments_only(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id

    blocks = segment_manager.allocate_new_blocks(request_id, 64)
    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        block.ref_cnt = 1

    segment_manager.seal_segment(request, group_id)
    sealed = segment_manager.req_to_segments[request_id].last_segment
    assert sealed is not None and sealed.segment_hash is not None

    base_segment_hash = get_segment_hash(sealed.segment_hash)

    # Segment matching should return the full segment blocks without relying on
    # block-level cached_block_hash_to_block.
    hit_blocks = segment_manager.find_longest_cache_hit(
        segment_hashes=[base_segment_hash],
        block_hashes=[],
        max_length=10**9,
        kv_cache_group_ids=[group_id],
        block_size=64,
        use_eagle=False,
    )
    assert len(hit_blocks) == 1
    assert hit_blocks[0] == sealed.blocks
