import pytest
from unittest.mock import MagicMock
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.semantic_segment_manager import EvictionPolicy, SemanticSegmentManager
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
    assert blocks[0].size == 128
    assert blocks[1].size == 128
    assert blocks[0].ref_cnt == 1
    assert blocks[1].ref_cnt == 1
    assert all(not b.is_sealed for b in blocks)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 1
    assert segments.unsealed_segment is not None
    assert len(segments.unsealed_segment) == len(blocks)
    
    
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
        
    # Seal
    segment_manager.seal_segment(request, group_id)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 1
    assert segments.unsealed_segment is None
    assert segments.last_segment.is_sealed
    assert segments.last_segment.segment_hash is not None
    assert segments.last_segment.ref_cnt == 1
    assert len(segments.last_segment) == 2
    
    
# def test_seal_segment_different_ref_counts(segment_manager):
#     request_id = "req1"
#     request = MagicMock(spec=Request)
#     request.request_id = request_id
    
#     # Allocate blocks
#     blocks = segment_manager.allocate_new_blocks(request_id, 512)
    
#     # Manually set block hashes as if they were computed and cached
#     group_id = 0
#     for i, block in enumerate(blocks):
#         block_hash = BlockHash(f"hash_{i}".encode())
#         block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
#         if i == 0:
#             block.ref_cnt = 3
#         elif i == 1:
#             block.ref_cnt = 2
#         else:
#             block.ref_cnt = 1
        
#     # Seal
#     segment_manager.seal_segment(request, group_id)
    
#     segments = segment_manager.req_to_segments[request_id]
#     assert len(segments) == 3
#     assert segments[0].ref_cnt == 3
#     assert len(segments[0]) == 1
#     assert segments[1].ref_cnt == 2
#     assert len(segments[1]) == 1
#     assert segments[2].ref_cnt == 1
#     assert len(segments[2]) == 2
    
    
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
    
    segment_manager.seal_segment(request, group_id)
    
    # Second allocation and seal
    blocks2 = segment_manager.allocate_new_blocks(request_id, 64)
    for i, block in enumerate(blocks2):
        block_hash = BlockHash(f"hash2_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
    
    segment_manager.seal_segment(request, group_id)
    
    segments = segment_manager.req_to_segments[request_id]
    assert len(segments) == 2
    assert segments.unsealed_segment is None
    assert segments[0].segment_hash is not None
    assert segments[1].segment_hash is not None
    assert segments[0].segment_hash != segments[1].segment_hash
    
    
# def test_segment_sharing(segment_manager):
#     req1 = MagicMock(spec=Request)
#     req1.request_id = "req1"
#     req2 = MagicMock(spec=Request)
#     req2.request_id = "req2"
    
#     # Allocate blocks for req1 (2 blocks)
#     blocks1 = segment_manager.allocate_new_blocks("req1", 256)
    
#     # Allocate blocks for req2 (2 blocks)
#     blocks2 = segment_manager.allocate_new_blocks("req2", 129)
    
#     # Simulate sharing: req2 has [blocks1, blocks2]
#     req2_segments = segment_manager.req_to_segments["req2"]
#     req2_unsealed = req2_segments.unsealed_segment
#     req2_unsealed.blocks = blocks1 + req2_unsealed.blocks
    
#     # Update ref counts
#     for b in blocks1:
#         b.ref_cnt = 2
#     for b in blocks2:
#         b.ref_cnt = 1
        
#     # Set hashes
#     group_id = 0
#     all_blocks = blocks1 + blocks2
#     for i, block in enumerate(all_blocks):
#         block_hash = BlockHash(f"hash_{i}".encode())
#         block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
#     # Seal req1
#     segment_manager.seal_segment(req1, group_id)
    
#     # Verify req1
#     req1_segs = segment_manager.req_to_segments["req1"]
#     assert len(req1_segs) == 1
#     assert req1_segs[0].is_sealed
#     shared_segment = req1_segs[0]
    
#     # Seal req2
#     segment_manager.seal_segment(req2, group_id)
    
#     # Verify req2
#     req2_segs = segment_manager.req_to_segments["req2"]
#     assert len(req2_segs) == 2
#     assert req2_segs[0] is shared_segment
#     assert req2_segs[1].is_sealed
#     assert req2_segs[1].blocks == blocks2
    
    
def test_allocate_free_eviction():
    # Create a small pool: 2 blocks of size 64
    pool = BuddyBlockPool(num_max_gpu_blocks=2, supported_sizes=[64], enable_caching=True)
    manager = SemanticSegmentManager(pool)
    
    req1 = MagicMock(spec=Request)
    req1.request_id = "req1"
    
    # Allocate all memory for req1 (128 tokens)
    manager.allocate_new_blocks("req1", 128)
    
    # Seal and free req1
    group_id = 0
    segments = manager.req_to_segments["req1"]
    for i, block in enumerate(segments.unsealed_segment.blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
    manager.free(req1, group_id)
    
    # Check that segments are in free queue
    assert manager.free_segment_queue.num_free_segments == 1
    
    # Now allocate for req2, should trigger eviction
    req2 = MagicMock(spec=Request)
    req2.request_id = "req2"
    
    # This should succeed by evicting req1's segments
    blocks = manager.allocate_new_blocks("req2", 64)
    assert len(blocks) == 1
    
    # Check that free queue is empty (segment evicted)
    assert manager.free_segment_queue.num_free_segments == 0
    # Check that we have 1 free block left in the pool (2 freed - 1 allocated)
    assert pool.slabs[64].num_free_blocks == 1


def test_reclaim_from_allocated_blocks():
    # Pool: 1 block of 64.
    pool = BuddyBlockPool(num_max_gpu_blocks=1, supported_sizes=[64, 32], 
                          enable_caching=True)
    manager = SemanticSegmentManager(pool)
    
    blocks = manager.allocate_new_blocks("req1", 32)
    assert len(blocks) == 1
    assert blocks[0].size == 64  # Allocated size
    pool.update_block_usage(0, 64, 0, 32)
    assert blocks[0].num_tokens == 32
    
    # Now try to allocate another 32 tokens for req2.
    # The pool has no free blocks.
    # But it should be able to reclaim from req1's block (split 64 -> 32 used + 32 free).
    blocks2 = manager.allocate_new_blocks("req2", 32)
    assert len(blocks2) == 1
    assert blocks2[0].size == 32
    assert len(pool.allocated_blocks[blocks2[0].size]) == 2  # req1 and req2 blocks
    assert pool.slabs[blocks2[0].size].num_free_blocks == 0
    
    first_block = manager.req_to_segments["req1"].unsealed_segment.head
    assert first_block.size == 32
    assert first_block.num_tokens == 32
    assert blocks[0].size == 64
    assert blocks[0] not in pool.allocated_blocks[blocks[0].size]
    assert pool.slabs[blocks[0].size].num_free_blocks == 0
    
    
@pytest.mark.parametrize("eviction_policy", [EvictionPolicy.TIGHT, EvictionPolicy.OVERPROVISION])
@pytest.mark.parametrize("request_size", [32, 64])
def test_automatic_merge(request_size, eviction_policy):
    # 1. Create pool with 1 block of size 64
    pool = BuddyBlockPool(num_max_gpu_blocks=1, supported_sizes=[64, 32], 
                          enable_caching=True)
    manager = SemanticSegmentManager(pool, eviction_policy=eviction_policy)
    
    # 2. Allocate two 32-token requests to split the 64 block
    # req1 takes 32 tokens (half of the 64 block)
    blocks1 = manager.allocate_new_blocks("req1", 32)
    pool.update_block_usage(*blocks1[0].full_id, 32)
    assert len(blocks1) == 1
    assert blocks1[0].size == 64
    assert blocks1[0].num_tokens == 32
    assert blocks1[0].parent is None  # Largest block
    
    # req2 takes 32 tokens (the other half)
    blocks2 = manager.allocate_new_blocks("req2", 32)
    assert len(blocks2) == 1
    assert blocks2[0].size == 32
    assert blocks2[0].parent is blocks1[0]
    
    # Verify correct reclamation
    segments1 = manager.req_to_segments["req1"]
    first_block = segments1.unsealed_segment.head
    assert first_block.size == 32
    assert first_block.num_tokens == 32
    assert first_block.parent is blocks1[0]
    assert first_block.buddy is blocks2[0]
    
    # Verify pool is empty
    assert pool.slabs[64].num_free_blocks == 0
    assert pool.slabs[32].num_free_blocks == 0
    
    # 3. Free both requests to put them in free_segment_queue
    group_id = 0
    
    # Setup req1 blocks
    for i, block in enumerate(manager.req_to_segments["req1"].unsealed_segment):
        block_hash = BlockHash(f"hash1_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
    
    # Setup req2 blocks
    for i, block in enumerate(blocks2):
        block_hash = BlockHash(f"hash2_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
    req1 = MagicMock(spec=Request)
    req1.request_id = "req1"
    manager.free(req1, group_id)
    
    req2 = MagicMock(spec=Request)
    req2.request_id = "req2"
    manager.free(req2, group_id)
    
    # Verify segments are in free queue
    assert manager.free_segment_queue.num_free_segments == 2
    
    # 4. Allocate a request
    # This requires merging the two 32 blocks back into a 64 block if request_size is 64
    # or if eviction_policy is OVERPROVISION (which rounds up to max_block_size)
    blocks3 = manager.allocate_new_blocks("req3", request_size)
    
    assert len(blocks3) == 1
    
    expect_merge = (request_size == 64) or (eviction_policy == EvictionPolicy.OVERPROVISION)
    
    if expect_merge:
        assert blocks3[0].size == 64
        assert blocks3[0] is blocks1[0]
        assert first_block.parent is blocks3[0]
        assert blocks2[0].parent is blocks3[0]
        assert pool.slabs[64].num_free_blocks == 0
        assert pool.slabs[32].num_free_blocks == 0
        assert pool.allocated_blocks[64] == {blocks3[0]}
        assert pool.allocated_blocks[32] == set()
        assert manager.free_segment_queue.num_free_segments == 0
    else:
        assert blocks3[0].size == 32
        assert pool.slabs[64].num_free_blocks == 0
        assert pool.slabs[32].num_free_blocks == 0
        assert len(pool.allocated_blocks[64]) == 0
        assert len(pool.allocated_blocks[32]) == 2
        assert len(manager.cached_segments) == 1
        assert manager.free_segment_queue.num_free_segments == 1



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
