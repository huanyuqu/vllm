import pytest
from unittest.mock import MagicMock
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.semantic_segment_manager import EvictionPolicy, SemanticSegmentManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    SemanticSegment,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_request_block_hasher,
    get_segment_hash,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
    swap_blocks,
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
    return SemanticSegmentManager(block_pool=block_pool, kv_cache_group_id=0)


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


def test_allocate_new_blocks_with_block_pool():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True)
    manager = SemanticSegmentManager(
        block_pool=pool,
        kv_cache_group_id=0,
        block_size=16,
    )

    blocks = manager.allocate_new_blocks("req1", 33)
    assert len(blocks) == 3
    assert [block.size for block in blocks] == [16, 16, 16]
    assert [block.ref_cnt for block in blocks] == [1, 1, 1]

    manager.update_block_usage("req1", 33)
    assert [block.num_tokens for block in blocks] == [16, 16, 1]

    manager.seal_segment("req1")
    assert manager.req_to_segments["req1"].unsealed_segment is not None

    blocks[-1].block_hash = make_block_hash_with_group_id(
        BlockHash(b"req1_tail"),
        0,
    )
    manager.free("req1")
    assert manager.free_segment_queue.num_free_segments == 1

    reused_blocks = manager.allocate_new_blocks("req2", 64)
    assert len(reused_blocks) == 4


def test_consolidate_memory_with_block_pool():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
    manager = SemanticSegmentManager(
        block_pool=pool,
        kv_cache_group_id=0,
        block_size=16,
    )

    blocks = manager.allocate_new_blocks("req1", 48)
    manager.update_block_usage("req1", 48)
    blocks[-1].block_hash = make_block_hash_with_group_id(
        BlockHash(b"req1_tail"),
        0,
    )
    manager.seal_segment("req1")

    # Simulate a fragmented fixed-block segment. The physical block order is
    # now [1, 3, 2], so a contiguous segmented-attention span cannot represent
    # it until compaction moves it elsewhere.
    swap_blocks(blocks[1], blocks[2])
    segment = manager.req_to_segments["req1"].last_segment
    assert [block.block_id for block in segment.blocks] == [1, 3, 2]

    manager.consolidate_segment_memory("req1")

    assert segment.is_consolidated
    assert [block.block_id for block in segment.blocks] == [4, 5, 6]
    moves, swaps = manager.get_pending_moves()
    assert swaps == []
    assert moves == [
        (0, 16, 64, 16),
        (0, 48, 80, 16),
        (0, 32, 96, 16),
    ]


def test_block_pool_partial_segments_hit_multiple_prefix_segments():
    init_none_hash(sha256)
    block_size = 16
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True)
    manager = SemanticSegmentManager(
        block_pool=pool,
        kv_cache_group_id=0,
        block_size=block_size,
    )
    block_hasher = get_request_block_hasher(block_size, sha256)

    def make_request(request_id: str, token_ids: list[int]) -> Request:
        return Request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            eos_token_id=None,
            block_hasher=block_hasher,
        )

    conversation: list[int] = []
    segment_lengths = [33, 29, 41]
    for turn_index, segment_length in enumerate(segment_lengths):
        new_tokens = list(
            range(turn_index * 1000, turn_index * 1000 + segment_length)
        )
        request = make_request(f"req{turn_index}", conversation + new_tokens)
        hit_segments = manager.find_longest_cache_hit(
            request,
            request.num_tokens - 1,
            [0],
            False,
        )
        manager.touch(hit_segments[0])
        manager.save_new_computed_segments(
            request.request_id,
            hit_segments[0],
        )
        hit_tokens = sum(segment.num_tokens for segment in hit_segments[0])
        num_new_tokens = request.num_tokens - hit_tokens
        if num_new_tokens:
            manager.allocate_new_blocks(request.request_id, num_new_tokens)
            manager.update_block_usage(request.request_id, request.num_tokens)
        manager.seal_segment(request, allow_partial=True)
        manager.free(request)
        conversation = list(request.all_token_ids)

    next_request = make_request("next", conversation + [9000, 9001])
    hit_segments = manager.find_longest_cache_hit(
        next_request,
        next_request.num_tokens - 1,
        [0],
        False,
    )

    assert [segment.num_tokens for segment in hit_segments[0]] == segment_lengths


def test_seal_segment(segment_manager):
    request_id = "req1"
    
    # Allocate blocks
    blocks = segment_manager.allocate_new_blocks(request_id, 256)
    
    # Manually set block hashes as if they were computed and cached
    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
    # Seal
    segment_manager.seal_segment(request_id)
    
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
    
    # First allocation and seal
    blocks1 = segment_manager.allocate_new_blocks(request_id, 256)
    group_id = 0
    for i, block in enumerate(blocks1):
        block_hash = BlockHash(f"hash1_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
    
    segment_manager.seal_segment(request_id)
    
    # Second allocation and seal
    blocks2 = segment_manager.allocate_new_blocks(request_id, 64)
    for i, block in enumerate(blocks2):
        block_hash = BlockHash(f"hash2_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
    
    segment_manager.seal_segment(request_id)
    
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
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)
    
    # Allocate all memory for req1 (128 tokens)
    manager.allocate_new_blocks("req1", 128)
    
    # Seal and free req1
    group_id = 0
    segments = manager.req_to_segments["req1"]
    for i, block in enumerate(segments.unsealed_segment.blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)
        
    manager.free("req1")
    
    # Check that segments are in free queue
    assert manager.free_segment_queue.num_free_segments == 1

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
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)
    
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
    manager = SemanticSegmentManager(
        block_pool=pool,
        kv_cache_group_id=0,
        eviction_policy=eviction_policy,
    )
    
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
        
    manager.free("req1")
    manager.free("req2")
    
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
        
        
def test_touch(segment_manager):
    request_id = "req1"
    
    # Allocate blocks
    blocks = segment_manager.allocate_new_blocks(request_id, 256)
    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)

    segment = segment_manager.req_to_segments[request_id].last_segment
    assert segment.ref_cnt == 1
    
    # Free request to put segment in free queue
    segment_manager.free(request_id)
    
    assert segment.ref_cnt == 0
    assert segment_manager.free_segment_queue.num_free_segments == 1
    
    # Touch the segment
    segment_manager.touch([segment])
    
    assert segment.ref_cnt == 1
    assert segment_manager.free_segment_queue.num_free_segments == 0


def test_get_cached_segment_hit(segment_manager):
    request_id = "req1"

    blocks = segment_manager.allocate_new_blocks(request_id, 256)

    group_id = 0
    for i, block in enumerate(blocks):
        block_hash = BlockHash(f"hash_{i}".encode())
        block.block_hash = make_block_hash_with_group_id(block_hash, group_id)

    segment_manager.seal_segment(request_id)
    segments = segment_manager.req_to_segments[request_id]
    sealed = segments.last_segment
    assert sealed is not None
    assert sealed.is_sealed
    assert sealed.segment_hash is not None
    assert sealed.segment_hash == sealed.tail.block_hash
    assert not(sealed.segment_hash is sealed.tail.block_hash)

    # Lookup uses the group-agnostic SegmentHash.
    base_segment_hash = get_segment_hash(sealed.segment_hash)
    cached = segment_manager.get_cached_segment(base_segment_hash, [group_id])
    assert cached is not None
    assert cached[0].segment_id == sealed.segment_id


def test_cache_blocks_seal_segment(segment_manager):
    request_id = "req1"
    request = MagicMock(spec=Request)
    request.request_id = request_id
    
    # Allocate blocks
    blocks = segment_manager.allocate_new_blocks(request_id, 256)
    
    # Setup request block hashes
    group_id = 0
    block_hashes = [BlockHash(f"hash_{i}".encode()) for i in range(len(blocks))]
    request.block_hashes = block_hashes
    
    # Cache full blocks
    segment_manager.block_pool.cache_full_blocks(
        request,
        blocks,
        num_cached_blocks=0,
        num_full_blocks=len(blocks),
        kv_cache_group_id=group_id
    )
    
    # Verify blocks have hashes
    for i, block in enumerate(blocks):
        assert block.block_hash is not None
        expected = make_block_hash_with_group_id(block_hashes[i], group_id)
        assert block.block_hash == expected

    # Seal segment
    segment_manager.seal_segment(request_id)
    
    # Verify segment
    segments = segment_manager.req_to_segments[request_id]
    sealed = segments.last_segment
    assert sealed.is_sealed
    assert sealed.segment_hash is not None
    
    # Verify segment hash matches last block hash
    assert sealed.segment_hash == blocks[-1].block_hash
    

def test_split_block(block_pool):
    # Get a free block from the largest slab (128)
    assert block_pool.slabs[128].num_free_blocks > 0
    block = block_pool.slabs[128].fake_free_list_head.next_free_block
    initial_128_free = block_pool.slabs[128].num_free_blocks
    initial_64_free = block_pool.slabs[64].num_free_blocks
    initial_32_free = block_pool.slabs[32].num_free_blocks
    
    # Split the block of size 128 into a block of size 32
    result_block = block_pool.split_block(block, 32)
    
    assert result_block.size == 32
    assert result_block.block_id == block.block_id
    assert block_pool.calculate_address(result_block) == \
           block_pool.calculate_address(block)
    
    # Verify slabs
    # split_block implementation currently adds BOTH children to the slab
    # when splitting a free block.
    # 1. 128 -> 64L, 64R. Both added to slabs[64].
    # 2. 64L -> 32L, 32R. Both added to slabs[32].
    assert block_pool.slabs[128].num_free_blocks == initial_128_free - 1
    assert block_pool.slabs[64].num_free_blocks == initial_64_free + 1
    assert block_pool.slabs[32].num_free_blocks == initial_32_free + 2

    # Verify blocks are in slabs
    # result_block is 32L
    assert result_block in block_pool.slabs[32]
    assert result_block.buddy in block_pool.slabs[32]
    
    # Parent (64L) is also in slab, waiting to be used or just dangling
    assert result_block.parent not in block_pool.slabs[64]
    assert result_block.parent.buddy in block_pool.slabs[64]
    
    
def test_consolidate_segment_memory_no_change():
    pool = BuddyBlockPool(num_max_gpu_blocks=4,
                          supported_sizes=[128],
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    # Allocate contiguous blocks
    # Returns [block0, block1]
    blocks = manager.allocate_new_blocks("req", 256)

    group_id = 0
    blocks[0].block_hash = make_block_hash_with_group_id(BlockHash(b"0"),
                                                        group_id)
    blocks[1].block_hash = make_block_hash_with_group_id(BlockHash(b"1"),
                                                        group_id)
    manager.seal_segment("req")

    segment = manager.req_to_segments["req"].last_segment

    result = SemanticSegmentManager._consolidate_segment_memory(segment, pool)
    assert result is None


def test_consolidate_completed_request_cached_segment():
    pool = BuddyBlockPool(num_max_gpu_blocks=4,
                          supported_sizes=[128],
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    blocks = manager.allocate_new_blocks("req", 256)
    for i, block in enumerate(blocks):
        block.block_hash = make_block_hash_with_group_id(
            BlockHash(f"req-{i}".encode()),
            0,
        )
    manager.seal_segment("req")
    segment = manager.req_to_segments["req"].last_segment

    manager.free("req")
    assert "req" in manager.completed_req_to_segments

    manager.consolidate_segment_memory("req")

    assert segment is not None
    assert segment.is_consolidated


def test_consolidate_completed_request_after_prefix_reset_is_noop():
    pool = BuddyBlockPool(num_max_gpu_blocks=4,
                          supported_sizes=[128],
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    blocks = manager.allocate_new_blocks("req", 128)
    blocks[0].block_hash = make_block_hash_with_group_id(BlockHash(b"req"), 0)
    manager.seal_segment("req")
    manager.free("req")

    assert manager.reset_prefix_cache()

    manager.consolidate_segment_memory("req")

    assert "req" not in manager.completed_req_to_segments
    assert "req" not in manager.req_to_segments
    
    
def test_consolidate_segment_memory_move():
    # Setup a small pool to control physical addresses easily
    pool = BuddyBlockPool(num_max_gpu_blocks=4,
                          supported_sizes=[128],
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    # 1. Allocate block A for req_main (address 0)
    blocks_a = manager.allocate_new_blocks("req_main", 128)
    block_a = blocks_a[0]

    # 2. Allocate block B for a gap (address 128)
    blocks_b = manager.allocate_new_blocks("req_gap", 128)
    block_b = blocks_b[0]

    # 3. Allocate block C for req_main (address 256)
    blocks_c = manager.allocate_new_blocks("req_main", 128)
    block_c = blocks_c[0]

    # Physical addresses
    addr_a = block_a.block_id * pool.max_block_size
    addr_b = block_b.block_id * pool.max_block_size
    addr_c = block_c.block_id * pool.max_block_size

    assert addr_b == addr_a + 128
    assert addr_c == addr_b + 128

    # Free req_gap so address 128 is empty
    group_id = 0
    block_b.block_hash = make_block_hash_with_group_id(BlockHash(b"b"),
                                                       group_id)
    manager.free("req_gap")

    # Finalize the release to make it a MOVE test
    segment_b: SemanticSegment = manager.free_segment_queue.popleft()
    # We don't necessarily need to evict from cache for this test, 
    # but we must unseal and free blocks to pool.
    segment_b.unseal()
    pool.free_blocks(reversed(segment_b.blocks))

    # Seal req_main
    block_a.block_hash = make_block_hash_with_group_id(BlockHash(b"a"),
                                                       group_id)
    block_c.block_hash = make_block_hash_with_group_id(BlockHash(b"c"),
                                                       group_id)
    manager.seal_segment("req_main")

    segment = manager.req_to_segments["req_main"].last_segment
    assert segment.blocks == [block_a, block_c]

    # 4. Consolidate
    result = SemanticSegmentManager._consolidate_segment_memory(segment, pool)

    assert result is not None
    moves, swaps = result

    # It should move block_c (src) to the empty slot at addr_b
    assert len(moves) == 1
    assert len(swaps) == 0
    src, dst = moves[0]
    assert src == block_c
    # dst should be the block at addr_b (which was block_b)
    assert dst.block_id * pool.max_block_size == addr_b
    assert dst == block_b

    # Verify metadata updates
    # segment.blocks should now be [block_a, dst]
    new_blocks = segment.blocks
    assert new_blocks == [block_a, dst]
    assert dst.segment == segment
    assert dst.prev_block == block_a
    assert block_a.next_block == dst

    # Old block_c is detached and scheduled for deferred free by caller.
    assert not block_c.is_free
    assert block_c.segment is None
    assert block_c.prev_block is None
    assert block_c.next_block is None
    
    
def test_swap_blocks(segment_manager):
    # Setup: 3 blocks in one segment
    blocks = []
    for _ in range(3):
        new_blocks = segment_manager.allocate_new_blocks("req1", 128)
        blocks.extend(new_blocks)
        
    segment = segment_manager.req_to_segments["req1"].unsealed_segment
    assert len(segment.blocks) == 3
    b0, b1, b2 = segment.blocks
    
    # Check initial state
    assert b0.next_block == b1
    assert b1.prev_block == b0
    assert b1.next_block == b2
    assert b2.prev_block == b1
    assert segment.head == b0
    assert segment.tail == b2
    
    # 1. Swap non-adjacent (Head and Tail)
    swap_blocks(b0, b2)
    
    # Expected: [b2, b1, b0]
    assert segment.blocks == [b2, b1, b0]
    assert segment.head == b2
    assert segment.tail == b0
    assert b2.next_block == b1
    assert b2.prev_block is None
    assert b1.next_block == b0
    assert b1.prev_block == b2
    assert b0.next_block is None
    assert b0.prev_block == b1
    
    # 2. Swap adjacent (b2 and b1) => [b1, b2, b0]
    # Current: b2 -> b1 -> b0
    swap_blocks(b2, b1)
    
    assert segment.blocks == [b1, b2, b0]
    assert segment.head == b1
    assert segment.tail == b0
    assert b1.next_block == b2
    assert b1.prev_block is None
    assert b2.next_block == b0
    assert b2.prev_block == b1
    assert b0.next_block is None
    assert b0.prev_block == b2
    
    # 3. Swap between different segments
    # req1 segment: [b1, b2, b0]
    # Create req2 segment: [b3]
    blocks_req2 = segment_manager.allocate_new_blocks("req2", 128)
    b3 = blocks_req2[0]
    segment2 = segment_manager.req_to_segments["req2"].unsealed_segment
    
    # Swap b0 (end of seg1) with b3 (head of seg2)
    swap_blocks(b0, b3)
    
    # Expected req1: [b1, b2, b3]
    # Expected req2: [b0]
    assert segment.blocks == [b1, b2, b3]
    assert segment2.blocks == [b0]
    
    assert b0.segment == segment2
    assert b3.segment == segment
    
    assert segment.tail == b3
    assert b3.prev_block == b2
    assert b2.next_block == b3
    
    assert segment2.head == b0
    assert segment2.tail == b0
    assert b0.prev_block is None
    assert b0.next_block is None


def test_consolidate_segment_memory_swap():
    # Setup a small pool to control physical addresses easily
    pool = BuddyBlockPool(num_max_gpu_blocks=4,
                          supported_sizes=[128],
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    # 1. Allocate block A for req_main (address 0)
    blocks_a = manager.allocate_new_blocks("req_main", 128)
    block_a = blocks_a[0]

    # 2. Allocate block B for req_gap (address 128)
    blocks_b = manager.allocate_new_blocks("req_gap", 128)
    block_b = blocks_b[0]

    # 3. Allocate block C for req_main (address 256)
    blocks_c = manager.allocate_new_blocks("req_main", 128)
    block_c = blocks_c[0]

    # Physical addresses
    addr_a = block_a.block_id * pool.max_block_size
    addr_b = block_b.block_id * pool.max_block_size
    addr_c = block_c.block_id * pool.max_block_size

    assert addr_b == addr_a + 128
    assert addr_c == addr_b + 128

    # Seal req_main
    group_id = 0
    block_a.block_hash = make_block_hash_with_group_id(BlockHash(b"a"),
                                                       group_id)
    block_c.block_hash = make_block_hash_with_group_id(BlockHash(b"c"),
                                                       group_id)
    manager.seal_segment("req_main")

    segment = manager.req_to_segments["req_main"].last_segment
    assert segment.blocks == [block_a, block_c]

    # req_gap unsealed segment
    gap_unsealed = manager.req_to_segments["req_gap"].unsealed_segment
    assert gap_unsealed.blocks == [block_b]

    # 4. Consolidate
    result = SemanticSegmentManager._consolidate_segment_memory(segment, pool)

    assert result is not None
    moves, swaps = result

    # It should swap block_c (src) with block_b (dst)
    # block_b is at addr_a + 128, which is where block_c should be to be
    # contiguous with block_a.
    assert len(moves) == 0
    assert len(swaps) == 1
    assert swaps[0] == (block_c, block_b)

    # Verify metadata updates
    # segment.blocks should now be [block_a, block_b]
    new_blocks = segment.blocks
    assert new_blocks == [block_a, block_b]
    assert block_b.prev_block == block_a
    assert block_a.next_block == block_b
    assert block_b.segment == segment

    # req_gap's unsealed segment should now contain block_c
    assert gap_unsealed.blocks == [block_c]
    assert block_c.segment == gap_unsealed
    assert block_c.prev_block is None
    assert block_c.next_block is None
    
    
def test_consolidate_segment_memory_with_split():
    # Pool: supported sizes [64, 32]. 
    pool = BuddyBlockPool(num_max_gpu_blocks=4, supported_sizes=[64, 32], 
                          enable_caching=True)
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    # 1. Alloc req_main (64) -> Addr 0
    blocks_m1 = manager.allocate_new_blocks("req_main", 64)
    assert blocks_m1[0].size == 64
    assert pool.calculate_address(blocks_m1[0]) == 0

    # 2. Alloc req_gap (64) -> Addr 64
    blocks_g1 = manager.allocate_new_blocks("req_gap", 64)
    assert blocks_g1[0].size == 64
    assert pool.calculate_address(blocks_g1[0]) == 64

    # 3. Alloc req_main (32) -> Addr 128
    # Allocator prefers largest blocks, so it gives a 64 block.
    # We manually split it to simulate having a smaller block to test consolidation logic.
    blocks_m2 = manager.allocate_new_blocks("req_main", 32)
    assert blocks_m2[0].size == 64
    split_block = pool.split_block(blocks_m2[0], 32)
    assert not blocks_m2[0].is_free
    assert not pool.is_allocated(blocks_m2[0])
    assert split_block.size == 32
    assert split_block.buddy.size == 32

    segment = manager.req_to_segments["req_main"].last_segment
    # Initial State: Main [64(0), 32(128), 32(160)], Gap [64(64)]
    assert len(segment.blocks) == 3
    assert pool.calculate_address(segment.blocks[-2]) == 128
    assert pool.calculate_address(segment.blocks[-1]) == 160

    # 4. Seal
    group_id = 0
    for i, b in enumerate(segment.blocks):
        b.block_hash = make_block_hash_with_group_id(BlockHash(f"m{i}".encode()), group_id)
    manager.seal_segment("req_main")
    
    for b in blocks_g1:
        b.block_hash = make_block_hash_with_group_id(BlockHash(b"g"), group_id)
    manager.seal_segment("req_gap")

    seg_main = manager.req_to_segments["req_main"].last_segment
    seg_gap = manager.req_to_segments["req_gap"].last_segment

    # 5. Consolidate
    # Expectation:
    # - Process 64(0): OK. Next Start 64.
    # - Process 32(128):
    #   - Target at 64 is Gap(64).
    #   - Target 64(64) > Src 32(128). Split Target -> 32L(64), 32R(96).
    #   - Swap Src 32(128) with Gap 32L(64).
    #   - Main has [64(0), 32(64), 32(160)].
    #   - Gap has [32(128), 32(96)].
    # - Process 32(160):
    #   - Target at 96 is Gap 32R(96).
    #   - Swap Src 32(160) with Gap 32R(96).
    #   - Main has [64(0), 32(64), 32(96)]. Contiguous.
    #   - Gap has [32(128), 32(160)].

    moves, swaps = SemanticSegmentManager._consolidate_segment_memory(seg_main, pool)

    # 6. Validate
    assert len(moves) == 0
    assert len(swaps) == 2
    # Because the metadata has already been swapped
    # The order is the opposite of what was expected
    assert swaps[0] == (seg_gap.blocks[0], segment.blocks[1])  # 32(128) <-> 32(64)
    assert swaps[1] == (seg_gap.blocks[1], segment.blocks[2])  # 32(160) <-> 32(96)

    # seg_main should be contiguous 0, 64, 96
    blocks = seg_main.blocks
    assert blocks[0].size == 64 and pool.calculate_address(blocks[0]) == 0
    assert blocks[1].size == 32 and pool.calculate_address(blocks[1]) == 64
    assert blocks[2].size == 32 and pool.calculate_address(blocks[2]) == 96

    # GAP should be split and moved
    # Gap originally had 1 block of 64. Now logically it should have 2 blocks of 32
    assert len(seg_gap.blocks) == 2
    addrs = [pool.calculate_address(b) for b in seg_gap.blocks]
    assert addrs == [128, 160]


def test_seal_segment_without_tail_hash_does_not_crash_or_cache():
    pool = BuddyBlockPool(
        num_max_gpu_blocks=4,
        supported_sizes=[16, 32],
        enable_caching=True,
    )
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    manager.allocate_new_blocks(request_id="req", num_tokens=8)
    segment = manager.req_to_segments["req"].unsealed_segment
    assert segment is not None
    assert segment.tail is not None
    assert segment.tail.block_hash is None

    # Regression: this used to raise when tail.block_hash is None.
    manager.seal_segment("req")

    sealed = manager.req_to_segments["req"].last_segment
    assert sealed is not None
    assert sealed.is_sealed
    assert sealed.segment_hash is None
    assert len(manager.cached_segments) == 0

    # Free path should also be robust when sealed segment has no segment_hash.
    manager.free("req")


def test_seal_segment_partial_tail_computes_hash_and_caches():
    init_none_hash(sha256)

    pool = BuddyBlockPool(
        num_max_gpu_blocks=4,
        supported_sizes=[16, 32],
        enable_caching=True,
    )
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    request = Request(
        request_id="req_partial",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(16, sha256),
    )

    manager.allocate_new_blocks(request_id=request.request_id, num_tokens=8)
    manager.seal_segment(request)

    sealed = manager.req_to_segments[request.request_id].last_segment
    assert sealed is not None
    assert sealed.is_sealed
    assert sealed.segment_hash is not None
    assert len(manager.cached_segments) == 1

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, request.num_tokens, 0)
    expected = hash_block_tokens(
        sha256,
        None,
        request.all_token_ids[:request.num_tokens],
        extra_keys,
    )
    assert get_block_hash(sealed.segment_hash) == expected


def test_find_longest_cache_hit_matches_partial_tail():
    init_none_hash(sha256)

    pool = BuddyBlockPool(
        num_max_gpu_blocks=4,
        supported_sizes=[16, 32],
        enable_caching=True,
    )
    manager = SemanticSegmentManager(block_pool=pool, kv_cache_group_id=0)

    request = Request(
        request_id="req_partial",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    manager.allocate_new_blocks(request_id=request.request_id, num_tokens=8)
    manager.update_block_usage(request.request_id, request.num_tokens)
    manager.seal_segment(request)

    next_request = Request(
        request_id="req_next",
        prompt_token_ids=list(request.all_token_ids) + [9, 10],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(16, sha256),
    )

    hit_segments = manager.find_longest_cache_hit(
        request=next_request,
        max_length=next_request.num_tokens - 1,
        kv_cache_group_ids=[0],
        use_eagle=False,
    )
    assert len(hit_segments[0]) == 1
    assert hit_segments[0][0].num_tokens == request.num_tokens


def test_block_pool_completed_partial_tail_caches_and_hits():
    init_none_hash(sha256)

    pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
    manager = SemanticSegmentManager(
        block_pool=pool,
        kv_cache_group_id=0,
        block_size=16,
    )

    request = Request(
        request_id="req_partial",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(16, sha256),
    )

    manager.allocate_new_blocks(request_id=request.request_id, num_tokens=8)
    manager.update_block_usage(request.request_id, request.num_tokens)

    assert not manager.seal_segment(request)
    manager.free(request)

    sealed = manager.completed_req_to_segments[request.request_id][0]
    assert sealed.is_sealed
    assert sealed.segment_hash is not None
    assert len(manager.cached_segments) == 1

    next_request = Request(
        request_id="req_next",
        prompt_token_ids=list(request.all_token_ids) + [9, 10],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(16, sha256),
    )

    hit_segments = manager.find_longest_cache_hit(
        request=next_request,
        max_length=next_request.num_tokens - 1,
        kv_cache_group_ids=[0],
        use_eagle=False,
    )
    assert len(hit_segments[0]) == 1
    assert hit_segments[0][0].num_tokens == request.num_tokens
