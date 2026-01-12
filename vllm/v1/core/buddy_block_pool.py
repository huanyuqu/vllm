"""
Buddy Memory Allocator for KV Cache with Variable Block Sizes.

This module implements a multi-slab memory management system where blocks can be
split and merged dynamically. It supports multiple block sizes (e.g., 16, 32, 64, 128 tokens)
organized in slabs, with a buddy tree for tracking split relationships.

Each slab is simply a FreeKVCacheBlockQueue that manages free blocks of a specific size.
This directly reuses the existing doubly linked list implementation with O(1) middle
removal support, which is essential for prefix caching.
"""
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    FreeKVCacheBlockQueue, 
    KVCacheBlock, 
    BuddyTreeBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    replace_block_in_segment,
)
from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.distributed.kv_events import MEDIUM_GPU, BlockRemoved, BlockStored, KVCacheEvent
from vllm.v1.request import Request

logger = init_logger(__name__)


class BuddyBlockPool:
    """
    Buddy Memory Allocator for variable-sized KV cache blocks.
    
    This allocator maintains TWO separate data structures:
    
    1. Buddy Tree: Pre-creates all potential blocks at initialization to track
       hierarchical split/merge relationships. All blocks exist in the tree,
       but only blocks in slabs are available for allocation.
    
    2. Free Block Queues (Slabs): Each size has a FreeKVCacheBlockQueue.
       A block is "available" if and only if it's in a slab. Initially, only
       max-size blocks are in slabs. Splitting moves child blocks into slabs;
       merging removes blocks from slabs and moves parent back to its slab.
    
    This class provides a compatible interface with BlockPool while supporting
    variable-sized blocks through buddy memory allocation.
    """
    
    def __init__(
        self,
        num_max_gpu_blocks: int,
        supported_sizes: list[int],
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        """
        Initialize the Buddy Block Allocator.
        
        Args:
            num_gpu_blocks: Total number of blocks at max_block_size
            supported_sizes: List of supported block sizes
            enable_caching: Whether to enable prefix caching (for compatibility)
            enable_kv_cache_events: Whether to enable KV cache events (for compatibility)
        """
        self.supported_sizes = supported_sizes
        self._complete_supported_sizes()
        self.num_max_gpu_blocks = num_max_gpu_blocks
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events

        # Map coordinate (block_id, size, relative_id) to BuddyTreeBlock
        self._blocks: dict[tuple[int, int, int], BuddyTreeBlock] = {}
        # Initialize free block queues (slabs) for each supported size
        # Each queue is a FreeKVCacheBlockQueue that manages blocks of that size
        self.slabs = self._initialize_block_pool(num_max_gpu_blocks)
        self.num_total_tokens = num_max_gpu_blocks * self.max_block_size
        self.num_free_tokens = self.num_total_tokens

        # TODO(huanyu): The best way to record allocated blocks is to use a
        # min-heap based on their current token usage for better reclamation.
        # However, this increases complexity from O(1) to O(log n).
        # For now, we use a simple set for allocated blocks.
        self.allocated_blocks: dict[int, set[BuddyTreeBlock]] = {
            size: set() for size in self.supported_sizes
        }

        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        
        # TODO(huanyu): implement KV events
        self.kv_event_queue: list[KVCacheEvent] = []
        
    def _check_supported_sizes(self) -> None:
        """
        Validate that all supported sizes are powers of 2.
            
        Raises:
            ValueError: If any size is not a power of 2
        """
        if not self.supported_sizes:
            raise ValueError("supported_sizes cannot be empty")

        for size in self.supported_sizes:
            if size <= 0 or (size & (size - 1)) != 0:
                raise ValueError(
                    f"Block size {size} must be a positive power of 2. "
                    f"Got sizes: {self.supported_sizes}"
                )

    def _complete_supported_sizes(self) -> None:
        """
        Complete the supported_sizes list by adding missing powers of 2
        between the minimum and maximum sizes.
        
        For example, if supported_sizes is [128, 32, 8], it will be
        completed to [128, 64, 32, 16, 8].
        """
        self._check_supported_sizes()
        
        self.supported_sizes = sorted(set(self.supported_sizes), reverse=True)
                
        self.min_block_size = self.supported_sizes[-1]
        self.max_block_size = self.supported_sizes[0]
        
        completed_sizes = []
        current_size = self.max_block_size
        while current_size >= self.min_block_size:
            completed_sizes.append(current_size)
            current_size //= 2
        
        self.supported_sizes = tuple(completed_sizes)

    def _initialize_block_pool(
        self, num_blocks: int
    ) -> dict[int, FreeKVCacheBlockQueue]:
        """
        Initialize buddy tree and slabs.

        Strategy:
        1. Pre-create all potential blocks in the buddy tree for all sizes
        2. Establish parent-child relationships in the tree
        3. Only add max-size blocks to their slab (making them available)
        4. Smaller blocks exist in the tree but not in slabs (unavailable until split)
        
        Args:
            num_blocks: Number of blocks to create at max size
            
        Returns:
            slabs: A dictionary mapping block sizes to their FreeKVCacheBlockQueue (slab)
        """
        # First pass: create all blocks in the buddy tree
        for size in self.supported_sizes:
            size_ratio = self.max_block_size // size
            total_blocks = num_blocks * size_ratio
            
            for i in range(total_blocks):
                idx = i // size_ratio
                relative_id = i % size_ratio
                
                block = BuddyTreeBlock(
                    block_id=idx,
                    relative_id=relative_id,
                    size=size,
                )
                self._blocks[block.full_id] = block
        
        # Second pass: establish parent-child relationships
        for size in self.supported_sizes[1:]:  # Skip max size (no parent)
            size_ratio = self.max_block_size // size
            total_blocks = num_blocks * size_ratio
            
            for i in range(total_blocks):
                idx = i // size_ratio
                relative_id = i % size_ratio
                
                block = self._blocks[(idx, size, relative_id)]
                
                # Find parent
                parent_size = size * 2
                parent_relative_id = relative_id // 2
                parent_block = self._blocks[(idx, parent_size, parent_relative_id)]
                
                block.parent = parent_block
                
                # Link as left or right child
                if relative_id % 2 == 0:
                    parent_block.left_child = block
                else:
                    parent_block.right_child = block
        
        # Third pass: create slabs and only add max-size blocks
        slabs: dict[int, FreeKVCacheBlockQueue] = {}
        
        for size in self.supported_sizes:
            if size == self.max_block_size:
                # Add all max-size blocks to slab (available for allocation)
                max_blocks: list[KVCacheBlock] = [
                    self._blocks[(idx, size, 0)] for idx in range(num_blocks)
                ]
                slabs[size] = FreeKVCacheBlockQueue(max_blocks)
            else:
                # Create empty slab for smaller sizes
                slabs[size] = FreeKVCacheBlockQueue([])

        return slabs
    
    def get_usage(self):
        raise 1.0 - (self.get_num_free_tokens() / self.num_total_tokens)
    
    def get_num_free_tokens(self):
        return self.num_free_tokens
    
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> Optional[list[BuddyTreeBlock]]:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks: list[BuddyTreeBlock] = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[BuddyTreeBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

    def _maybe_evict_cached_block(self, block: BuddyTreeBlock) -> bool:
        """
        Evict the block from the cache if it is cached in `cached_block_hash_to_block`,
        reset its hash metadata, and attempt to merge it with its buddy if possible.

        Args:
            block: The block to evict and potentially merge.

        Returns:
            True if the block was evicted from the cache, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()
        
        # Block might have been merged already
        if (block.prev_free_block is None and
            block.next_free_block is None):
            return True
            
        self._try_merge(block)
            
        return True

    def get_new_blocks(self, num_tokens: int):
        """
        Allocate blocks to hold `num_tokens`.
        
        Args:
            num_tokens: Number of tokens to allocate blocks for
            
        Returns:
            A tuple of (allocated_blocks: list[BuddyTreeBlock], remaining_tokens: int)
            where allocated_blocks contains the successfully allocated blocks 
            ([] if no available blocks), and remaining_tokens is the number of 
            tokens still needing allocation (0 if fully satisfied).
        """
        assert num_tokens > 0, "num_tokens must be positive"
        
        # Try to allocate from slabs, preferring larger sizes
        blocks, remaining_tokens = self._allocate_largest_blocks(num_tokens)
        
        if blocks:
            if self.enable_caching:
                for block in blocks:
                    self._maybe_evict_cached_block(block)
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
            else:
                for block in blocks:
                    assert block.ref_cnt == 0
                    block.ref_cnt += 1
                
        return blocks, max(remaining_tokens, 0)
        
    # TODO(huanyu): This method needs to consider the impact on hash after reclaim
    def reclaim_new_blocks(self, num_tokens: int) -> list[BuddyTreeBlock]:
        # No free blocks available, try to reclaim space from allocated blocks
        remaining_tokens = self._reclaim_from_allocated_blocks(num_tokens)
        
        if remaining_tokens > 0:
            raise ValueError("Failed to allocate blocks: insufficient memory")
        
        # Successfully reclaimed blocks for remaining tokens
        # Now try allocation again
        reclaimed_blocks, remaining_tokens = self._allocate_largest_blocks(
            remaining_tokens
        )
        
        assert remaining_tokens <= 0, "Allocation should be satisfied after reclamation"
        assert reclaimed_blocks is not None, "reclaimed_blocks should not be None after reclamation"
        
        if self.enable_caching:
            for block in reclaimed_blocks:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in reclaimed_blocks:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                
        return reclaimed_blocks
    
    def update_block_usage(self, block_id: int, size: int, relative_id: int,
                           num_tokens_used: int) -> None:
        """
        Update the token usage for a block.
        
        This should be called by the system when tokens are written to a block
        to track actual usage for potential reclamation.
        
        Args:
            block_id: ID of the block
            size: Size of the block
            relative_id: Relative ID within the block's size category
            num_tokens_used: Number of tokens currently stored in the block
        """
        coord = (block_id, size, relative_id)
        if coord in self._blocks:
            self._blocks[coord].num_tokens = num_tokens_used
            self.num_free_tokens -= ((num_tokens_used + self.min_block_size - 1) // self.min_block_size) * self.min_block_size
        else:
            raise KeyError(f"Block {coord} not found")

    def _allocate_largest_blocks(self, num_tokens: int):
        """
        Allocate blocks from the largest available slabs until the request is satisfied.

        This method repeatedly requests the largest available free block via
        _allocate_largest_block() and subtracts its capacity from num_tokens.
        It prefers larger slabs first and continues until either the requested
        number of tokens is covered or no more blocks can be obtained.

        Args:
            num_tokens: Number of tokens to allocate.

        Returns:
            blocks: list[BuddyTreeBlock] of allocated blocks if allocation
            made progress (may be a single block), or None if no allocation
            was possible.
            remaining_tokens: number of tokens still needing allocation
            after the attempted allocations (0 if fully satisfied).
        """
        blocks: list[BuddyTreeBlock] = []

        # Allocate the first block
        block = self._allocate_largest_block()
        if block is None:
            return blocks, num_tokens

        remaining_tokens = num_tokens - block.size
        blocks.append(block)
        while remaining_tokens > 0 and block is not None:
            block = self._allocate_largest_block()
            if block is not None:
                remaining_tokens -= block.size
                blocks.append(block)

        return blocks, remaining_tokens
    
    def _allocate_largest_block(self) -> Optional[BuddyTreeBlock]:
        """
        Allocate a block from the largest available slab.

        Returns:
            BuddyTreeBlock if successful, None otherwise
        """
        for size in self.supported_sizes:
            block = self._allocate_block(size)
            if block is not None:
                return block
        return None
    
    def _allocate_block(self, size: int) -> Optional[BuddyTreeBlock]:
        """
        Allocates a block from the slab corresponding to the given size.
        
        Args:
            size (int): The size of the block to allocate.
            
        Returns:
            Optional[BuddyTreeBlock]: The allocated block if available, otherwise None.
        """
        slab = self.slabs[size]
        if slab.num_free_blocks > 0:
            block: BuddyTreeBlock = slab.popleft()  # type: ignore
            block = self._try_merge(block)
            self.allocated_blocks[block.size].add(block)
            return block
        else:
            return None
    
    def _reclaim_from_allocated_blocks(self, num_tokens: int) -> int:
        """
        Repeatedly call _reclaim_one_block and try larger slabs first.

        Returns:
            Remaining number of tokens that still need allocation (0 if satisfied).
        """
        remaining = num_tokens

        # Keep trying while there is work and progress can still be made
        progress_made = True
        while remaining > 0 and progress_made:
            progress_made = False

            for size in self.supported_sizes:
                if remaining <= 0:
                    break

                if size not in self.allocated_blocks or not self.allocated_blocks[size]:
                    continue

                while remaining > 0:
                    new_remaining = self._reclaim_one_block(size, remaining)
                    if new_remaining < remaining:
                        progress_made = True
                        remaining = new_remaining
                    else:
                        break

                if remaining <= 0:
                    break

            # If a full pass made no progress, stop to avoid infinite loop
            if not progress_made:
                break

        return remaining
    
    def _reclaim_one_block(
        self, size: int, num_tokens: int
    ) -> int:
        """
        Try to reclaim unused space from blocks in the specified slab.
        
        Find a block in the slab that can be split to free up space.
        
        Args:
            size: The size of the slab to reclaim from
            num_tokens: Number of tokens to accommodate
            
        Returns:
            The remaining number of tokens that still need reclamation
        """
        slab = self.allocated_blocks[size]

        # If no allocated blocks in this slab can be split to reclaim space,
        # return immediately to avoid unnecessary work.
        if not any(block.num_tokens <= size - self.min_block_size 
                   for block in slab):
            return num_tokens
        
        while slab:
            reclaimed_block = slab.pop()
            # Get current usage for this block
            num_tokens_used = reclaimed_block.num_tokens
            if num_tokens_used > size - self.min_block_size:
                # Not enough space can be reclaimed
                slab.add(reclaimed_block)
                continue
            
            # Build a list of supported sizes (< current size), descending
            idx = self.supported_sizes.index(size) + 1
            supported_sizes = self.supported_sizes[idx:]

            # Pick sizes (large to small) until we cover num_tokens_used
            needed_sizes: list[int] = []
            total = 0
            for s in supported_sizes:
                if total < num_tokens_used:
                    needed_sizes.append(s)
                    total += s
                else:
                    break
            
            # TODO(huanyu): how to link the original request to the new blocks
            self._split_block(reclaimed_block, needed_sizes)
            
            return num_tokens - needed_sizes[-1]

    def split_block(self, block: BuddyTreeBlock, 
                    target_size: int) -> BuddyTreeBlock:
        """
        Split a block down to the target size.
        If the block is free, the right remainder is returned to the free slab.
        If the block is allocated, the right remainder remains allocated.
        
        Args:
            block: The block to split.
            target_size: The target size to split down to.
            
        Returns:
            The left-most child block at the target size.
        """
        if block.size == target_size:
            return block
            
        if block.size < target_size:
            raise ValueError(f"Cannot split block of size {block.size} to {target_size}")
            
        # If free, verify it's in slab and remove it
        if block.is_free:
            self.slabs[block.size].remove(block)
            
        # We split step by step
        while block.size > target_size:
            child_size = block.size // 2
            left_rel = block.relative_id * 2
            right_rel = left_rel + 1
            
            left_child = self._blocks[(block.block_id, child_size, left_rel)]
            right_child = self._blocks[(block.block_id, child_size, right_rel)]
            
            # If parent was allocated (has segment), children inherit it
            seg = block.segment
            if seg:
                # Distribute tokens
                if block.num_tokens > child_size:
                    left_child.num_tokens = child_size
                    right_child.num_tokens = block.num_tokens - child_size
                else:
                    left_child.num_tokens = block.num_tokens
                    right_child.num_tokens = 0
                
                # Both remain allocated to the segment
                self.allocated_blocks[child_size].add(left_child)
                self.allocated_blocks[child_size].add(right_child)
                replace_block_in_segment(block, [left_child, right_child])
            else:
                self.slabs[child_size].append(left_child)
                self.slabs[child_size].append(right_child)
            
            block.reset()
            if block.size in self.allocated_blocks:
                self.allocated_blocks[block.size].discard(block)
                
            block = left_child
            
        return block
    
    def _split_block(
        self, 
        parent_block: BuddyTreeBlock, 
        needed_sizes: list[int]
    ) -> None:
        """
        Split parent_block into child blocks and add them to slabs.
        
        The left children are allocated (added to allocated_blocks),
        while the rightmost child goes to its free slab.

        Args:
            parent_block: The block to split (must already be in allocated_blocks)
            needed_sizes: List of sizes for the descendant blocks, in descending order
        """
        block_id = parent_block.block_id
        total_tokens = parent_block.num_tokens

        # Split from parent_size down to each needed size
        # e.g., parent_size=64, needed_sizes=[32, 16] means:
        #   1. Split 64 -> two 32s: allocate left, keep splitting right
        #   2. Split remaining 32 -> two 16s: allocate left, free right
        current_parent = parent_block
        allocated_children: list[BuddyTreeBlock] = []
        
        for child_size in needed_sizes:
            parent_relative_id = current_parent.relative_id
            left_relative_id = parent_relative_id * 2
            right_relative_id = left_relative_id + 1

            left_child = self._blocks.get((block_id, child_size, left_relative_id))
            right_child = self._blocks.get((block_id, child_size, right_relative_id))

            if left_child is None or right_child is None:
                raise KeyError(
                    f"Child blocks not found for block_id={block_id}, "
                    f"size={child_size}, left_rel={left_relative_id}, right_rel={right_relative_id}"
                )

            # Allocate left child
            left_child.ref_cnt = parent_block.ref_cnt
            left_child.num_tokens = min(child_size, total_tokens)
            total_tokens -= left_child.num_tokens
            self.allocated_blocks[child_size].add(left_child)
            allocated_children.append(left_child)
            
            # Handle right child
            if child_size == needed_sizes[-1]:
                # Last split: right child goes to slab as free block
                right_child.reset()
                self.slabs[child_size].append(right_child)
            else:
                # Continue splitting the right child
                right_child.ref_cnt = current_parent.ref_cnt
                current_parent = right_child

        # Update segment if parent block belongs to one
        replace_block_in_segment(parent_block, allocated_children)
            
        # Reset parent block
        parent_block.reset()
        self.allocated_blocks[parent_block.size].discard(parent_block)

    def touch(self, blocks: tuple[list[BuddyTreeBlock], ...]) -> None:
        """
        Touch blocks to increase their reference count.
        
        Similar to BlockPool.touch(), this increases the reference count
        and removes blocks from the free list if they were there.
        This prevents cached blocks from being evicted when they are
        hit by new requests.
        
        Args:
            blocks: Tuple of block sequences, one per KV cache group.
                   Each element is a list of KVCacheBlock objects.
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # If ref_cnt is 0, the block is in the free list
                if block.ref_cnt == 0:
                    self.slabs[block.size].remove(block)
                block.ref_cnt += 1
    
    def free_blocks(self, ordered_blocks: Iterable[BuddyTreeBlock]) -> None:
        """
        Free a list of blocks.

        The blocks should be ordered by their eviction priority, where the
        first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free, ordered by eviction priority
        """
        for block in ordered_blocks:
            if block.is_sealed:
                raise ValueError(f"Cannot free sealed block {block}")
            
            size = block.size
            if (size not in self.allocated_blocks or 
                block not in self.allocated_blocks[size]):
                raise ValueError(f"Block {block} not found in allocated blocks for size {size}")

            block.ref_cnt -= 1
            if block.ref_cnt > 0:
                continue

            self.allocated_blocks[size].discard(block)
            block.num_tokens = 0
            self.slabs[block.size].append(block)
    
    def _try_merge(self, block: BuddyTreeBlock) -> BuddyTreeBlock:
        """
        Try to iteratively merge `block` with its free buddy.
        If the buddy is free (in slab), remove it and promote parent.
        Repeat until no further merge is possible or we reach the root.

        Args:
            block: The block to start merging from
        """
        current = block

        while True:
            buddy = current.buddy
            if buddy is None:
                break

            # Buddy must be available and free (i.e., currently in its slab) to merge.
            if not buddy.is_in_slab:
                break

            self.slabs[buddy.size].remove(buddy)

            # Promote to parent and continue attempting to merge upward.
            current = current.parent

        return current
    
    def reset_prefix_cache(self):
        raise NotImplementedError
    
    def is_allocated(self, block: BuddyTreeBlock) -> bool:
        """
        Check if a block is currently allocated.

        Args:
            block: The BuddyTreeBlock to check.
            
        Returns:
            True if the block is allocated, False otherwise.
        """
        return (block.size in self.allocated_blocks and 
                block in self.allocated_blocks[block.size])


    def calculate_address(self, block: BuddyTreeBlock) -> int:
        """
        Calculate the starting address of a block in token units.

        Args:
            block: The BuddyTreeBlock to calculate the address for.
            max_block_size: The maximum block size in the BuddyBlockPool.

        Returns:
            The starting address of the block in token units.
        """
        address = block.block_id * self.max_block_size
        address += block.relative_id * block.size
        return address