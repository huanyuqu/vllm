"""
Buddy Memory Allocator for KV Cache with Variable Block Sizes.

This module implements a multi-slab memory management system where blocks can be
split and merged dynamically. It supports multiple block sizes (e.g., 16, 32, 64, 128 tokens)
organized in slabs, with a buddy tree for tracking split relationships.

Each slab is simply a FreeKVCacheBlockQueue that manages free blocks of a specific size.
This directly reuses the existing doubly linked list implementation with O(1) middle
removal support, which is essential for prefix caching.
"""
from collections import deque
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue, 
    KVCacheBlock, 
    BuddyTreeBlock,
)

logger = init_logger(__name__)


class BuddyBlockPool:
    """
    Buddy Memory Allocator for variable-sized KV cache blocks.
    
    This allocator maintains TWO separate data structures:
    
    1. Buddy Tree: Tracks the hierarchical split/merge relationships of blocks.
       Tree nodes may not correspond to actual usable blocks (e.g., a parent
       node that has been split exists only for tracking merge opportunities).
    
    2. Free Block Queues (Slabs): Each size has a FreeKVCacheBlockQueue that
       manages actual free KVCacheBlock objects.
    
    This class provides a compatible interface with BlockPool while supporting
    variable-sized blocks through buddy memory allocation.
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
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
        self.supported_sizes = sorted(supported_sizes, reverse=True)
        self._check_supported_sizes()
        self._complete_supported_sizes()
        self.max_block_size = self.supported_sizes[0]
        self.min_block_size = self.supported_sizes[-1]
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events

        # Map coordinate (block_id, size, relative_id) to BuddyTreeBlock
        self._blocks: dict[tuple[int, int, int], BuddyTreeBlock] = {}
        # Initialize free block queues (slabs) for each supported size
        # Each queue is a FreeKVCacheBlockQueue that manages blocks of that size
        self.slabs = self._initialize_block_pool(num_gpu_blocks)
        # Allocated blocks stored as a min-heap: (num_tokens, block)
        # TODO(huanyu): The best way to record allocated blocks is to use a
        # min-heap based on their current token usage for better reclamation.
        # However, this increases complexity from O(1) to O(log n).
        # For now, we use a simple queue for allocated blocks.
        self.allocated_blocks: dict[int, deque[BuddyTreeBlock]] = {}

        logger.info(
            f"BuddyBlockPool initialized with {num_gpu_blocks} blocks of "
            f"size {self.max_block_size}, supporting sizes: {supported_sizes}"
        )
        
    def _check_supported_sizes(self) -> None:
        """
        Validate that all supported sizes are powers of 2.
            
        Raises:
            ValueError: If any size is not a power of 2
        """
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
        if not self.supported_sizes:
            return
        
        min_size = self.supported_sizes[-1]
        max_size = self.supported_sizes[0]
        
        completed_sizes = []
        current_size = min_size
        while current_size <= max_size:
            completed_sizes.append(current_size)
            current_size *= 2
        
        self.supported_sizes = sorted(completed_sizes, reverse=True)

    def _initialize_block_pool(
        self, num_blocks: int
    ) -> dict[int, FreeKVCacheBlockQueue]:
        """
        Initialize all slabs with virtual blocks and establish buddy tree relationships.

        For the maximum block size, create actual blocks.
        For smaller sizes, create virtual blocks that represent potential splits.
        Build the buddy tree hierarchy where each max block has child blocks with the same block_id but different relative_block_ids.
        
        Args:
            num_blocks: Number of blocks to create at max size
            
        Returns:
            slabs: A dictionary mapping block sizes to their FreeKVCacheBlockQueue (slab)
        """
        slabs: dict[int, FreeKVCacheBlockQueue] = {}
        
        for size in self.supported_sizes:
            blocks: list[BuddyTreeBlock] = []
            
            if size == self.max_block_size:
                # For max size, create actual root blocks
                for idx in range(num_blocks):
                    block = BuddyTreeBlock(
                        block_id=idx,
                        relative_id=0,
                        size=size,
                        is_virtual=False
                    )
                    blocks.append(block)
                    self._blocks[(idx, size, 0)] = block
            else:
                # For smaller sizes, create virtual blocks and link to parents
                size_ratio = self.max_block_size // size
                total_virtual_blocks = num_blocks * size_ratio
                
                for i in range(total_virtual_blocks):
                    idx = i // size_ratio
                    relative_id = i % size_ratio
                    
                    block = BuddyTreeBlock(
                        block_id=idx,
                        relative_id=relative_id,
                        size=size,
                        is_virtual=True
                    )
                    blocks.append(block)
                    self._blocks[(idx, size, relative_id)] = block
                    
                    # Establish parent-child relationships in buddy tree
                    parent_size = size * 2
                    parent_relative_id = relative_id // 2
                    parent_coord = (idx, parent_size, parent_relative_id)
                    parent_block = self._blocks.get(parent_coord)
                    
                    if parent_block:
                        block.parent = parent_block
                        child_position = relative_id % 2
                        if child_position == 0:
                            parent_block.left_child = block
                        else:
                            parent_block.right_child = block
                    else:
                        raise RuntimeError(
                            f"Parent block not found for block_id={idx}, "
                            f"size={size}, relative_id={relative_id}"
                        )
            
            slabs[size] = FreeKVCacheBlockQueue(blocks)

        return slabs

    def get_new_blocks(self, num_tokens: int) -> list[BuddyTreeBlock]:
        """
        Allocate blocks to hold num_tokens.
        
        Strategy:
        1. Try to allocate from the largest available slab first
        2. If no blocks available in any slab, try to reclaim space from 
           allocated blocks by splitting them
        3. Split reclaimed blocks and return freed portions to their slabs
        
        Args:
            num_tokens: Number of tokens to allocate blocks for
            
        Returns:
            BuddyTreeBlocks if allocation succeeds, None otherwise
        """
        assert num_tokens > 0, "num_tokens must be positive"
        
        # Try to allocate from slabs, preferring larger sizes
        blocks, remaining_tokens = self._allocate_largest_blocks(num_tokens)

        if blocks is not None and remaining_tokens <= 0:
            # Successfully allocated from block pool
            return blocks
        
        # No free blocks available, try to reclaim space from allocated blocks
        remaining_tokens = self._reclaim_from_allocated_blocks(remaining_tokens)
        
        if remaining_tokens > 0:
            raise ValueError("Failed to allocate blocks: insufficient memory")
        
        # Successfully reclaimed blocks for remaining tokens
        # Now try allocation again
        reclaimed_blocks, remaining_tokens = self._allocate_largest_blocks(
            remaining_tokens
        )
        
        assert remaining_tokens <= 0, "Allocation should be satisfied after reclamation"
        
        return blocks + reclaimed_blocks
    
    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
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
                    # Find which queue this block belongs to
                    for size, queue in self.slabs.items():
                        try:
                            queue.remove(block)  # O(1) removal from middle
                            break
                        except RuntimeError:
                            # Block not in this queue, continue
                            continue
                
                # Increase reference count
                block.ref_cnt += 1
    
    def update_block_usage(self, block_id: int, num_tokens_used: int) -> None:
        """
        Update the token usage for a block.
        
        This should be called by the system when tokens are written to a block
        to track actual usage for potential reclamation.
        
        Args:
            block_id: ID of the block
            num_tokens_used: Number of tokens currently stored in the block
        """
        self.block_usage[block_id] = num_tokens_used
        logger.debug(f"Updated block {block_id} usage to {num_tokens_used} tokens")

    def _allocate_largest_blocks(
        self, num_tokens: int
    ) -> tuple[Optional[list[BuddyTreeBlock]], int]:
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
            return None, num_tokens

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
            block: BuddyTreeBlock = slab.popleft()
            self.allocated_blocks[size].append(block)
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
                slab.appendleft(reclaimed_block)
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
    
    def _split_block(
        self, 
        parent_block: BuddyTreeBlock, 
        needed_sizes: list[int]
    ) -> None:
        """
        Materialize virtual descendant blocks of parent_block according
        to needed_sizes and place them into their corresponding slabs.

        Args:
            parent_block: The block to split
            needed_sizes: List of sizes for the descendant blocks to create
        """
        block_id = parent_block.block_id
        parent_block.is_split = True
        parent_block.is_virtual = True
        total_tokens = parent_block.num_tokens

        # Split from parent_size down to each needed size
        # e.g., parent_size=64, needed_sizes=[32, 16] means:
        #   1. Split 64 -> two 32s: allocate one, keep splitting the other
        #   2. Split remaining 32 -> two 16s: allocate one, free the other
        current_parent = parent_block
        
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

            # Allocate left child to the original request
            left_child.is_virtual = False
            left_child.ref_cnt = parent_block.ref_cnt
            left_child.num_tokens = min(child_size, total_tokens)
            total_tokens -= left_child.num_tokens
            # TODO(huanyu): link left_child to the original request
            self.allocated_blocks[child_size].append(left_child)
            
            # Right child becomes the next block to split (or goes to slab if last)
            if child_size == needed_sizes[-1]:
                # Last split: right child goes to its slab as a free block
                right_child.is_virtual = False
                right_child.ref_cnt = 0
                right_child.num_tokens = 0
                self.slabs[child_size].append(right_child)
            else:
                # Continue splitting the right child
                right_child.is_virtual = True
                right_child.is_split = True
                current_parent = right_child
    
    def free_blocks(self, ordered_blocks: list[KVCacheBlock]) -> None:
        """
        Free a list of blocks and attempt to merge with buddies.
        
        The blocks should be ordered by their eviction priority, where the 
        first block will be evicted first.
        
        Args:
            ordered_blocks: A list of blocks to free, ordered by eviction priority
        """
        for block in ordered_blocks:
            block_id = block.block_id
            
            # Check if block is allocated
            if block_id not in self.allocated_blocks:
                continue
            
            # Decrease reference count
            block.ref_cnt -= 1
            
            # Only process further if ref_cnt reaches 0
            if block.ref_cnt > 0:
                continue
            
            # Remove from allocated blocks
            _, size = self.allocated_blocks.pop(block_id)
            
            # Try to merge with buddy (updates buddy tree)
            merged_block, merged_size = self._try_merge(block, size)
            
            # Add the final block to appropriate queue
            queue = self.slabs[merged_size]
            queue.append(merged_block)
            
            logger.debug(
                f"Freed block {block_id}, final block {merged_block.block_id}, "
                f"final size: {merged_size}"
            )
    
    def _try_merge(
        self, 
        block: KVCacheBlock, 
        block_size: int
    ) -> tuple[KVCacheBlock, int]:
        """
        Recursively merge a block with its buddy if possible.
        
        Args:
            block: The block to merge
            block_size: Current size of the block
            
        Returns:
            Tuple of (merged_block, merged_size)
        """
        tree_node = self.buddy_tree[block.block_id]
        
        if not tree_node.can_merge_with_buddy:
            return block, block_size
        
        parent_tree_node = tree_node.parent
        buddy_tree_node = (
            parent_tree_node.right_child 
            if parent_tree_node.left_child == tree_node 
            else parent_tree_node.left_child
        )
        
        # buddy_tree_node is itself the buddy block
        buddy_block = buddy_tree_node
        
        # Remove buddy from its queue's free list
        buddy_queue = self.slabs[block_size]
        try:
            buddy_queue.remove(buddy_block)  # O(1) removal from middle
        except RuntimeError:
            # Buddy was not in free list (shouldn't happen but handle gracefully)
            logger.warning(
                f"Attempted to merge with buddy block {buddy_block.block_id} "
                "which was not in free list"
            )
            return block, block_size
        
        # The parent tree node now represents the merged block
        parent_block = parent_tree_node
        parent_size = block_size * 2
        
        # Reset parent tree node state
        parent_tree_node.left_child = None
        parent_tree_node.right_child = None
        parent_tree_node.is_split = False
        
        logger.debug(
            f"Merged blocks {block.block_id} and {buddy_block.block_id} "
            f"into block {parent_block.block_id} (size {parent_size})"
        )
        
        # Recursively try to merge parent
        return self._try_merge(parent_block, parent_size)
    
    def can_split_block(self, block_id: int, num_tokens_used: int) -> tuple[bool, int]:
        """
        Check if a block can be split based on usage.
        
        Args:
            block_id: ID of the block to check
            num_tokens_used: Number of tokens currently used in the block
            
        Returns:
            Tuple of (can_split, suggested_new_size)
        """
        if block_id not in self.allocated_blocks:
            return False, 0
        
        _, current_size = self.allocated_blocks[block_id]
        
        # Find the smallest size that can accommodate the used tokens
        new_size = self._find_suitable_size(num_tokens_used)
        
        if new_size is None:
            return False, 0
        
        # Can split if new_size < current_size
        if new_size < current_size:
            return True, new_size
        
        return False, 0
    
    def get_num_free_blocks(self, size: Optional[int] = None) -> int:
        """
        Get the number of free blocks.
        
        Args:
            size: If specified, return free blocks of this size only.
                  If None, return total free blocks across all sizes.
            
        Returns:
            Number of free blocks
        """
        if size is not None:
            return self.slabs[size].num_free_blocks
        
        # Return total free blocks across all queues
        return sum(queue.num_free_blocks for queue in self.slabs.values())
    
    def get_usage(self) -> float:
        """
        Get the KV cache usage.
        
        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        total_gpu_blocks = self.num_gpu_blocks
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
    
    def reset_prefix_cache(self):
        raise NotImplementedError
