"""
Buddy Memory Allocator for KV Cache with Variable Block Sizes.

This module implements a multi-slab memory management system where blocks can be
split and merged dynamically. It supports multiple block sizes (e.g., 16, 32, 64, 128 tokens)
organized in slabs, with a buddy tree for tracking split relationships.

Each slab is simply a FreeKVCacheBlockQueue that manages free blocks of a specific size.
This directly reuses the existing doubly linked list implementation with O(1) middle
removal support, which is essential for prefix caching.
"""
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
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events

        # Map coordinate (block_id, size, relative_id) to KVCacheBlock
        self._blocks: dict[tuple[int, int, int], BuddyTreeBlock] = {}
        # Initialize free block queues (slabs) for each supported size
        # Each queue is a FreeKVCacheBlockQueue that manages blocks of that size
        self.slabs = self._initialize_block_pool(num_gpu_blocks)
        
        # Track allocated blocks: block_id -> (KVCacheBlock, size)
        self.allocated_blocks: dict[int, tuple[KVCacheBlock, int]] = {}
        
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
            block = slab.popleft()
            return block
        else:
            return None
    
    def _reclaim_from_allocated_blocks(self) -> Optional[BuddyTreeBlock]:
        """
        Try to reclaim unused space from allocated blocks.
        
        Find an allocated block with unused capacity, split it to fit actual usage,
        and return the freed portions to their respective slabs.
        
        Returns:
            The resized block if successful, None otherwise
        """
        # Iterate through allocated blocks to find one with unused space
        for block_id, (block, current_size) in list(self.allocated_blocks.items()):
            # Skip if block has multiple references (can't safely resize)
            if block.ref_cnt > 1:
                continue
            
            # Get actual token usage from tracking
            num_tokens_used = self.block_usage.get(block_id, 0)
            
            # If no usage info, assume worst case (can't reclaim)
            if num_tokens_used == 0:
                continue
            
            # Check if we can split this block
            can_split, new_size = self.can_split_block(block_id, num_tokens_used)
            
            if can_split:
                logger.debug(
                    f"Reclaiming space from block {block_id}: "
                    f"current_size={current_size}, used={num_tokens_used}, "
                    f"new_size={new_size}"
                )
                
                # Split the block and return freed portions to slabs
                self._split_allocated_block(block, current_size, new_size)
                return block
        
        return None
    
    def _split_allocated_block(
        self, 
        block: BuddyTreeBlock, 
        current_size: int, 
        new_size: int
    ) -> None:
        """
        Split an allocated block to reduce its size, returning freed space to slabs.
        
        For example: 128-token block with 32 tokens used
        -> Split into: 1x 32-token (keep allocated) + 1x 32-token (free) + 1x 64-token (free)
        
        Args:
            block: The block to split
            current_size: Current size of the block
            new_size: New size to fit actual usage
        """
        remaining_size = current_size
        kept_block = block
        
        # Repeatedly split until we reach new_size
        while remaining_size > new_size:
            half_size = remaining_size // 2
            
            # Create buddy blocks at half_size
            left_child, right_child = self._create_buddy_pair(
                kept_block, remaining_size, half_size
            )
            
            if half_size >= new_size:
                # Keep splitting the left child
                kept_block = left_child
                # Free the right child
                self.slabs[half_size].append(right_child)
                logger.debug(f"Freed right child of size {half_size} to slab")
            else:
                # We've split too far, this shouldn't happen with proper logic
                logger.error(
                    f"Split logic error: half_size={half_size} < new_size={new_size}"
                )
                break
            
            remaining_size = half_size
        
        # Update the allocated block entry
        self.allocated_blocks[block.block_id] = (kept_block, new_size)
        logger.debug(
            f"Resized block {block.block_id} from {current_size} to {new_size}"
        )
    
    def _create_buddy_pair(
        self,
        parent_block: BuddyTreeBlock,
        parent_size: int,
        child_size: int
    ) -> tuple[BuddyTreeBlock, BuddyTreeBlock]:
        """
        Create a pair of buddy blocks by splitting a parent.
        
        Args:
            parent_block: The parent block to split
            parent_size: Size of the parent block
            child_size: Size of each child block
            
        Returns:
            Tuple of (left_child, right_child)
        """
        assert parent_size == child_size * 2, "Invalid split ratio"
        
        # Create left and right children
        left_child = BuddyTreeBlock(
            block_id=parent_block.block_id,
            relative_id=parent_block.relative_id * 2,
            size=child_size,
            is_virtual=False
        )
        
        right_child = BuddyTreeBlock(
            block_id=parent_block.block_id,
            relative_id=parent_block.relative_id * 2 + 1,
            size=child_size,
            is_virtual=False
        )
        
        # Establish parent-child relationships
        left_child.parent = parent_block
        right_child.parent = parent_block
        parent_block.left_child = left_child
        parent_block.right_child = right_child
        parent_block.is_split = True
        
        # Store in blocks map
        self._blocks[(left_child.block_id, child_size, left_child.relative_id)] = left_child
        self._blocks[(right_child.block_id, child_size, right_child.relative_id)] = right_child
        
        return left_child, right_child
    
    def _find_suitable_size(self, requested_size: int) -> Optional[int]:
        """
        Find the smallest block size that can accommodate the requested size.
        
        Since supported_sizes is in descending order, we need to find from small to large.
        
        Args:
            requested_size: Number of tokens to accommodate
            
        Returns:
            The smallest block size >= requested_size, or None if none exists
        """
        # supported_sizes is sorted in descending order [128, 64, 32, 16]
        # Find the smallest size that is >= requested_size
        suitable = None
        for size in reversed(self.supported_sizes):  # Iterate from small to large
            if size >= requested_size:
                suitable = size
            else:
                break  # No smaller size will work
        return suitable
    
    def _allocate_from_slab(self, size: int) -> Optional[KVCacheBlock]:
        """
        Allocate a block from the specified queue (slab), splitting if necessary.
        
        Args:
            size: Requested block size
            
        Returns:
            KVCacheBlock if successful, None otherwise
        """
        queue = self.slabs[size]
        
        # Try to get a free block from the queue
        if queue.num_free_blocks > 0:
            return queue.popleft()
        
        # No free blocks, try to split a larger block
        return self._split_larger_block(size)
    
    def _split_larger_block(self, target_size: int) -> Optional[KVCacheBlock]:
        """
        Split a larger block to create a block of target_size.
        
        Args:
            target_size: Desired block size after splitting
            
        Returns:
            KVCacheBlock of target_size, or None if splitting fails
        """
        # Find the next larger size
        larger_size = None
        for size in self.supported_sizes:
            if size > target_size:
                larger_size = size
                break
        
        if larger_size is None:
            logger.warning(f"Cannot split: no larger block available for size {target_size}")
            return None
        
        # Recursively get a block of larger size
        larger_block = self._allocate_from_slab(larger_size)
        
        if larger_block is None:
            return None
        
        # Split the larger block
        return self._split_block(larger_block, larger_size, target_size)
    
    def _split_block(
        self, 
        parent_block: KVCacheBlock, 
        parent_size: int,
        target_size: int
    ) -> KVCacheBlock:
        """
        Split a block into two buddies.
        
        This updates the buddy tree (for tracking merge opportunities) and
        adds the new blocks to the appropriate slab.
        
        Args:
            parent_block: The block to split
            parent_size: Current size of the parent block
            target_size: Size of the blocks after splitting
            
        Returns:
            One of the buddy blocks (left child)
        """
        assert parent_size == target_size * 2, "Can only split into half"
        
        # Get the parent tree node
        parent_tree_node = self.buddy_tree[parent_block.block_id]
        
        # Create two new BuddyTreeBlock objects for the buddies
        # Use a scheme to derive unique block IDs for children
        left_block_id = parent_block.block_id * 2
        right_block_id = parent_block.block_id * 2 + 1
        
        left_block = BuddyTreeBlock(block_id=left_block_id, ref_cnt=0)
        right_block = BuddyTreeBlock(block_id=right_block_id, ref_cnt=0)
        
        self._blocks[left_block_id] = left_block
        self._blocks[right_block_id] = right_block
        
        # Create buddy tree nodes for tracking
        left_block.parent = parent_tree_node
        right_block.parent = parent_tree_node
        
        # Update parent node in the buddy tree
        parent_tree_node.left_child = left_block
        parent_tree_node.right_child = right_block
        parent_tree_node.is_split = True
        # Note: parent_tree_node.is_allocated stays False
        # The parent tree node no longer represents a usable block
        
        # Store in buddy tree
        self.buddy_tree[left_block_id] = left_block
        self.buddy_tree[right_block_id] = right_block
        
        # Add right child to target queue's free list
        # (we'll return left child as allocated)
        target_queue = self.slabs[target_size]
        target_queue.append(right_block)
        
        logger.debug(
            f"Split block {parent_block.block_id} (size {parent_size}) "
            f"into blocks {left_block_id} and {right_block_id} "
            f"(size {target_size})"
        )
        
        # Return left child (caller will add to allocated_blocks)
        return left_block
    
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
