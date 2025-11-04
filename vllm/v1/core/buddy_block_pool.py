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
    FreeKVCacheBlockQueue, KVCacheBlock, BuddyTreeNode
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
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
        supported_sizes: list[int] = None,
    ):
        """
        Initialize the Buddy Block Allocator.
        
        Args:
            num_gpu_blocks: Total number of blocks at max_block_size
            supported_sizes: List of supported block sizes
        """
        if supported_sizes is None:
            supported_sizes = [16, 32, 64, 128]
        else:
            self._check_supported_sizes()

        self.supported_sizes = sorted(supported_sizes)
        self.max_block_size = max(self.supported_sizes)
        
        # Initialize free block queues (slabs) for each supported size
        # Each queue is a FreeKVCacheBlockQueue that manages blocks of that size
        self.slabs: dict[int, FreeKVCacheBlockQueue] = {
            size: FreeKVCacheBlockQueue([]) for size in self.supported_sizes
        }
        
        # Buddy tree: track split/merge relationships
        # Maps block_id -> BuddyTreeNode
        self.buddy_tree: dict[int, BuddyTreeNode] = {}
        
        # Track which KVCacheBlock corresponds to which tree node
        # Maps block_id -> KVCacheBlock
        self.blocks: dict[int, KVCacheBlock] = {}
        
        # Track allocated blocks: block_id -> (KVCacheBlock, size)
        self.allocated_blocks: dict[int, tuple[KVCacheBlock, int]] = {}
        
        # Initialize all blocks at maximum size
        self._initialize_block_pool(num_gpu_blocks)
        
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

    def _initialize_block_pool(self, num_blocks: int) -> None:
        """
        Initialize the slab with blocks of maximum size.

        Other slabs are empty.
        
        Args:
            num_blocks: Number of blocks to create at max size
        """
        max_queue = self.slabs[self.max_block_size]

        for i in range(num_blocks):
            kv_block = KVCacheBlock(block_id=i)
            
            # Create buddy tree node for tracking split/merge
            tree_node = BuddyTreeNode(block=kv_block)
            self.buddy_tree[i] = tree_node
            
            max_queue.append(kv_block)
    
    def allocate(self, request_id: str, size: int, num_tokens_in_use: int = 0) -> Optional[KVCacheBlock]:
        """
        Allocate a block of the requested size or larger.
        
        Args:
            request_id: ID of the request
            size: Requested block size in tokens
            num_tokens_in_use: Number of tokens currently in use (for splitting decisions)
            
        Returns:
            KVCacheBlock if allocation succeeds, None otherwise
        """
        # Find the smallest suitable slab
        suitable_size = self._find_suitable_size(size)
        
        if suitable_size is None:
            logger.warning(f"No suitable block size for {size} tokens")
            return None
        
        # Try to allocate from the suitable slab
        block = self._allocate_from_slab(suitable_size)
        
        if block:
            self.allocated_blocks[block.block_id] = (block, suitable_size)
            logger.debug(
                f"Allocated block {block.block_id} of size {suitable_size} "
                f"for request {request_id}"
            )
        
        return block
    
    def touch(self, block: KVCacheBlock, size: int) -> None:
        """
        Touch a block to indicate it's being reused (prefix cache hit).
        
        Similar to BlockPool.touch(), this increases the reference count
        and removes the block from the free list if it was there.
        This prevents cached blocks from being evicted when they are
        hit by new requests.
        
        Args:
            block: The KVCacheBlock to touch
            size: The size category this block belongs to
        """
        # If ref_cnt is 0, the block is in the free list
        if block.ref_cnt == 0:
            # Find which queue this block belongs to
            queue = self.free_block_queues.get(size)
            if queue:
                try:
                    queue.remove(block)  # O(1) removal from middle
                except RuntimeError:
                    # Block was not in free list, that's okay
                    pass
        
        # Increase reference count
        block.ref_cnt += 1
    
    def _find_suitable_size(self, requested_size: int) -> Optional[int]:
        """Find the smallest block size that can accommodate the request."""
        for size in self.supported_sizes:
            if size >= requested_size:
                return size
        return None
    
    def _allocate_from_slab(self, size: int) -> Optional[KVCacheBlock]:
        """
        Allocate a block from the specified queue (slab), splitting if necessary.
        
        Args:
            size: Requested block size
            
        Returns:
            KVCacheBlock if successful, None otherwise
        """
        queue = self.free_block_queues[size]
        
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
        
        # Create two new KVCacheBlock objects for the buddies
        # Use a scheme to derive unique block IDs for children
        left_block_id = parent_block.block_id * 2
        right_block_id = parent_block.block_id * 2 + 1
        
        left_block = KVCacheBlock(block_id=left_block_id, ref_cnt=0)
        right_block = KVCacheBlock(block_id=right_block_id, ref_cnt=0)
        
        self.blocks[left_block_id] = left_block
        self.blocks[right_block_id] = right_block
        
        # Create buddy tree nodes for tracking
        left_tree_node = BuddyTreeNode(
            block=left_block,
            size=target_size,
            parent=parent_tree_node,
        )
        right_tree_node = BuddyTreeNode(
            block=right_block,
            size=target_size,
            parent=parent_tree_node,
        )
        
        self.buddy_tree[left_block_id] = left_tree_node
        self.buddy_tree[right_block_id] = right_tree_node
        
        # Update parent node in the buddy tree
        parent_tree_node.left_child = left_tree_node
        parent_tree_node.right_child = right_tree_node
        parent_tree_node.is_split = True
        # Note: parent_tree_node.is_allocated stays False
        # The parent tree node no longer represents a usable block
        
        # Add right child to target queue's free list
        # (we'll return left child as allocated)
        target_queue = self.free_block_queues[target_size]
        target_queue.append(right_block)
        
        logger.debug(
            f"Split block {parent_block.block_id} (size {parent_size}) "
            f"into blocks {left_block_id} and {right_block_id} "
            f"(size {target_size})"
        )
        
        # Return left child (caller will add to allocated_blocks)
        return left_block
    
    def free(self, block_id: int) -> None:
        """
        Free a block and attempt to merge with buddy.
        
        Args:
            block_id: ID of the block to free
        """
        if block_id not in self.allocated_blocks:
            logger.warning(f"Attempted to free non-allocated block {block_id}")
            return
        
        block, size = self.allocated_blocks.pop(block_id)
        
        # Decrease reference count
        block.ref_cnt -= 1
        
        # Only add to free list if ref_cnt reaches 0
        if block.ref_cnt > 0:
            return
        
        # Try to merge with buddy (updates buddy tree)
        merged_block, merged_size = self._try_merge(block, size)
        
        # Add the final block to appropriate queue
        queue = self.free_block_queues[merged_size]
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
        
        buddy_block = buddy_tree_node.block
        
        # Remove buddy from its queue's free list
        buddy_queue = self.free_block_queues[block_size]
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
        parent_block = parent_tree_node.block
        parent_size = parent_tree_node.size
        
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
        
        # Can't split if already at minimum size
        smaller_sizes = [s for s in self.supported_sizes if s < current_size]
        
        if not smaller_sizes:
            return False, 0
        
        # Find the smallest size that can accommodate the used tokens
        for size in smaller_sizes:
            if size >= num_tokens_used:
                return True, size
        
        return False, 0
    
    def get_num_free_blocks(self, size: Optional[int] = None) -> int:
        """
        Get the number of free blocks.
        
        Args:
            size: If specified, return free blocks of this size only
            
        Returns:
            Number of free blocks
        """
        if size is not None:
            return self.free_block_queues[size].num_free_blocks
        
        # Return total free blocks across all queues
        return sum(queue.num_free_blocks for queue in self.free_block_queues.values())
