from collections import defaultdict
from typing import Any, Optional, Sequence, overload
from dataclasses import dataclass, field
from enum import Enum

from vllm.logger import init_logger
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    SegmentHash,
    SegmentHashWithGroupId, 
    BuddyTreeBlock, 
    SemanticSegment, 
    generate_block_hash_extra_keys,
    get_block_hash,
    hash_block_tokens,
    make_segment_hash_with_group_id,
    replace_block_in_segment,
    swap_blocks,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class SegmentIdGenerator:
    """
    Singleton generator for creating unique segment IDs.
    """
    _instance = None
    _counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(self) -> int:
        """Generate and return the next unique segment ID."""
        self._counter += 1
        return self._counter

    def reset(self) -> None:
        """Reset the counter."""
        self._counter = 0


class SegmentHashToSegmentMap:
    """
    Cache of segments that are used for prefix caching. It caches segments
    from hash directly to a segment or multiple segments.
    
    Similar to BlockHashToBlockMap, but for SemanticSegments.
    """

    def __init__(self):
        self._cache: dict[SegmentHashWithGroupId, SemanticSegment | dict[int, SemanticSegment]] = {}

    def get_one_segment(self, key: SegmentHashWithGroupId) -> Optional[SemanticSegment]:
        """
        Gets any segment with the given segment hash key.
        """
        segments = self._cache.get(key)
        if segments is not None:
            if isinstance(segments, SemanticSegment):
                return segments
            if isinstance(segments, dict):
                return next(iter(segments.values()))
            self._unexpected_segments_type(segments)
        return None

    def insert(self, key: SegmentHashWithGroupId, 
               segment: SemanticSegment) -> None:
        """
        Inserts the SemanticSegment to the cache
        """
        segments = self._cache.get(key)
        if segments is None:
            self._cache[key] = segment
        elif isinstance(segments, SemanticSegment):
            self._cache[key] = {segments.segment_id: segments, 
                                segment.segment_id: segment}
        elif isinstance(segments, dict):
            segments[segment.segment_id] = segment
        else:
            self._unexpected_segments_type(segments)

    def pop(
        self, key: SegmentHashWithGroupId, segment_id: int
    ) -> Optional[SemanticSegment]:
        """
        Checks if segment_hash exists and pop segment from the cache
        """
        segments = self._cache.pop(key, None)
        if segments is None:
            return None
            
        if isinstance(segments, SemanticSegment):
            if segments.segment_id == segment_id:
                return segments
            # If the single segment ID doesn't match, put it back
            self._cache[key] = segments
            return None
            
        if isinstance(segments, dict):
            # Try to pop segment_id from the dict, and if dict still
            # contains segments, put back to the cache
            segment = segments.pop(segment_id, None)
            if len(segments) > 0:
                self._cache[key] = segments
            return segment
            
        self._unexpected_segments_type(segments)
        return None

    def contains(
        self,
        key: SegmentHashWithGroupId,
        segment_id: int,
    ) -> bool:
        segments = self._cache.get(key)
        if segments is None:
            return False
        if isinstance(segments, SemanticSegment):
            return segments.segment_id == segment_id
        if isinstance(segments, dict):
            return segment_id in segments
        self._unexpected_segments_type(segments)
        return False
    
    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_segments_type(self, segments: Any) -> None:
        raise AssertionError(f"Invalid KV cache segment type {type(segments)}")
    
    
@dataclass
class SemanticSegments:
    """
    Container for managing multiple semantic segments.
    """
    segments: list[SemanticSegment] = field(default_factory=list)
    
    @property
    def capacity(self) -> int:
        """Get the total capacity of all segments."""
        return sum(segment.capacity for segment in self.segments)
    
    def append(self, segment: SemanticSegment) -> None:
        """Add a segment to the collection."""
        self.segments.append(segment)
        
    def pop(self) -> SemanticSegment:
        """Remove and return the last segment."""
        return self.segments.pop()

    def __iadd__(
        self, other: "SemanticSegments | Sequence[SemanticSegment]"
    ) -> "SemanticSegments":
        """Extend the collection with multiple segments."""
        if self.unsealed_segment is not None:
             raise ValueError("Cannot extend segments when the last segment is unsealed.")

        if isinstance(other, SemanticSegments):
            self.segments.extend(other.segments)
        else:
            self.segments.extend(other)
        return self
    
    @property
    def last_segment(self) -> Optional[SemanticSegment]:
        """Get the last segment in the collection."""
        return self.segments[-1] if self.segments else None
    
    @last_segment.setter
    def last_segment(self, segment: SemanticSegment) -> None:
        """Set/replace the last segment in the collection.
        If the collection is empty, append the given segment.
        """
        if not isinstance(segment, SemanticSegment):
            raise TypeError("last_segment must be a SemanticSegment")
        if self.segments:
            self.segments[-1] = segment
        else:
            self.segments.append(segment)
    
    @property
    def unsealed_segment(self) -> Optional[SemanticSegment]:
        """Get the last unsealed segment, if any."""
        if self.segments and not self.segments[-1].is_sealed:
            return self.segments[-1]
        return None
    
    def __len__(self) -> int:
        return len(self.segments)
    
    @overload
    def __getitem__(self, key: int) -> SemanticSegment: ...
    @overload
    def __getitem__(self, key: slice) -> "SemanticSegments": ...

    def __getitem__(self, key: slice | int) -> "SemanticSegment | SemanticSegments":
        """
        Support indexing and slicing: 
        - segment = semantic_segments[0]
        - segments = semantic_segments[1:3]
        """
        if isinstance(key, slice):
            return SemanticSegments(self.segments[key])
        else:
            return self.segments[key]

    def __iter__(self):
        """Support iteration: for segment in semantic_segments:"""
        return iter(self.segments)

    def __reversed__(self):
        """Support reversed(): reversed(semantic_segments)"""
        return reversed(self.segments)

    def __contains__(self, segment: SemanticSegment) -> bool:
        """Support in operator: if segment in semantic_segments:"""
        return segment in self.segments
    
    def __add__(self, other: "SemanticSegments") -> "SemanticSegments":
        """Concatenate two SemanticSegments objects."""
        if not isinstance(other, SemanticSegments):
            raise TypeError("Can only add SemanticSegments to SemanticSegments")
        return SemanticSegments(self.segments + other.segments)
    
    
class EvictionPolicy(Enum):
    TIGHT = 0
    OVERPROVISION = 1


@dataclass
class ActiveBlockCursor:
    """
    Cursor to track the active block being filled in a segment.
    """
    segment_id: int
    block: BuddyTreeBlock
    block_offset_in_segment: int
    # The total number of tokens computed in previous segments.
    # This helps avoid iterating over previous segments to calculate offset.
    computed_tokens_base: int

# TODO(huanyu): record KV events
class SemanticSegmentManager:
    """
    Manages semantic segments for KV cache.
    
    This class handles:
    1. Accumulating blocks for active requests until a segment boundary is reached.
    2. promoting active blocks to a SemanticSegment.
    3. Managing prefix caching for segments.
    4. Managing reference counts for segments.
    """

    def __init__(
        self,
        block_pool: BuddyBlockPool,
        kv_cache_group_id: int,
        eviction_policy: EvictionPolicy = EvictionPolicy.TIGHT,
    ):
        self.block_pool = block_pool
        self.eviction_policy = eviction_policy
        
        # request_id -> semantic segments used by this request
        # The last segment might be unsealed (is_sealed=False)
        self.req_to_segments: dict[str, SemanticSegments] = defaultdict(SemanticSegments)
        
        # request_id -> active block cursor for O(1) update
        self.req_cursors: dict[str, ActiveBlockCursor] = {}
        
        # Prefix cache: hash -> SemanticSegment
        self.cached_segments = SegmentHashToSegmentMap()
        
        # This queue records the segments with ref_cnt = 0
        # All segments in this queue must in self.cached_segments
        self.free_segment_queue = FreeKVCacheBlockQueue([])
        self.num_free_segment_tokens = 0

        self.kv_cache_group_id = kv_cache_group_id
        self.num_cached_segments: dict[str, int] = {}
        self.completed_req_to_segments: dict[str, SemanticSegments] = {}
        
        self.pending_moves: list[tuple[int, int, int, int]] = [] # list of (group_id, src_addr, dst_addr, size)
        self.pending_swaps: list[tuple[int, int, int, int]] = [] # list of (group_id, addr1, addr2, size)
        self.blocks_to_free_later: list[BuddyTreeBlock] = [] # blocks to be freed after current moves are executed
        self.blocks_being_moved: list[BuddyTreeBlock] = [] # blocks currently being moved by the worker

    def get_pending_moves(self) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        """Get and clear pending moves."""
        moves = self.pending_moves
        swaps = self.pending_swaps
        self.pending_moves = []
        self.pending_swaps = []
        
        # Free blocks that were pending free from PREVIOUS step
        # These blocks have been moved by the worker in the previous step,
        # so it is safe to free them now.
        if self.blocks_being_moved:
            self.block_pool.free_blocks(self.blocks_being_moved)
            self.blocks_being_moved = []
            
        # Move current blocks to be freed to the next step
        self.blocks_being_moved = self.blocks_to_free_later
        self.blocks_to_free_later = []
            
        return moves, swaps

    def consolidate_segment_memory(self, request_id: str) -> None:
        """Consolidate the memory of the sealed segments for the request."""
        active_segments = self.req_to_segments.get(request_id)
        if active_segments is not None:
            segments = list(active_segments)
        else:
            segments = self._consolidatable_segments(
                self.completed_req_to_segments.get(
                    request_id,
                    SemanticSegments(),
                )
            )
        
        for segment in segments:
            if segment.is_sealed and segment.capacity > 0 and not segment.is_consolidated:
                result = SemanticSegmentManager._consolidate_segment_memory(
                    segment, self.block_pool)
                if result:
                    moves, swaps = result

                    # Order moves so we never overwrite a later move's source.
                    # This enables single-round consolidation without relying
                    # on multi-step async frees.
                    # if moves:
                    #     moves = SemanticSegmentManager._order_moves_safely(
                    #         moves, self.block_pool)

                    # Some destinations may be previous move sources (i.e., an
                    # in-flight source location reused as a destination). Only
                    # free sources that are not reused as destinations.
                    dst_blocks = {dst for _, dst in moves}
                    
                    # Process moves
                    for src_block, dst_block in moves:
                        src_addr = self.block_pool.calculate_address(src_block)
                        dst_addr = self.block_pool.calculate_address(dst_block)
                        self.pending_moves.append((self.kv_cache_group_id, src_addr, dst_addr, src_block.size))
                        if src_block not in dst_blocks:
                            self.blocks_to_free_later.append(src_block)
                        
                    # Process swaps
                    for block1, block2 in swaps:
                        addr1 = self.block_pool.calculate_address(block1)
                        addr2 = self.block_pool.calculate_address(block2)
                        self.pending_swaps.append((self.kv_cache_group_id, addr1, addr2, block1.size))

                # Single-round planner: if we produced any ops, we treat the
                # segment as consolidated for subsequent attention metadata.
                segment.is_consolidated = True

    def _consolidatable_segments(
        self,
        segments: SemanticSegments,
    ) -> list[SemanticSegment]:
        return [
            segment for segment in segments
            if segment.ref_cnt == 0
            and segment.is_sealed
            and segment.capacity > 0
            and not segment.is_consolidated
            and self._is_segment_cached(segment)
        ]

    def _is_segment_cached(self, segment: SemanticSegment) -> bool:
        if segment.segment_hash is None:
            return False
        return self.cached_segments.contains(
            segment.segment_hash,
            segment.segment_id,
        )

    @staticmethod
    def _order_moves_safely(
        moves: list[tuple[BuddyTreeBlock, BuddyTreeBlock]],
        block_pool: BuddyBlockPool,
    ) -> list[tuple[BuddyTreeBlock, BuddyTreeBlock]]:
        """Order moves to avoid overwriting any move's source before it is read.

        Each move is a copy from src_addr->dst_addr for a contiguous token range.
        If move A writes into a range that overlaps move B's source range, then
        B must be executed before A.

        This returns a topologically-sorted order. If a cycle is detected, 
        we raise a ValueError as it should not happen.
        """

        def interval(addr: int, size: int) -> tuple[int, int]:
            return addr, addr + size

        src_addrs: list[int] = [block_pool.calculate_address(s) for s, _ in moves]
        dst_addrs: list[int] = [block_pool.calculate_address(d) for _, d in moves]
        sizes: list[int] = [s.size for s, _ in moves]

        src_intervals = [interval(a, sz) for a, sz in zip(src_addrs, sizes)]
        dst_intervals = [interval(a, sz) for a, sz in zip(dst_addrs, sizes)]

        n = len(moves)
        adj: list[list[int]] = [[] for _ in range(n)]
        indeg = [0] * n

        def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return a[0] < b[1] and b[0] < a[1]

        # Edge j->i if i's dst overlaps j's src (j must run before i).
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if overlaps(dst_intervals[i], src_intervals[j]):
                    adj[j].append(i)
                    indeg[i] += 1

        queue = [i for i in range(n) if indeg[i] == 0]
        ordered: list[int] = []
        while queue:
            idx = queue.pop()
            ordered.append(idx)
            for nxt in adj[idx]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        if len(ordered) != n:
            raise ValueError(
                "Cycle detected in memory moves, cannot order safely.")
        return [moves[i] for i in ordered]

    def update_block_usage(
        self, request_id: str, 
        num_computed_tokens: int
    ) -> None:
        """
        Update the block usage for the request.
        
        Args:
            request_id: The request ID.
            num_computed_tokens: The total number of tokens computed for the request.
        """
        segments = self.req_to_segments[request_id]
        unsealed_segment = segments.unsealed_segment
        if not unsealed_segment:
            return

        # Find the active block
        cursor = self.req_cursors.get(request_id)
        
        # Check if cursor is valid for current unsealed segment
        if cursor is not None and cursor.segment_id == unsealed_segment.segment_id:
             current_offset = cursor.computed_tokens_base
             current_block = cursor.block
             current_block_offset = cursor.block_offset_in_segment
        else:
             # Initialize cursor or re-calculate base offset if segment changed
             current_offset = 0
             for segment in segments:
                 if segment is unsealed_segment:
                     break
                 current_offset += segment.capacity
             
             current_block = unsealed_segment.head
             current_block_offset = 0

        tokens_remaining = num_computed_tokens - current_offset
        if tokens_remaining < 0:
            # Should not happen
            return
            
        # Traverse forward to find the block containing tokens_remaining
        while current_block:
            block_end = current_block_offset + current_block.size
            
            # Check if this block is the one being filled or just filled
            if tokens_remaining <= block_end:
                usage = max(0, tokens_remaining - current_block_offset)
                
                if current_block.num_tokens != usage:
                    self.block_pool.update_block_usage(
                        current_block.block_id, current_block.size, 
                        current_block.relative_id, usage
                    )
                
                # Update cursor
                self.req_cursors[request_id] = ActiveBlockCursor(
                    unsealed_segment.segment_id, 
                    current_block, 
                    current_block_offset,
                    current_offset
                )
                return
            
            # This block is fully used, mark it as full if needed
            if current_block.num_tokens != current_block.size:
                self.block_pool.update_block_usage(
                    current_block.block_id, current_block.size, 
                    current_block.relative_id, current_block.size
                )
            
            current_block_offset += current_block.size
            current_block = current_block.next_block

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[BuddyTreeBlock]:
        """
        Allocate blocks for a request.
        
        These blocks are added to the last segment of the request.
        If the last segment is sealed or doesn't exist, a new unsealed segment is created.
        """
        # 1. Try allocate from slabs
        blocks, remaining = self.block_pool.get_new_blocks(num_tokens)
            
        if remaining > 0:
            # 2. Free segments if needed
            freed_capacity = 0
            original_remaining = remaining
            if self.eviction_policy == EvictionPolicy.OVERPROVISION:
                remaining = (((remaining + self.block_pool.max_block_size - 1) // 
                             self.block_pool.max_block_size) * 
                             self.block_pool.max_block_size)          
            while (remaining > freed_capacity and 
                self.free_segment_queue.num_free_segments > 0):
                segment: SemanticSegment = self.free_segment_queue.popleft()
                self.num_free_segment_tokens -= segment.capacity
                self._maybe_evict_cached_segment(segment)
                segment.unseal()
                freed_capacity += segment.capacity
                self.block_pool.free_blocks(reversed(segment.blocks))
            
            if freed_capacity > 0:
                new_blocks, remaining = self.block_pool.get_new_blocks(
                    original_remaining)
                if new_blocks:
                    blocks.extend(new_blocks)
            else:
                remaining = original_remaining

            if remaining > 0:
                # 3. Reclaim from allocated blocks if still needed
                reclaimed_blocks = self.block_pool.reclaim_new_blocks(remaining)
                blocks.extend(reclaimed_blocks)
        
        segments = self.req_to_segments[request_id]
        if not segments.unsealed_segment:
            # Create new unsealed segment
            segment_id = SegmentIdGenerator().generate()
            new_segment = SemanticSegment(segment_id=segment_id, ref_cnt=1)
            segments.append(new_segment)
        
        # Append to unsealed segment
        segments.unsealed_segment.append(blocks)
            
        return blocks
    
    def _maybe_evict_cached_segment(self, segment: SemanticSegment) -> bool:
        """
        Evict a segment from the cache.

        Returns True if evicted, False if not found or still in use (ref_cnt > 0).
        """
        segment_hash = segment.segment_hash
        if segment_hash is None:
            # No hash provided, eviction not applicable
            return False

        if self.cached_segments.pop(segment_hash, segment.segment_id) is None:
            # segment not found in cached_segments,
            # eviction is not needed
            return False
        
        self._remove_completed_segment(segment)
        segment.reset_hash()

        return True

    def _remove_completed_segment(self, segment: SemanticSegment) -> None:
        for request_id, segments in list(self.completed_req_to_segments.items()):
            kept = [
                completed_segment for completed_segment in segments
                if completed_segment.segment_id != segment.segment_id
            ]
            if len(kept) == len(segments):
                continue
            if kept:
                self.completed_req_to_segments[request_id] = SemanticSegments(kept)
            else:
                del self.completed_req_to_segments[request_id]

    def _compute_partial_tail_hash(self, request: Request) -> Optional[BlockHash]:
        """Compute hash for a partial tail block of a request.

        Returns None if there is no partial tail to hash.
        """
        hasher = request.get_hash_new_full_blocks
        if hasher is not None and hasattr(hasher, "func"):
            hasher_func = hasher.func
            block_size = getattr(hasher_func, "block_size", self.block_pool.min_block_size)
            caching_hash_fn = getattr(hasher_func, "caching_hash_fn", None)
        else:
            block_size = self.block_pool.min_block_size
            caching_hash_fn = None

        if caching_hash_fn is None:
            return None

        start_token_idx = len(request.block_hashes) * block_size
        if start_token_idx >= request.num_tokens:
            # No partial tail.
            return None

        end_token_idx = request.num_tokens
        curr_mm_idx = -1 if start_token_idx > 0 else 0
        extra_keys, _ = generate_block_hash_extra_keys(
            request, start_token_idx, end_token_idx, curr_mm_idx
        )

        parent_block_hash = request.block_hashes[-1] if request.block_hashes else None
        block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
        return hash_block_tokens(
            caching_hash_fn,
            parent_block_hash,
            block_tokens,
            extra_keys,
        )

    def seal_segment(
        self, request: Request | str
    ) -> None:
        """
        Seal the unsealed segment for a request.
        
        This method seals the entire unsealed segment, computes its hash based on the last block,
        and inserts it into the cache for potential reuse.
        
        Note: blocks within a segment cannot be shared by other requests before sealing.
        
        Args:
            request: The request or request ID whose unsealed segment is to be sealed.
            
        Raises:
            ValueError: If there are no blocks in the unsealed segment.

        Notes:
            If the tail block has no block hash (e.g. request teardown before
            cache hashes are populated), the segment is still sealed but will
            not be inserted into prefix cache.
        """
        if isinstance(request, Request):
            request_obj: Optional[Request] = request
            request_id = request.request_id
        else:
            request_obj = None
            request_id = request

        segments = self.req_to_segments[request_id]
        unsealed_segment = segments.unsealed_segment
        if not unsealed_segment:
            logger.debug(f"Request {request_id} has no unsealed segment.")
            return

        if unsealed_segment.head is None or unsealed_segment.tail is None:
            raise ValueError(
                f"Cannot seal segment for request {request_id}: "
                "no blocks in unsealed segment."
            )

        # We can directly use the tail block's hash as the segment hash because
        # cache_full_blocks is always executed before sealing the segment,
        # ensuring all blocks have a block_hash.
        last_block_hash_with_group_id = unsealed_segment.tail.block_hash
        if last_block_hash_with_group_id is None:
            # If the tail block is partial and request context is available,
            # force a tail hash computation so the sealed segment still has a
            # stable prefix-cache key.
            partial_tail_hash = (
                self._compute_partial_tail_hash(request_obj)
                if request_obj is not None
                else None
            )

            if partial_tail_hash is None:
                # Keep lifecycle invariants even when no stable hash is available.
                unsealed_segment.seal()
                logger.debug(
                    "Seal segment for request %s without block hash; skipping cache insertion.",
                    request_id,
                )
                return

            segment_hash = SegmentHash(partial_tail_hash)
        else:
            # Fast path: a full tail block already has a block hash.
            segment_hash = SegmentHash(get_block_hash(last_block_hash_with_group_id))

        # SegmentHash is group-agnostic; we pack group id only when forming
        # the cache key.
        segment_hash_with_group_id = make_segment_hash_with_group_id(
            segment_hash, self.kv_cache_group_id
        )

        unsealed_segment.seal()

        # If prefix caching is disabled, do not insert segments into the cache.
        if not self.block_pool.enable_caching:
            return

        unsealed_segment.segment_hash = segment_hash_with_group_id
        self.cached_segments.insert(segment_hash_with_group_id, unsealed_segment)
        return
    
    def cache_segments(
        self,
        request: Request,
        num_sealed_segments: int
    ) -> None:
        """Cache full segments for prefix caching.

        This method iterates through the segments of a request. For any sealed segment
        that falls fully within the range of [num_cached_segments, num_sealed_segments], it ensures
        the segment is inserted into the prefix cache.

        Args:
            request: The request to cache the segments.
            num_sealed_segments: The number of segments that are sealed and should be cached after this function.
        """
        num_cached_segments = self.num_cached_segments.get(
            request.request_id, 0)
        if num_cached_segments >= num_sealed_segments:
            return
        segments = self.req_to_segments[request.request_id]
        new_sealed_segments = segments[num_cached_segments:num_sealed_segments]
        assert len(request.segment_hashes) >= num_sealed_segments
        new_segment_hashes = request.segment_hashes[num_cached_segments:]

        for i, segment in enumerate(new_sealed_segments):
            assert segment.is_sealed, (
                f"Segment {num_cached_segments + i} is not sealed."
            )
            assert segment.segment_hash is None
            segment_hash = new_segment_hashes[i]

            segment_hash_with_group_id = make_segment_hash_with_group_id(
                segment_hash, self.kv_cache_group_id
            )
            segment.segment_hash = segment_hash_with_group_id
            self.cached_segments.insert(segment_hash_with_group_id, segment)

        # TODO(huanyu): record KV cache events

    def free(self, request: Request | str) -> None:
        """
        Release resources for a request.

        Steps:
        1) Seal any unsealed segment and free its blocks.
        2) Remove the request's segment list and decrement each segment's ref_cnt.
        3) Enqueue segments with ref_cnt == 0 into free_segment_queue for later reclamation.

        Note: Segments remain in cached_segments for potential reuse (prefix caching);
        actual eviction is handled separately by cache policy.
        """
        # Default to empty SemanticSegments in case request is freed before allocation
        if isinstance(request, Request):
            request_id = request.request_id
            request_obj: Request | str = request
        else:
            request_id = request
            request_obj = request_id

        self.seal_segment(request_obj)
        segments = self.req_to_segments.pop(request_id, SemanticSegments())
        self.req_cursors.pop(request_id, None)
        self.num_cached_segments.pop(request_id, None)
        
        for segment in reversed(segments):
            segment.ref_cnt -= 1
            if segment.ref_cnt == 0:
                # If prefix caching is disabled, return blocks to the pool
                # immediately (behave like a normal allocator).
                if not self.block_pool.enable_caching:
                    segment.unseal()
                    self._maybe_evict_cached_segment(segment)
                    self.block_pool.free_blocks(reversed(segment.blocks))
                else:
                    self.free_segment_queue.append(segment)  # type: ignore
                    self.num_free_segment_tokens += segment.capacity
        completed_segments = self._consolidatable_segments(segments)
        if completed_segments:
            self.completed_req_to_segments[request_id] = SemanticSegments(
                completed_segments
            )
        else:
            self.completed_req_to_segments.pop(request_id, None)
                
    def touch(self, segments: SemanticSegments) -> None:
        """
        Touch segments to increase their reference count and prevent eviction.

        Args:
            segments: A sequence of segments to touch.
        """
        for segment in segments:
            # ref_cnt=0 means this segment is in the free list (i.e. 
            # eviction candidate), so remove it.
            if segment.ref_cnt == 0:
                self.free_segment_queue.remove(segment)  # type: ignore
                self.num_free_segment_tokens -= segment.capacity
                self._remove_completed_segment(segment)
            segment.ref_cnt += 1
                
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.req_to_segments:
            logger.warning(
                "Failed to reset prefix cache because some requests are active."
            )
            return False

        # Free all evictable segments to return blocks to the pool
        while self.free_segment_queue.num_free_segments > 0:
            segment: SemanticSegment = self.free_segment_queue.popleft()  # type: ignore
            self.num_free_segment_tokens -= segment.capacity
            self._maybe_evict_cached_segment(segment)
            segment.unseal()
            self.block_pool.free_blocks(reversed(segment.blocks))

        # Reset the hash map
        self.cached_segments = SegmentHashToSegmentMap()
        self.completed_req_to_segments.clear()

        logger.info("Successfully reset prefix cache")
        
        # if self.enable_kv_cache_events:
        #     self.kv_event_queue.append(AllBlocksCleared())

        return True
                
    def get_cached_segment(
        self, segment_hash: SegmentHash,
        kv_cache_group_ids: list[int]
    ) -> Optional[list[SemanticSegment]]:
        """Get the cached segment by the segment hash for the given group,
        or None if cache miss.

        Args:
            segment_hash: The hash value of the segment.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached segment if exists, or None.
        """
        cached_segments = []
        for group_id in kv_cache_group_ids:
            segment_hash_with_group_id = make_segment_hash_with_group_id(
                segment_hash, group_id
            )
            segment = self.cached_segments.get_one_segment(
                segment_hash_with_group_id
            )
            if not segment:
                return None
            cached_segments.append(segment)
        return cached_segments
    
    def find_longest_cache_hit(
        self,
        segment_hashes: list[SegmentHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        use_eagle: bool
    ) -> tuple[SemanticSegments, ...]:
        matched_segments: tuple[SemanticSegments, ...] = tuple(
            SemanticSegments() for _ in range(len(kv_cache_group_ids))
        )
        
        current_length = 0
        for segment_hash in segment_hashes:
            cached_segments = self.get_cached_segment(segment_hash, kv_cache_group_ids)
            if not cached_segments:
                break
            
            # Assuming all segments in the group have the same capacity
            segment_length = cached_segments[0].capacity
            if current_length + segment_length > max_length:
                break

            # Add segments to matched_segments
            for matched, segment in zip(matched_segments, cached_segments):
                matched.append(segment)
            
            current_length += segment_length
                
        if use_eagle and matched_segments[0]:
            for matched in matched_segments:
                matched.pop()
                
        return matched_segments

    @classmethod
    # TODO(huanyu): consolidating one segment may affect the memory layout of 
    # already consolidated segments
    def _consolidate_segment_memory(
        cls, segment: SemanticSegment, block_pool: BuddyBlockPool
    ) -> Optional[tuple[list[tuple[BuddyTreeBlock, BuddyTreeBlock]],
                        list[tuple[BuddyTreeBlock, BuddyTreeBlock]]]]:
        """
        Consolidate all blocks in a segment into a contiguous region.
        This includes swapping data with other allocated blocks or free blocks
        to ensure physical contiguity starting from the segment head.

        Args:
            segment: The `SemanticSegment` to consolidate.
            block_pool: The `BuddyBlockPool` to allocate new blocks from.

        Returns:
            A tuple of (moves, swaps).
            - moves: list[(src, dst)] where data moves from src to an empty dst.
            - swaps: list[(src, dst)] where data at src and dst are exchanged.
            The caller should perform the actual memory copy/swap.
            Returns None if no consolidation was performed.
        """
        if not isinstance(segment, SemanticSegment):
            raise TypeError("segment must be a SemanticSegment")

        if not segment.is_sealed or segment.capacity <= 0:
            raise ValueError("Can only consolidate sealed and non-empty segments.")

        start_address = block_pool.calculate_address(segment.head)
        
        # Helper to find leaf block covering a physical address
        def find_leaf_block(address: int) -> BuddyTreeBlock:
            max_size = block_pool.max_block_size
            block_id = address // max_size
            offset = address % max_size
            
            for size in block_pool.supported_sizes:
                rel_id = offset // size
                block = block_pool._blocks.get((block_id, size, rel_id))
                if not block:
                    raise ValueError(f"Block not found for {address} at size {size}")
                
                if block_pool.is_allocated(block) or block.is_free:
                    return block
                    
            raise ValueError(f"No allocated or free block found for address {address}")

        moves: list[tuple[BuddyTreeBlock, BuddyTreeBlock]] = []
        swaps: list[tuple[BuddyTreeBlock, BuddyTreeBlock]] = []

        # Addresses of blocks that are sources of already-planned moves.
        # When a target lands on one of these addresses, it is safe to reuse it
        # as a destination as long as moves are executed in a safe order.
        planned_src_addrs: set[int] = set()
        
        # The segment starts at 'head' and is contiguous.
        curr_start = start_address + segment.head.size
        curr_logical_block = segment.head.next_block
        
        while curr_logical_block:
            src_block = curr_logical_block
            next_logical_block = src_block.next_block
            
            # 1. Identify Target Block at physical location
            target_block = find_leaf_block(curr_start)
            
            # 2. Check if already correct
            if target_block == src_block:
                curr_start += src_block.size
                curr_logical_block = next_logical_block
                continue
                
            # 3. Match sizes
            if target_block.size > src_block.size:
                target_block = block_pool.split_block(
                    target_block, src_block.size)
            
            if src_block.size > target_block.size:
                src_block = block_pool.split_block(src_block, target_block.size)
                # After split, src_block is the left child.
                # The right child is now part of the segment and will be seen next.
                next_logical_block = src_block.next_block
            
            # 4. Perform Swap
            target_segment = target_block.segment

            # If the target block is allocated but not part of any segment, it
            # can be an in-flight move source location. We can still complete
            # consolidation in one round iff this location is known to be a
            # source of an already-planned move in this same plan.
            if target_segment is None and not target_block.is_free:
                target_addr = block_pool.calculate_address(target_block)
                if target_addr not in planned_src_addrs:
                    # Cannot safely overwrite an unknown allocated block.
                    # Leave for the next scheduling step.
                    return None
            
            if target_segment is None:
                moves.append((src_block, target_block))

                planned_src_addrs.add(block_pool.calculate_address(src_block))
                
                # Metadata: src (me) -> target (free)
                # target becomes allocated (to me)
                if target_block.is_free:
                    block_pool.slabs[target_block.size].remove(target_block)
                    block_pool.allocated_blocks[target_block.size].add(target_block)
                
                # 1. Update segment mapping
                target_block.num_tokens = src_block.num_tokens
                replace_block_in_segment(src_block, [target_block])
                
                # 2. Release old block
                # The src_block will be freed later after the move is completed
                # src_block.reset()
                # block_pool.allocated_blocks[src_block.size].discard(src_block)
                # block_pool.slabs[src_block.size].append(src_block)
            else:
                swaps.append((src_block, target_block))
                
                # Metadata: Swap ownership/links using the helper
                swap_blocks(src_block, target_block)
            
            # 5. Advance
            curr_start += target_block.size
            curr_logical_block = next_logical_block

        if not moves and not swaps:
            return None

        return moves, swaps

    
    def get_num_tokens_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_segments: SemanticSegments,
    ) -> int:
        """
        Get the number of tokens needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_segments: The new computed segments just hitting the
                prefix caching.

        Returns:
            The number of tokens.
        """
        # Calculate total existing tokens for the request
        segments = self.req_to_segments.get(request_id, SemanticSegments())
        num_allocated_tokens = segments.capacity

        # Total tokens in the newly computed segments
        num_computed_tokens = sum(seg.capacity for seg in new_computed_segments)

        num_new_tokens = num_tokens - num_computed_tokens - num_allocated_tokens

        # If a computed segment is an eviction candidate (present in the
        # free queue with ref_cnt == 0), it will be converted from a free
        # segment to a computed segment when the request is allocated, so
        # we also count it as needing to be allocated.
        num_evictable_computed_tokens = sum(
            seg.capacity for seg in new_computed_segments if seg.ref_cnt == 0
        )

        return num_new_tokens + num_evictable_computed_tokens
    
    def save_new_computed_segments(
        self, request_id: str, new_computed_segments: SemanticSegments
    ) -> None:
        """
        Add the new computed segments to the request.
        
        Args:
            request_id: The request ID.
            new_computed_segments: The new computed segments just hitting the
                prefix cache.
        """
        segments = self.req_to_segments[request_id]
        segments += new_computed_segments

    def get_num_free_tokens(self) -> int:
        return (self.block_pool.get_num_free_tokens() + 
                self.num_free_segment_tokens)
