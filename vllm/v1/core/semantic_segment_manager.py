from collections import defaultdict
from typing import Any, Optional
from dataclasses import dataclass, field
import itertools

from vllm.logger import init_logger
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    SegmentHash,
    SegmentHashWithGroupId, 
    BuddyTreeBlock, 
    SemanticSegment, 
    get_block_hash,
    make_segment_hash_with_group_id,
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
    
    def append(self, segment: SemanticSegment) -> None:
        """Add a segment to the collection."""
        self.segments.append(segment)
        
    def pop(self) -> SemanticSegment:
        """Remove and return the last segment."""
        return self.segments.pop()

    def extend(self, segments: list[SemanticSegment]) -> None:
        """Extend the collection with multiple segments."""
        self.segments.extend(segments)
        
    def __getitem__(self, key):
        """Support indexing and slicing: segment = semantic_segments[0] or segments = semantic_segments[1:3]"""
        if isinstance(key, slice):
            return SemanticSegments(self.segments[key])
        else:
            return self.segments[key]
    
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
    
    def __getitem__(self, key: slice | int):
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

    def __init__(self, block_pool: BuddyBlockPool):
        self.block_pool = block_pool
        
        # request_id -> semantic segments used by this request
        # The last segment might be unsealed (is_sealed=False)
        self.req_to_segments: dict[str, SemanticSegments] = defaultdict(SemanticSegments)
        
        # Prefix cache: hash -> SemanticSegment
        self.cached_segments = SegmentHashToSegmentMap()
        
        # This queue records the segments with ref_cnt = 0
        # All segments in this queue must in self.cached_segments
        self.free_segment_queue = FreeKVCacheBlockQueue([])
        
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
            while remaining > 0 and self.free_segment_queue.num_free_blocks > 0:
                segment: SemanticSegment = self.free_segment_queue.popleft()
                self._maybe_evict_cached_segment(segment)
                segment.unseal()
                self.block_pool.free_blocks(segment.blocks)
                
                new_blocks, new_remaining = self.block_pool.get_new_blocks(remaining)
                if new_blocks:
                    blocks.extend(new_blocks)
                remaining = new_remaining

            if remaining > 0:
                # 3. Reclaim from allocated blocks if still needed
                reclaimed_blocks = self.block_pool.reclaim_new_blocks(remaining)
                blocks.extend(reclaimed_blocks)
        
        segments = self.req_to_segments[request_id]
        if not segments.unsealed_segment:
            # Create new unsealed segment
            segment_id = SegmentIdGenerator().generate()
            new_segment = SemanticSegment(segment_id=segment_id)
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
        
        segment.reset_hash()

        return True

    def seal_segment(
        self, request: Request, kv_cache_group_id: int
    ) -> None:
        """
        Seal the unsealed segment for a request by processing its blocks.
        
        For each block in the unsealed segment:
        - If the block is sealed, reuse the existing sealed segment and increment its ref_cnt.
        - If the block is unsealed, group consecutive blocks with the same ref_cnt into a new segment,
          seal it, compute its hash, and insert it into the cache.
        
        Extend the request's segment list with the processed segments.
        """
        request_id = request.request_id
        segments = self.req_to_segments[request_id]
        unsealed_segment = segments.unsealed_segment
        if not unsealed_segment:
            logger.debug(f"Request {request_id} has no unsealed segment to seal.")
            return
        
        blocks = unsealed_segment.blocks
        if not blocks:
            raise ValueError(f"Cannot seal segment for request {request_id}: "
                             f"no blocks in unsealed segment.")


        new_segments_list = []
        current_idx = 0
        
        while current_idx < len(blocks):
            current_block = blocks[current_idx]
            
            if current_block.is_sealed:
                sealed_segment = current_block.segment
                
                # Find prefix length
                end_idx = current_idx + 1
                while end_idx < len(blocks):
                    blk = blocks[end_idx]
                    if blk.is_sealed and blk.segment == sealed_segment:
                        end_idx += 1
                    else:
                        break
                
                # Reuse sealed_segment
                sealed_segment.ref_cnt += 1
                new_segments_list.append(sealed_segment)
                current_idx = end_idx
            else:
                # Unsealed sequence
                # Since we cannot have sealed blocks after unsealed blocks,
                # the rest of the blocks must be unsealed.
                # We group consecutive blocks with the same ref_cnt into one segment.
                while current_idx < len(blocks):
                    start_idx = current_idx
                    current_ref_cnt = blocks[start_idx].ref_cnt
                    end_idx = start_idx + 1
                    
                    while end_idx < len(blocks):
                        if blocks[end_idx].ref_cnt == current_ref_cnt:
                            end_idx += 1
                        else:
                            break
                    
                    sub_blocks = blocks[start_idx:end_idx]
                    last_block_hash_with_group_id = sub_blocks[-1].block_hash
                    if last_block_hash_with_group_id is None:
                        raise ValueError(
                            f"Cannot seal segment for request {request_id}: "
                            "last block has no block_hash."
                        )

                    # SegmentHash is group-agnostic; we pack group id only when
                    # forming the cache key.
                    segment_hash = SegmentHash(get_block_hash(last_block_hash_with_group_id))
                    segment_hash_with_group_id = make_segment_hash_with_group_id(
                        segment_hash, kv_cache_group_id
                    )
                    new_segment = SemanticSegment(
                        segment_id=SegmentIdGenerator().generate(),
                        blocks=sub_blocks
                    )
                    new_segment.seal(current_ref_cnt)
                    new_segment.segment_hash = segment_hash_with_group_id
                    self.cached_segments.insert(
                        segment_hash_with_group_id, new_segment
                    )
                    new_segments_list.append(new_segment)
                    current_idx = end_idx
        segments.segments[-1:] = new_segments_list
        return

    def free(self, request: Request, kv_cache_group_id: int) -> None:
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
        self.seal_segment(request, kv_cache_group_id)
        request_id = request.request_id
        segments = self.req_to_segments.pop(request_id, SemanticSegments())
        
        for segment in reversed(segments):
            segment.ref_cnt -= 1
            if segment.ref_cnt == 0:
                self.free_segment_queue.append(segment)  # type: ignore
                
    # TODO(huanyu): This method is not expected to be used during serving; it is primarily for RL.
    def reset(self) -> None:
        raise NotImplementedError("SemanticSegmentManager.reset is not implemented yet.")
                
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
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_size: int,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[BuddyTreeBlock], ...]:
        computed_blocks: tuple[list[BuddyTreeBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
            
        # 1. Try to match full segments first
        current_token_count = 0
        
        for segment_hash in segment_hashes:
            cached_segments = self.get_cached_segment(segment_hash, kv_cache_group_ids)
            if not cached_segments:
                break
                
            # Verify segment length doesn't exceed max_length
            segment_len = len(cached_segments[0].blocks) * block_size
            if current_token_count + segment_len > max_length:
                break
                
            # Add segment blocks to computed_blocks
            for computed, segment in zip(computed_blocks, cached_segments):
                computed.extend(segment.blocks)
                
            current_token_count += segment_len

        # 2. For the remaining part, try to match individual blocks
        # Calculate starting block index for block matching
        start_block_idx = current_token_count // block_size
        max_num_blocks = max_length // block_size
        
        for block_hash in itertools.islice(block_hashes, start_block_idx, max_num_blocks):
            if cached_block := self.block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
                
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
                
        return computed_blocks
