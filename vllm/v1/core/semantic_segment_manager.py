from collections import defaultdict
from typing import Any, Optional
from dataclasses import dataclass, field
import itertools

from vllm.logger import init_logger
from vllm.v1.core.buddy_block_pool import BuddyBlockPool
from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    SegmentHash, 
    BuddyTreeBlock, 
    SemanticSegment, 
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
        self._cache: dict[SegmentHash, SemanticSegment | dict[int, SemanticSegment]] = {}

    def get_one_segment(self, key: SegmentHash) -> Optional[SemanticSegment]:
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

    def insert(self, key: SegmentHash, segment: SemanticSegment) -> None:
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
        self, key: SegmentHash, segment_id: int
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
    
    @property
    def last_segment(self) -> Optional[SemanticSegment]:
        """Get the last segment in the collection."""
        return self.segments[-1] if self.segments else None
    
    @property
    def unsealed_segment(self) -> Optional[SemanticSegment]:
        """Get the last unsealed segment, if any."""
        if self.segments and not self.segments[-1].is_sealed:
            return self.segments[-1]
        return None
    
    def __len__(self) -> int:
        return len(self.segments)


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
        
        # request_id -> list of segments used by this request
        # The last segment in the list might be unsealed (is_sealed=False)
        self.req_to_segments: dict[str, SemanticSegments] = defaultdict(SemanticSegments)
        
        # Prefix cache: hash -> SemanticSegment
        self.cached_segments: SegmentHashToSegmentMap = SegmentHashToSegmentMap()
        
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
        # Allocate by blocks and manage by segments
        blocks = self.block_pool.get_new_blocks(num_tokens)
        
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
        Seal the currently unsealed segment of a request.
        
        If a segment with the same hash already exists in the cache, we reuse it
        and free the newly allocated blocks (deduplication). (normally this
        will not happen)
        Otherwise, we mark the current segment as sealed and add it to the cache.
        """
        request_id = request.request_id
        segments = self.req_to_segments[request_id]
        unsealed_segment = segments.unsealed_segment
        if not unsealed_segment:
            logger.warning(f"Request {request_id} has no unsealed segment to seal.")
            return
        
        segment_hash = request.segment_hashes[-1] if request.segment_hashes else None
        if segment_hash is None:
            raise ValueError(f"Request {request_id} has no segment hash and cannot seal segment.")
        
        segment_hash_with_group_id = make_segment_hash_with_group_id(
            segment_hash, kv_cache_group_id
        )
        unsealed_segment.segment_hash = segment_hash_with_group_id
        
        # Check for prefix cache hit
        cached_segment = self.cached_segments.get_one_segment(
            unsealed_segment.segment_hash
        )
        if cached_segment:
            # Cache hit: replace unsealed segment with cached segment
            segments.last_segment = cached_segment
            cached_segment.ref_cnt += 1
            del unsealed_segment
        else:
            # Cache miss: seal the current segment
            unsealed_segment.seal()
            self.cached_segments.insert(unsealed_segment.segment_hash, 
                                        unsealed_segment)
        self.block_pool.free_blocks(reversed(unsealed_segment.blocks))

    def free(self, request_id: str) -> None:
        """
        Free resources associated with a request.
        
        1. Free any unfinalized blocks.
        2. Decrement reference counts for finalized segments.
        """
        # Default to empty SemanticSegments in case request is freed before allocation
        segments = self.req_to_segments.pop(request_id, SemanticSegments())
        
        for segment in reversed(segments.segments):
            if segment.is_sealed:
                segment.ref_cnt -= 1
                if segment.ref_cnt == 0:
                    self.free_segment_queue.append(segment)
                # Note: We do not automatically free the segment from cache when ref_cnt drops to 0.
                # It remains in cached_segments for future reuse (prefix caching).
                # Eviction should be handled by a separate policy or when memory is low.
            else:
                # When segment is unsealed, the management is still at the block level.
                # Free blocks of unsealed segment in reverse order
                self.block_pool.free_blocks(reversed(segment.blocks))
                
    # TODO(huanyu): This method is not expected to be used during serving; it is primarily for RL.
    def reset(self) -> None:
        raise NotImplementedError("SemanticSegmentManager.reset is not implemented yet.")
                
    def get_cached_segment(
        self, segment_hash: SegmentHash, kv_cache_group_ids: list[int]
    ) -> Optional[list[SemanticSegment]]:
        """Get the cached segment by the segment hash for the given group,
        or None if cache miss.

        Args:
            segment_hash: The hash value of the segment.
            kv_cache_group_id: The id of the KV cache group.

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
        matched_segments_count = 0
        
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
            matched_segments_count += 1

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
