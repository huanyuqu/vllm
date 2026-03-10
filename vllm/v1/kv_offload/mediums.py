# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

import numpy as np

from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class AtomicRangeLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing contiguous KV ranges in atomic GPU-block units.

    Each row in ranges is `(start_block_id, length_in_atomic_blocks)`.
    """

    def __init__(self, ranges: list[tuple[int, int]]):
        self.ranges = np.array(ranges, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.ranges)


class MixedAtomicLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing a mix of contiguous atomic KV ranges and
    discrete atomic KV blocks.

    Each row in ranges is `(start_block_id, length_in_atomic_blocks)`.
    The block_ids list stores leftover atomic blocks that must be copied via
    explicit block mappings.
    """

    def __init__(
        self,
        ranges: list[tuple[int, int]],
        block_ids: list[int],
    ):
        self.ranges = np.array(ranges, dtype=np.int64).reshape((-1, 2))
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return f"ranges={self.ranges!r}, block_ids={self.block_ids!r}"


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class GPUAtomicRangeLoadStoreSpec(AtomicRangeLoadStoreSpec):
    """
    Spec for loading/storing contiguous KV ranges to GPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"


class CPUAtomicRangeLoadStoreSpec(AtomicRangeLoadStoreSpec):
    """
    Spec for loading/storing contiguous KV ranges to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"


class GPUMixedAtomicLoadStoreSpec(MixedAtomicLoadStoreSpec):
    """
    Spec for loading/storing mixed contiguous/discrete atomic KV data to GPU.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPUMixedAtomicLoadStoreSpec(MixedAtomicLoadStoreSpec):
    """
    Spec for loading/storing mixed contiguous/discrete atomic KV data to CPU.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"
