# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention import AttentionBackend
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.mediums import (
    AtomicRangeLoadStoreSpec,
    MixedAtomicLoadStoreSpec,
    CPULoadStoreSpec,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class CpuGpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert cpu_block_size % gpu_block_size == 0
        self.block_size_factor = cpu_block_size // gpu_block_size

        # cuda streams for gpu->cpu and cpu->gpu
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # job_id -> transfer cuda event
        self.transfer_events: dict[int, torch.Event] = {}
        # list of cuda events available for re-use
        self.events_pool: list[torch.Event] = []

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.gpu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        self.kv_dim_before_num_blocks: list[bool] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            self.gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            test_shape = attn_backends[layer_name].get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )
            if test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2

                num_blocks_idx = 1
                self.kv_dim_before_num_blocks.append(True)

            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * self.block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            self.cpu_tensors.append(
                torch.zeros(
                    cpu_shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )

    def _copy_tensor_ranges(
        self,
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
        src_start: int,
        dst_start: int,
        length: int,
        kv_dim_before_num_blocks: bool,
    ) -> None:
        if kv_dim_before_num_blocks:
            dst_tensor[0, dst_start : dst_start + length].copy_(
                src_tensor[0, src_start : src_start + length],
                non_blocking=True,
            )
            dst_tensor[1, dst_start : dst_start + length].copy_(
                src_tensor[1, src_start : src_start + length],
                non_blocking=True,
            )
        else:
            dst_tensor[dst_start : dst_start + length].copy_(
                src_tensor[src_start : src_start + length],
                non_blocking=True,
            )

    def _transfer_atomic_ranges_async(
        self,
        job_id: int,
        src_spec: AtomicRangeLoadStoreSpec,
        dst_spec: AtomicRangeLoadStoreSpec,
        stream: torch.cuda.Stream,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
    ) -> bool:
        src_ranges = src_spec.ranges
        dst_ranges = dst_spec.ranges
        assert src_ranges.shape == dst_ranges.shape

        event = self.events_pool.pop() if self.events_pool else torch.Event()
        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                src_tensors,
                dst_tensors,
                self.kv_dim_before_num_blocks,
            ):
                for (src_start, src_len), (dst_start, dst_len) in zip(
                    src_ranges.tolist(),
                    dst_ranges.tolist(),
                ):
                    assert src_len == dst_len
                    self._copy_tensor_ranges(
                        src_tensor,
                        dst_tensor,
                        src_start,
                        dst_start,
                        src_len,
                        kv_dim,
                    )
            event.record(stream)

        self.transfer_events[job_id] = event
        return True

    def _transfer_block_mapping_async(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        stream: torch.cuda.Stream,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        event: torch.Event,
    ) -> None:
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        dst_sub_blocks_to_skip = -src_blocks.size % dst_block_size_factor
        src_sub_block_count = src_blocks.size * src_block_size_factor

        assert (
            src_sub_block_count
            == dst_blocks.size * dst_block_size_factor - dst_sub_blocks_to_skip
        )

        src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])
        expand_block_ids(
            dst_blocks,
            dst_block_size_factor,
            src_to_dst[:, 1],
            skip_count=dst_sub_blocks_to_skip,
        )
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                src_tensors, dst_tensors, self.kv_dim_before_num_blocks
            ):
                if kv_dim:
                    src_key_cache = src_tensor[0]
                    dst_key_cache = dst_tensor[0]
                    ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst_tensor)
                    src_value_cache = src_tensor[1]
                    dst_value_cache = dst_tensor[1]
                    ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst_tensor)
                else:
                    ops.swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor)
            event.record(stream)

    def _transfer_mixed_atomic_async(
        self,
        job_id: int,
        src_spec: MixedAtomicLoadStoreSpec,
        dst_spec: MixedAtomicLoadStoreSpec,
        stream: torch.cuda.Stream,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
    ) -> bool:
        src_ranges = src_spec.ranges
        dst_ranges = dst_spec.ranges
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_ranges.shape == dst_ranges.shape
        assert src_blocks.shape == dst_blocks.shape

        event = self.events_pool.pop() if self.events_pool else torch.Event()
        with torch.cuda.stream(stream):
            if src_ranges.size:
                for src_tensor, dst_tensor, kv_dim in zip(
                    src_tensors,
                    dst_tensors,
                    self.kv_dim_before_num_blocks,
                ):
                    for (src_start, src_len), (dst_start, dst_len) in zip(
                        src_ranges.tolist(),
                        dst_ranges.tolist(),
                    ):
                        assert src_len == dst_len
                        self._copy_tensor_ranges(
                            src_tensor,
                            dst_tensor,
                            src_start,
                            dst_start,
                            src_len,
                            kv_dim,
                        )

        if src_blocks.size:
            self._transfer_block_mapping_async(
                src_blocks,
                dst_blocks,
                stream,
                src_tensors,
                dst_tensors,
                1,
                1,
                event,
            )
        else:
            with torch.cuda.stream(stream):
                event.record(stream)

        self.transfer_events[job_id] = event
        return True

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        if src_spec.medium() == CPULoadStoreSpec.medium():
            assert dst_spec.medium() == GPULoadStoreSpec.medium()
            stream = self.h2d_stream
            src_tensors = self.cpu_tensors
            dst_tensors = self.gpu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert src_spec.medium() == GPULoadStoreSpec.medium()
            assert dst_spec.medium() == CPULoadStoreSpec.medium()
            stream = self.d2h_stream
            src_tensors = self.gpu_tensors
            dst_tensors = self.cpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        if isinstance(src_spec, AtomicRangeLoadStoreSpec):
            assert isinstance(dst_spec, AtomicRangeLoadStoreSpec)
            return self._transfer_atomic_ranges_async(
                job_id,
                src_spec,
                dst_spec,
                stream,
                src_tensors,
                dst_tensors,
            )

        if isinstance(src_spec, MixedAtomicLoadStoreSpec):
            assert isinstance(dst_spec, MixedAtomicLoadStoreSpec)
            return self._transfer_mixed_atomic_async(
                job_id,
                src_spec,
                dst_spec,
                stream,
                src_tensors,
                dst_tensors,
            )

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        event = self.events_pool.pop() if self.events_pool else torch.Event()
        self._transfer_block_mapping_async(
            src_blocks,
            dst_blocks,
            stream,
            src_tensors,
            dst_tensors,
            src_block_size_factor,
            dst_block_size_factor,
            event,
        )

        self.transfer_events[job_id] = event

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, event in self.transfer_events.items():
            if event.query():
                results.append((job_id, True))
                self.events_pool.append(event)
        for job_id, _ in results:
            del self.transfer_events[job_id]
        return results
