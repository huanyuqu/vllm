import pytest
import torch

from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class FakeEngineCore:
    def __init__(self):
        self.calls = []

    async def consolidate_memory_async(self, request_id):
        self.calls.append(("consolidate", request_id))

    async def execute_semantic_memory_ops_async(self):
        self.calls.append(("execute",))
        return {
            "num_moves": 1,
            "num_swaps": 0,
            "moved_tokens": 8,
            "swapped_tokens": 0,
        }

    def shutdown(self):
        pass


@pytest.mark.asyncio
async def test_consolidate_memory_accepts_single_request_id():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.engine_core = FakeEngineCore()

    await engine.consolidate_memory("req-a")

    assert engine.engine_core.calls == [
        ("consolidate", "req-a"),
        ("execute",),
    ]


@pytest.mark.asyncio
async def test_consolidate_memory_batches_requests():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.engine_core = FakeEngineCore()

    result = await engine.consolidate_memory(["req-a", "req-b"])

    assert engine.engine_core.calls == [
        ("consolidate", "req-a"),
        ("consolidate", "req-b"),
        ("execute",),
    ]
    assert result == {
        "num_moves": 1,
        "num_swaps": 0,
        "moved_tokens": 8,
        "swapped_tokens": 0,
    }


@pytest.mark.asyncio
async def test_consolidate_memory_empty_request_list_noops():
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.engine_core = FakeEngineCore()

    result = await engine.consolidate_memory([])

    assert engine.engine_core.calls == []
    assert result == {
        "num_moves": 0,
        "num_swaps": 0,
        "moved_tokens": 0,
        "swapped_tokens": 0,
    }


def test_execute_semantic_memory_ops_applies_compaction_to_kv_cache():
    kv_cache = torch.arange(20, dtype=torch.float32).reshape(1, 20, 1)
    before = kv_cache.clone()
    moves = [(0, 2, 5, 3)]
    swaps = [(0, 10, 14, 2)]

    class FakeScheduler:
        def pop_pending_semantic_memory_ops(self):
            return moves, swaps

    class FakeExecutor:
        def __init__(self):
            self.calls = []
            self.runner = GPUModelRunner.__new__(GPUModelRunner)
            self.runner.kv_caches = [kv_cache]

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            self.calls.append((method, args))
            assert method == "apply_semantic_segment_memory_ops"
            self.runner.apply_semantic_segment_memory_ops(*args)

    executor = FakeExecutor()
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.scheduler = FakeScheduler()
    engine_core.model_executor = executor

    result = EngineCore.execute_semantic_memory_ops(engine_core)

    expected = before.clone()
    expected[:, 5:8, :] = before[:, 2:5, :]
    expected[:, 10:12, :] = before[:, 14:16, :]
    expected[:, 14:16, :] = before[:, 10:12, :]

    assert result == {
        "num_moves": 1,
        "num_swaps": 1,
        "moved_tokens": 3,
        "swapped_tokens": 2,
    }
    assert executor.calls == [
        ("apply_semantic_segment_memory_ops", (moves, swaps))
    ]
    assert torch.equal(kv_cache, expected)
