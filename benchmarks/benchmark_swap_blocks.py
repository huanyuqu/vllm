import torch
from vllm import _custom_ops as ops

def run_benchmark():
    # 模拟设置：
    # 隐层维度 4096, 32个Head, 16个Token为一个Block
    # FP16每元素2字节，K和V各占一半
    # Block Size: 16 * 4096 * 2 (KV) * 2 (bytes) = 262,144 bytes = 256 KB
    num_layers = 36
    seq_len = 32 * 1024
    block_size = 16
    head_dim = 128
    num_kv_heads = 8
    # 逻辑 block table 在多层间共享，因此 block 数只由序列长度决定。
    num_blocks = seq_len // block_size
    block_size_bytes = block_size * head_dim * num_kv_heads * 2 * 2  # float16 has 2 bytes
    elements_per_block = block_size_bytes // 2

    total_mb = (num_layers * num_blocks * block_size_bytes) / (1024**2)
    print(f"Total Data Size: {total_mb:.2f} MB")
    print(f"Block Size: {block_size_bytes / 1024:.2f} KB\n")

    # 1. 准备数据源和目标
    # 【核心】务必使用 pin_memory 控制 CPU 在锁页内存区，否则都会被缺页中断拖慢
    print("Allocating pinned memory on CPU... ", end="")
    src_layers = [
        torch.randn(num_blocks, elements_per_block, dtype=torch.float16).pin_memory()
        for _ in range(num_layers)
    ]
    print("Done.")
    dst_layers = [
        torch.empty(num_blocks, elements_per_block, dtype=torch.float16, device="cuda")
        for _ in range(num_layers)
    ]

    # 2. 生成极端随机碎片的 block_mapping
    # 多层共用同一份 block table/mapping。
    # 从 CPU 的一个随机乱序的 Block，拷贝到 GPU 上随机乱序的 Block 中
    map_src = torch.randperm(num_blocks, dtype=torch.int64)
    map_dst = torch.randperm(num_blocks, dtype=torch.int64)
    block_mapping = torch.stack([map_src, map_dst], dim=1).cpu()

    # == 测试 1：作为 Baseline 连续拷贝整个一大块内存 ==
    def test_contiguous_copy():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # 非阻塞逐层连续拷贝（共享 block table）
        for src, dst in zip(src_layers, dst_layers):
            dst.copy_(src, non_blocking=True)
        end.record()
        
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # == 测试 2：利用 vLLM 底层机制传输碎片 Block ==
    def test_vllm_swap_blocks():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # csrc 底层利用一个 for 循环连续发动 cudaMemcpyAsync
        # 这里逐层调用，但每层使用相同 block_mapping。
        for src, dst in zip(src_layers, dst_layers):
            ops.swap_blocks(src, dst, block_mapping)
        end.record()
        
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print("\nWarming up...")
    test_contiguous_copy()
    test_vllm_swap_blocks()

    print("Benchmarking...")
    iters = 10
    
    time_contig = sum(test_contiguous_copy() for _ in range(iters)) / iters
    time_swap = sum(test_vllm_swap_blocks() for _ in range(iters)) / iters

    bw_contig = total_mb / (time_contig / 1000) / 1024 # 转换为 GB/s
    bw_swap_vllm = total_mb / (time_swap / 1000) / 1024 # 转换为 GB/s

    print("\n====== Benchmark Results ======")
    print(f"[1. PyTorch Contiguous Copy] \n  Latency: {time_contig:.2f} ms \n  Bandwidth: {bw_contig:.2f} GB/s")
    print(f"[2. vLLM Random Blocks Swap] \n  Latency: {time_swap:.2f} ms \n  Bandwidth: {bw_swap_vllm:.2f} GB/s")
    print("===============================\n")

if __name__ == '__main__':
    run_benchmark()