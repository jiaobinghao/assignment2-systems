import torch
import argparse
import timeit
import torch

from cs336_basics.model import BasicsTransformerLM


def main():
    # 准备模型和输入数据
    device = "cuda"
    # 前向传播预热次数
    warmup_steps = 10
    # 模型规模
    model = BasicsTransformerLM(
        vocab_size=1000,
        context_length=32,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
    ).to(device)
    # 准备输入tokens，形状为(2, 32)，表示2个样本，每个样本有32个token
    tokens = torch.randint(
        low=0,
        high=1000,
        size=(2, 32),
        device=device,
    )

    # 预热
    for _ in range(warmup_steps):
        _ = model(tokens)
    
    # forward
    torch.cuda.synchronize(device)  # 确保所有CUDA操作完成
    start_time = timeit.default_timer() * 1000  # 转换为毫秒
    logits = model(tokens)
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer() * 1000

    print(f"Forward pass time: {end_time - start_time:.6f} milliseconds")

    print("tokens:", tokens.shape)
    print("logits:", logits.shape)
    print("device:", logits.device)


if __name__ == "__main__":
    main()