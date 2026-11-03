import argparse
import statistics
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM


MODEL_SIZES = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark a Transformer forward pass.")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="tiny")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1 or args.context_length < 1 or args.vocab_size < 1:
        parser.error("batch-size, context-length, and vocab-size must be positive")
    if args.warmup_steps < 0 or args.measure_steps < 2:
        parser.error("warmup-steps must be non-negative and measure-steps must be at least 2")

    # 准备模型和输入数据
    device = args.device
    # 模型规模
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **MODEL_SIZES[args.model_size],
    ).to(device)
    # 准备随机输入 tokens
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )

    # 预热
    for _ in range(args.warmup_steps):
        model(tokens)

    timings_ms = []
    for _ in range(args.measure_steps):
        torch.cuda.synchronize(device)
        start_time = timeit.default_timer()
        logits = model(tokens)
        torch.cuda.synchronize(device)
        end_time = timeit.default_timer()
        timings_ms.append((end_time - start_time) * 1000)

    print(f"Model: {args.model_size}; warmup: {args.warmup_steps}; measurements: {args.measure_steps}")
    for index, elapsed_ms in enumerate(timings_ms, start=1):
        print(f"Forward pass {index}: {elapsed_ms:.3f} ms")
    print(f"Mean: {statistics.mean(timings_ms):.3f} ms")
    print(f"Standard deviation: {statistics.stdev(timings_ms):.3f} ms")

    print("tokens:", tokens.shape)
    print("logits:", logits.shape)
    print("device:", logits.device)


if __name__ == "__main__":
    main()
