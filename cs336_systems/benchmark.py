import argparse
import statistics
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


MODEL_SIZES = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward, backward and training steps.")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="tiny")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("forward", "forward-backward", "train"), default="forward")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.context_length < 1 or args.vocab_size < 1:
        parser.error("batch-size, context-length, and vocab-size must be positive")
    if args.warmup_steps < 0 or args.measure_steps < 2:
        parser.error("warmup-steps must be non-negative and measure-steps must be at least 2")
    if args.learning_rate < 0:
        parser.error("learning-rate must be non-negative")

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
    # 随机目标与输入形状相同，每个位置对应词表中的一个目标 token。
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    ) if args.mode != "forward" else None
    # 优化器在计时前创建；第一次更新时会为参数创建状态。
    optimizer = AdamW(model.parameters(), lr=args.learning_rate) if args.mode == "train" else None

    def run_step():
        # 清除上一轮梯度，避免多次 backward 时梯度累积；这一步不计时。
        if args.mode != "forward":
            model.zero_grad(set_to_none=True)

        # 同步后再计时，确保前一轮 GPU 操作不会算入当前阶段。
        torch.cuda.synchronize(device)
        start = timeit.default_timer()
        logits = model(tokens)
        torch.cuda.synchronize(device)
        after_forward = timeit.default_timer()
        timings = {"Forward": (after_forward - start) * 1000}

        if args.mode != "forward":
            # 单独计入 loss 耗时，避免把它误认为 backward 耗时。
            loss = cross_entropy(logits, targets)
            torch.cuda.synchronize(device)
            after_loss = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize(device)
            after_backward = timeit.default_timer()
            timings["Loss"] = (after_loss - after_forward) * 1000
            timings["Backward"] = (after_backward - after_loss) * 1000
            end = after_backward
            if optimizer is not None:
                # 使用本轮梯度更新参数，单独记录优化器耗时。
                optimizer.step()
                torch.cuda.synchronize(device)
                end = timeit.default_timer()
                timings["Optimizer"] = (end - after_backward) * 1000
            timings["Total"] = (end - start) * 1000

        return timings, logits.shape

    # 预热和测量执行相同的流程，但不保存预热结果。
    for _ in range(args.warmup_steps):
        run_step()

    # measurements.shape = (args.measure_steps, len(timings))
    # timings 是一个字典，包含每个阶段的耗时（毫秒）；train 模式还包含 Optimizer。
    measurements = []
    for _ in range(args.measure_steps):
        timings, logits_shape = run_step()
        measurements.append(timings)

    print(f"Model: {args.model_size}; mode: {args.mode}; warmup: {args.warmup_steps}; measurements: {args.measure_steps}")
    for stage in measurements[0]:
        timings_ms = [measurement[stage] for measurement in measurements]
        print(f"{stage} times (ms): " + ", ".join(f"{value:.3f}" for value in timings_ms))
        print(f"{stage} mean: {statistics.mean(timings_ms):.3f} ms")
        print(f"{stage} standard deviation: {statistics.stdev(timings_ms):.3f} ms")

    print("tokens:", tokens.shape)
    print("logits:", logits_shape)
    print("device:", tokens.device)


if __name__ == "__main__":
    main()
