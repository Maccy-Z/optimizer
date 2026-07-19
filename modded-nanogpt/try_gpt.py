"""
try_gpt.py

Load a checkpoint produced by train_gpt_simple.py and evaluate it on a slice of
the FineWeb validation data.
"""

import time
import gc
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dataloader import distributed_data_generator


CHECKPOINT_PATH = Path("logs/2026-07-04_00-06-23/3300.pt")
DATA_PATTERN = "data/fineweb10B/fineweb_val_*.bin"
DATA_ROOT = Path.cwd()
TOKENS = 1_048_576
BATCH_SIZE = 8 * 1024
WARMUP_STEPS = 4
SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODES = [
    "none",
    "attention",
    "mlp",
    "blocks",
    "head",
    "transformer",
    "full",
]


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def transformer(self, inputs: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        return x

    def head_loss(self, x: Tensor, targets: Tensor):
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

    def forward(self, inputs: Tensor, targets: Tensor):
        return self.head_loss(self.transformer(inputs), targets)


def get_state_dict(checkpoint_path: Path) -> dict[str, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict) or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise TypeError(f"{checkpoint_path} does not look like a model state_dict")

    return {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}


def infer_model_config(state_dict: dict[str, Tensor]) -> tuple[int, int, int]:
    vocab_size, model_dim = state_dict["embed.weight"].shape
    block_ids = {
        int(key.split(".", 2)[1])
        for key in state_dict
        if key.startswith("blocks.")
    }
    if len(block_ids) == 0:
        raise ValueError("Could not infer num_layers from checkpoint state_dict")
    return vocab_size, max(block_ids) + 1, model_dim


def synchronize_if_cuda() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


def clear_compile_state() -> None:
    torch._dynamo.reset()
    gc.collect()
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()


def apply_compile_mode(model: GPT, mode: str) -> GPT:
    if mode == "none":
        return model
    if mode == "attention":
        for block in model.blocks:
            block.attn.compile(dynamic=False)
        return model
    if mode == "mlp":
        for block in model.blocks:
            block.mlp.compile(dynamic=False)
        return model
    if mode == "blocks":
        for block in model.blocks:
            block.compile(dynamic=False)
        return model
    if mode == "head":
        model.head_loss = torch.compile(model.head_loss, dynamic=False)
        return model
    if mode == "transformer":
        model.transformer = torch.compile(model.transformer, dynamic=False)
        return model
    if mode == "full":
        model.compile(dynamic=False)
        return model
    raise ValueError(f"Unknown compile mode: {mode}")


@torch.no_grad()
def evaluate(model: GPT) -> tuple[float, int, float]:
    if BATCH_SIZE % SEQ_LEN != 0:
        raise ValueError("BATCH_SIZE must be divisible by SEQ_LEN")
    if TOKENS < BATCH_SIZE:
        raise ValueError("TOKENS must be at least BATCH_SIZE")

    eval_tokens = TOKENS - (TOKENS % BATCH_SIZE)
    eval_steps = eval_tokens // BATCH_SIZE
    loader = distributed_data_generator(
        DATA_PATTERN,
        BATCH_SIZE,
        seq_len=SEQ_LEN,
        device=DEVICE,
        data_root=DATA_ROOT,
        rank=0,
        world_size=1,
    )

    model.eval()
    for _ in range(WARMUP_STEPS):
        inputs, targets = next(loader)
        model(inputs, targets)
    synchronize_if_cuda()

    loss_sum = torch.zeros((), device=DEVICE)
    t0 = time.perf_counter()
    for _ in range(eval_steps):
        inputs, targets = next(loader)
        loss_sum += model(inputs, targets)
    synchronize_if_cuda()
    elapsed = time.perf_counter() - t0

    loss = loss_sum.item() / eval_tokens
    return loss, eval_tokens, elapsed / eval_steps


def main() -> None:
    device = torch.device(DEVICE)
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"data: {DATA_ROOT / DATA_PATTERN}")

    state_dict = get_state_dict(CHECKPOINT_PATH)
    config = infer_model_config(state_dict)
    results = []
    for mode in COMPILE_MODES:
        clear_compile_state()
        model = GPT(vocab_size=config[0], num_layers=config[1], model_dim=config[2])
        model.load_state_dict(state_dict)
        model.to(device)
        model = apply_compile_mode(model, mode)

        loss, eval_tokens, avg_time = evaluate(model)
        results.append((mode, loss, avg_time))
        print(f"{mode:>11}: loss {loss:.5f}, avg_time_per_step {avg_time:.4f}s")

        del model

    baseline = results[0][2]
    print(f"tokens: {eval_tokens}")
    print("speedups_vs_none:")
    for mode, _, avg_time in results:
        print(f"{mode:>11}: {baseline / avg_time:.2f}x")


if __name__ == "__main__":
    main()
