from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor


__all__ = ["distributed_data_generator", "load_data_shard"]


def load_data_shard(file: str | Path) -> Tensor:
    """Load one FineWeb .bin shard into pinned CPU memory."""
    file = Path(file)
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _get_distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    seq_len: int = 1024,
    *,
    device: str | torch.device = "cuda",
    data_root: str | Path | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield local input/target batches from FineWeb .bin shards.

    If ``rank`` and ``world_size`` are omitted, the active torch.distributed
    process group is used. Without a process group, this falls back to a
    single-process loader.
    """
    if rank is None or world_size is None:
        default_rank, default_world_size = _get_distributed_context()
        rank = default_rank if rank is None else rank
        world_size = default_world_size if world_size is None else world_size

    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    root = Path.cwd() if data_root is None else Path(data_root)
    files = sorted(root.glob(filename_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No data files matched {filename_pattern!r} under {root}")

    file_iter = iter(files)
    tokens, pos = load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
