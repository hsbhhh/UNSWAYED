from __future__ import annotations


def is_distributed() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def rank() -> int:
    if not is_distributed():
        return 0
    import torch.distributed as dist

    return int(dist.get_rank())

