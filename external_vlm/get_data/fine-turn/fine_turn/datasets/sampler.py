from __future__ import annotations

import random
from typing import Sequence


class RoundRobinPairSampler:
    def __init__(self, lengths: Sequence[int], *, seed: int = 0):
        self.lengths = [int(x) for x in lengths]
        self.rng = random.Random(seed)

    def batches(self, batch_size: int):
        pools = [list(range(length)) for length in self.lengths]
        for pool in pools:
            self.rng.shuffle(pool)
        while any(pools):
            batch = []
            for dataset_id, pool in enumerate(pools):
                while pool and len(batch) < int(batch_size):
                    batch.append((dataset_id, pool.pop()))
                if len(batch) >= int(batch_size):
                    break
            if batch:
                yield batch

