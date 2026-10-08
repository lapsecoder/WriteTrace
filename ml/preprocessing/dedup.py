"""Exact + near-duplicate detection with group labels for leakage-safe splitting.

- Exact duplicates: identical normalized text (case-sensitive) -> redundant rows
  are later dropped by the caller with a dataset priority order.
- Near duplicates: MinHash-estimated Jaccard >= JACCARD_THRESHOLD over 5-word
  shingles -> rows are GROUPED (kept, but must not straddle data splits).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

import numpy as np

from .minhash import candidate_pairs, draw_coefficients, minhash_signature, band_hashes, shingle_hashes

DEFAULTS = {
    "shingle_size": 5,
    "max_shingles": 4000,
    "num_perm": 128,
    "bands": 32,
    "rows": 4,
    "jaccard_threshold": 0.75,   # on the MinHash estimate (documented)
    "max_pairs_per_bucket": 5000,
    "seed": 13,
}


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def exact_duplicate_pairs(texts: list[str]) -> list[tuple[int, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        buckets[hashlib.sha1(t.encode("utf-8")).hexdigest()].append(i)
    pairs = []
    for members in buckets.values():
        for k in range(1, len(members)):
            pairs.append((members[0], members[k]))
    return pairs


def cluster_duplicates(texts: list[str], *, workers: int = 1, **overrides) -> tuple[np.ndarray, dict]:
    """Return (labels, stats). labels[i] == labels[j] iff rows are duplicates.

    Labels are deterministic: cluster ids assigned by order of first occurrence.
    """
    params = {**DEFAULTS, **overrides}
    n = len(texts)
    if n == 0:
        return np.empty(0, dtype=np.int64), {"n": 0}
    a, b = draw_coefficients(params["num_perm"], params["seed"])

    # 1) signatures (optionally in parallel; output order preserved)
    if workers and workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_sig_worker_init,
                      initargs=(params, a, b)) as pool:
            blobs = list(pool.map(_sig_worker, texts, chunksize=64))
        # workers return packed uint64 bytes -> rebuild (n, num_perm) matrix
        joined = b"".join(blobs)
        sigs = np.frombuffer(joined, dtype=np.uint64).reshape(len(blobs), params["num_perm"])
    else:
        sigs = np.stack([
            minhash_signature(shingle_hashes(t, params["shingle_size"], params["max_shingles"]), a, b)
            for t in texts
        ])
    if sigs.ndim != 2 or sigs.shape[1] != params["num_perm"]:
        raise RuntimeError(f"bad signature matrix shape {sigs.shape}")

    # 2) exact duplicates -> union first
    uf = _UnionFind(n)
    exact_pairs = exact_duplicate_pairs(texts)
    for i, j in exact_pairs:
        uf.union(i, j)

    # 3) LSH candidate pairs -> verify with signature Jaccard estimate
    bh = band_hashes(sigs, params["bands"], params["rows"], params["seed"])
    cands, overflow = candidate_pairs(bh, params["max_pairs_per_bucket"])
    near_pairs = 0
    if cands:
        idx = np.asarray(cands, dtype=np.int64)
        agree = (sigs[idx[:, 0]] == sigs[idx[:, 1]]).mean(axis=1)
        for (i, j), est in zip(cands, agree.tolist()):
            if est >= params["jaccard_threshold"]:
                uf.union(i, j)
                near_pairs += 1

    # 4) deterministic relabel by first occurrence
    roots = [uf.find(i) for i in range(n)]
    mapping: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for i, r in enumerate(roots):
        if r not in mapping:
            mapping[r] = len(mapping)
        labels[i] = mapping[r]

    stats = {
        "n": n,
        "params": params,
        "exact_pairs": len(exact_pairs),
        "candidate_pairs": len(cands),
        "buckets_overrun": overflow,
        "near_pairs_verified": near_pairs,
        "clusters": int(labels.max()) + 1,
        "rows_in_multirow_clusters": int(np.where(np.bincount(labels) > 1, np.bincount(labels), 0).sum()),
    }
    return labels, stats


# --- parallel worker plumbing (spawn-safe: top-level functions) -------------
_WORKER_PARAMS: dict = {}
_WORKER_A: np.ndarray | None = None
_WORKER_B: np.ndarray | None = None


def _sig_worker_init(params: dict, a: np.ndarray, b: np.ndarray) -> None:
    global _WORKER_PARAMS, _WORKER_A, _WORKER_B
    _WORKER_PARAMS, _WORKER_A, _WORKER_B = params, a, b


def _sig_worker(text: str) -> bytes:
    sh = shingle_hashes(text, _WORKER_PARAMS["shingle_size"], _WORKER_PARAMS["max_shingles"])
    return minhash_signature(sh, _WORKER_A, _WORKER_B).tobytes()
