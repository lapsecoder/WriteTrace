"""MinHash signatures + LSH banding (deterministic, seeded, numpy only)."""
from __future__ import annotations

import re
import zlib

import numpy as np

TOKEN_RE = re.compile(r"\b[\w’']+\b", re.UNICODE)
_U64 = np.iinfo(np.uint64)


def token_hashes(text: str) -> list[int]:
    """Stable per-token hashes (crc32, NOT Python's randomized hash())."""
    return [zlib.crc32(w.lower().encode("utf-8")) for w in TOKEN_RE.findall(text)]


def shingle_hashes(text: str, shingle_size: int = 5, max_shingles: int = 4000) -> np.ndarray:
    """Sorted unique uint64 hashes of word n-gram shingles. Deterministic."""
    ids = token_hashes(text)
    if not ids:
        return np.empty(0, dtype=np.uint64)
    if len(ids) < shingle_size:
        # Whole text as one shingle (short texts still dedupable).
        joined = " ".join(TOKEN_RE.findall(text.lower()))
        return np.array([zlib.crc32(joined.encode("utf-8"))], dtype=np.uint64)
    arr = np.asarray(ids, dtype=np.uint64)
    weights = np.asarray([np.uint64(131 ** i) for i in range(shingle_size)], dtype=np.uint64)
    win = np.convolve(arr, weights, mode="valid")[:max_shingles]  # wraps mod 2^64 by design
    return np.unique(win)


def minhash_signature(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One MinHash signature. a/b are pre-drawn seeded coefficients (num_perm each)."""
    if shingles.size == 0:
        return np.zeros(a.size, dtype=np.uint64)
    # (k,1)*(P,) -> (k,P); uint64 wraparound is deterministic.
    proj = shingles[:, None] * a[None, :] + b[None, :]
    return proj.min(axis=0)


def draw_coefficients(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _U64.max, num_perm, dtype=np.uint64)
    b = rng.integers(0, _U64.max, num_perm, dtype=np.uint64)
    return a, b


def band_hashes(signatures: np.ndarray, bands: int, rows: int, seed: int) -> np.ndarray:
    """Fold each band (r consecutive signature entries) into one uint64.

    Returns shape (n_docs, bands) — one hash PER BAND, so docs sharing ANY band
    become LSH candidates. Deterministic for a fixed seed.
    """
    rng = np.random.default_rng(seed + 101)
    w = rng.integers(1, _U64.max, rows, dtype=np.uint64)
    n = signatures.shape[0]
    per_band = signatures.shape[1] // bands
    out = np.zeros((n, bands), dtype=np.uint64)
    for i in range(bands):
        chunk = signatures[:, i * per_band:(i + 1) * per_band]
        acc = np.zeros(n, dtype=np.uint64)  # zeros, never np.empty (uninitialized!)
        for j in range(per_band):
            acc = acc * np.uint64(31) + chunk[:, j] * w[j]
        out[:, i] = acc * np.uint64(0x9E3779B1 + i + 1)
    return out


def candidate_pairs(band_hash: np.ndarray, max_pairs_per_bucket: int = 5000) -> tuple[list[tuple[int, int]], int]:
    """All doc pairs sharing at least one band bucket (deterministic, capped)."""
    if band_hash.ndim != 2:
        raise ValueError("band_hash must be shape (n_docs, n_bands)")
    n, n_bands = band_hash.shape
    if n < 2:
        return [], 0
    arange64 = np.arange(n, dtype=np.uint64)
    pairs: set[tuple[int, int]] = set()
    overflow = 0
    for bi in range(n_bands):
        col = band_hash[:, bi]
        order = np.lexsort((arange64, col))
        scol = col[order]
        change = np.flatnonzero(scol[1:] != scol[:-1]) + 1
        starts = np.concatenate((np.array([0]), change))
        ends = np.concatenate((change, np.array([n])))
        for s, e in zip(starts.tolist(), ends.tolist()):
            length = e - s
            if length < 2:
                continue
            members = [int(i) for i in order[s:e]]
            count = 0
            for i in range(length - 1):
                for j in range(i + 1, length):
                    a, b = members[i], members[j]
                    pairs.add((a, b) if a < b else (b, a))
                    count += 1
                    if count >= max_pairs_per_bucket:
                        overflow += 1
                        break
                if count >= max_pairs_per_bucket:
                    break
    return sorted(pairs), overflow
