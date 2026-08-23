"""Perceptual-hash duplicate/similar detection."""
from __future__ import annotations

import numpy as np


def dhash(preview_u8: np.ndarray, size: int = 8) -> int:
    """Difference hash — 64-bit int."""
    from PIL import Image
    gray = np.asarray(Image.fromarray(
        preview_u8).convert("L").resize((size+1, size),
        Image.Resampling.LANCZOS), dtype=np.int16)
    # horizontal gradient: each pixel > its left neighbour
    diff = gray[:, 1:] > gray[:, :-1]
    out = 0
    for bit in diff.ravel():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_similar(hashes: dict[int, int], threshold: int = 8) -> list[list[int]]:
    """Group photo IDs whose hashes are within threshold Hamming distance.
    Brute force O(n²) but fine up to ~5k photos."""
    ids = list(hashes.keys())
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if hamming(hashes[ids[i]], hashes[ids[j]]) <= threshold:
                union(ids[i], ids[j])

    groups: dict[int, list[int]] = {}
    for pid in ids:
        root = find(pid)
        groups.setdefault(root, []).append(pid)
    return [g for g in groups.values() if len(g) > 1]
