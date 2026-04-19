"""
GF(4) — The 32-byte compute engine.

Two 4x4 lookup tables. 32 bytes total. Fits in one cache line.
This is the ENTIRE compute substrate for QPU inference.
"""

import numpy as np

ADD = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
], dtype=np.uint8)

MUL = np.array([
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
], dtype=np.uint8)

INV = np.array([0, 1, 3, 2], dtype=np.uint8)

COMP = np.array([0, 1, 2, 3], dtype=np.uint8)  # self-inverse under addition


def add_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ADD[a, b]


def mul_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return MUL[a, b]
