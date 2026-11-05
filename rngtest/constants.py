"""Эталонные последовательности (двоичные разложения математических констант).

NIST использует разложения e, π, √2, √3 как "заведомо хорошие" данные
(файлы data.e, data.pi и т.д. в sts-2.1.2). Здесь они вычисляются точно
целочисленной арифметикой, поэтому файлы данных не нужны.
"""
from __future__ import annotations

from functools import lru_cache
from math import isqrt

import numpy as np


def _int_to_bits(x: int, nbits: int) -> np.ndarray:
    raw = x.to_bytes((nbits + 7) // 8, "big")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[bits.size - nbits:]


def _e_series(a: int, b: int):
    """Бинарное разбиение для sum_{k=a+1}^{b} a!/k!: возвращает (P, Q)."""
    if b - a == 1:
        return 1, b
    m = (a + b) // 2
    p1, q1 = _e_series(a, m)
    p2, q2 = _e_series(m, b)
    return p1 * q2 + p2, q1 * q2


@lru_cache(maxsize=4)
def e_bits(n: int = 1_000_000) -> np.ndarray:
    """Первые n бит двоичной записи e = 10.1011011111... (как в файле data.e NIST:
    целая часть "10", затем дробная)."""
    # число членов K: log2(K!) > n + 64
    K, acc = 1, 0.0
    from math import log2
    while acc < n + 64:
        K += 1
        acc += log2(K)
    P, Q = _e_series(0, K)                  # e - 1 ≈ P/Q
    frac = ((P - Q) << n) // Q              # дробная часть e: e - 2 = (P - Q)/Q
    return np.concatenate([[1, 0], _int_to_bits(frac, n)[: n - 2]]).astype(np.uint8)


@lru_cache(maxsize=4)
def sqrt_bits(k: int, n: int = 1_000_000) -> np.ndarray:
    """Первые n бит дробной части √k."""
    r = isqrt(k << (2 * n))
    return _int_to_bits(r - (isqrt(k) << n), n)


@lru_cache(maxsize=4)
def pi_bits(n: int = 1_000_000) -> np.ndarray:
    """Первые n бит дробной части π (формула Мэчина, целочисленно)."""
    prec = n + 64
    one = 1 << prec

    def arctan_inv(x):
        total = term = one // x
        x2, k, sign = x * x, 3, -1
        while term:
            term //= x2
            total += sign * (term // k)
            sign, k = -sign, k + 2
        return total

    pi = 4 * (4 * arctan_inv(5) - arctan_inv(239))
    frac = (pi - (3 << prec)) >> 64
    return _int_to_bits(frac, n)
