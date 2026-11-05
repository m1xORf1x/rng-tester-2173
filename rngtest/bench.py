"""Профилирование и сравнение реализаций (ТЗ п. 6.2, этап 2.2).

Для ключевых операций сравниваются три уровня реализации:
* "Python" — наивная реализация циклом на чистом Python (как в учебниках);
* "NumPy"  — векторизованная реализация (используется по умолчанию);
* "C"      — нативное ядро на упакованных битах / с автовекторизацией (-O3).
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Dict, List

import numpy as np

from . import native
from .tests_nist import berlekamp_massey_batch


def _timeit(fn: Callable, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return best, peak


# --- наивные реализации
def _py_ones(bits_list):
    c = 0
    for b in bits_list:
        c += b
    return c


def _py_autocorr(bits_list, d):
    return sum(bits_list[i] ^ bits_list[i + d] for i in range(len(bits_list) - d))


def _py_bm(s):
    n = len(s)
    C, Bp = [1] + [0] * n, [1] + [0] * n
    L, m = 0, -1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            d ^= C[j] & s[i - j]
        if d:
            T = C[:]
            for j in range(0, n - (i - m) + 1):
                C[j + i - m] ^= Bp[j]
            if L <= i // 2:
                L, m, Bp = i + 1 - L, i, T
    return L


def benchmark(n_bits: int = 1_000_000, include_python: bool = True) -> List[Dict]:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
    rows = []
    have_c = native.available()
    packed = native.pack(bits) if have_c else None
    bl = bits.tolist() if include_python else None

    def add(op, impl, fn, n=n_bits, repeat=3):
        t, mem = _timeit(fn, repeat)
        rows.append({"operation": op, "implementation": impl, "n_bits": n, "time_s": t,
                     "throughput_mbit_s": n / t / 1e6, "peak_mem_bytes": mem})

    # 1. подсчёт единиц (частотный тест)
    if include_python:
        add("Частотный тест (подсчёт единиц)", "Python", lambda: _py_ones(bl), repeat=1)
    add("Частотный тест (подсчёт единиц)", "NumPy", lambda: int(bits.sum()))
    if have_c:
        add("Частотный тест (подсчёт единиц)", "C (popcount, упакованные биты)",
            lambda: native.packed_popcount(packed, n_bits))

    # 2. автокорреляция d=1 (= число серий − 1)
    if include_python:
        add("Автокорреляция / серии (d=1)", "Python", lambda: _py_autocorr(bl, 1), repeat=1)
    add("Автокорреляция / серии (d=1)", "NumPy", lambda: int(np.count_nonzero(bits[1:] != bits[:-1])))
    if have_c:
        add("Автокорреляция / серии (d=1)", "C (XOR+popcount, упакованные биты)",
            lambda: native.packed_autocorr(packed, n_bits, 1))
        add("Автокорреляция / серии (d=8)", "C (XOR+popcount, упакованные биты)",
            lambda: native.packed_autocorr(packed, n_bits, 8))
    add("Автокорреляция / серии (d=8)", "NumPy", lambda: int(np.count_nonzero(bits[8:] != bits[:-8])))

    # 3. Берлекэмп–Мэсси (линейная сложность), M = 500
    M = 500
    nb = min(n_bits // M, 2000)
    sub = bits[: nb * M]
    if include_python:
        k = max(nb // 20, 1)  # наивный Python очень медленный — меряем на части и масштабируем
        sl = [sub[i * M:(i + 1) * M].tolist() for i in range(k)]
        t, mem = _timeit(lambda: [_py_bm(s) for s in sl], 1)
        t = t * nb / k
        rows.append({"operation": "Линейная сложность (Берлекэмп–Мэсси)", "implementation": "Python",
                     "n_bits": nb * M, "time_s": t, "throughput_mbit_s": nb * M / t / 1e6,
                     "peak_mem_bytes": mem})
    add("Линейная сложность (Берлекэмп–Мэсси)", "NumPy (векторизация по блокам)",
        lambda: berlekamp_massey_batch(sub.reshape(nb, M)), n=nb * M, repeat=1)
    if have_c:
        add("Линейная сложность (Берлекэмп–Мэсси)", "C", lambda: native.lc_blocks(sub, nb, M), n=nb * M)

    # 4. хранение последовательности
    rows.append({"operation": "Хранение 1 бита", "implementation": "uint8 на бит",
                 "n_bits": n_bits, "time_s": 0, "throughput_mbit_s": None,
                 "peak_mem_bytes": bits.nbytes})
    rows.append({"operation": "Хранение 1 бита", "implementation": "упакованные биты",
                 "n_bits": n_bits, "time_s": 0, "throughput_mbit_s": None,
                 "peak_mem_bytes": (n_bits + 7) // 8})
    return rows


def scaling(test_ids: List[str], sizes=(10 ** 4, 10 ** 5, 10 ** 6)) -> List[Dict]:
    """Зависимость времени и памяти тестов от длины последовательности."""
    from .runner import run_test
    rng = np.random.default_rng(2)
    rows = []
    for n in sizes:
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        for t in test_ids:
            r = run_test(t, bits)
            if r.status in ("pass", "fail"):
                rows.append({"test": t, "name": r.name, "n_bits": n, "time_s": r.time_s,
                             "peak_mem_bytes": r.peak_mem_bytes})
    return rows
