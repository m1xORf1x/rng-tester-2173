"""Общие математические функции, используемые несколькими батареями."""
from __future__ import annotations

from math import log, sqrt
from typing import Sequence, Tuple

import numpy as np
from scipy import special, stats


def igamc(a: float, x: float) -> float:
    """Регуляризованная верхняя неполная гамма-функция Q(a, x) (как в NIST STS)."""
    return float(special.gammaincc(a, x))


def erfc(x: float) -> float:
    return float(special.erfc(x))


def normal_two_sided(z: float) -> float:
    return erfc(abs(z) / sqrt(2.0))


def chi2_test(observed: Sequence[float], expected: Sequence[float],
              min_expected: float = 5.0, ddof: int = 0) -> Tuple[float, int, float]:
    """Критерий хи-квадрат с объединением "редких" соседних категорий.

    Категории с ожидаемой частотой < ``min_expected`` склеиваются с соседними
    (как это делает TestU01), чтобы аппроксимация распределением χ² была корректной.
    Возвращает (статистика, число степеней свободы, p-значение).
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    o_m, e_m = [], []
    acc_o = acc_e = 0.0
    for o, e in zip(obs, exp):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            o_m.append(acc_o)
            e_m.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        if e_m:
            o_m[-1] += acc_o
            e_m[-1] += acc_e
        else:
            o_m.append(acc_o)
            e_m.append(acc_e)
    o_m, e_m = np.array(o_m), np.array(e_m)
    df = len(o_m) - 1 - ddof
    if df < 1:
        return float("nan"), 0, float("nan")
    chi2 = float(np.sum((o_m - e_m) ** 2 / e_m))
    return chi2, df, float(stats.chi2.sf(chi2, df))


def ks_uniform(p_values: Sequence[float]) -> float:
    """p-значение критерия Колмогорова–Смирнова на равномерность (второй уровень)."""
    p = np.asarray(p_values, dtype=float)
    if p.size < 2:
        return float(p[0]) if p.size else float("nan")
    return float(stats.kstest(p, "uniform").pvalue)


# ---------------------------------------------------------------- GF(2) ранги

def gf2_rank_batch(rows: np.ndarray, ncols: int) -> np.ndarray:
    """Ранги пачки двоичных матриц над GF(2).

    ``rows`` — массив формы (N, M) uint64: N матриц, M строк, каждая строка —
    целое число, в котором ``ncols`` <= 64 младших бит являются элементами строки.
    Метод Гаусса векторизован "поперёк" матриц: цикл только по столбцам.
    """
    rows = rows.astype(np.uint64, copy=True)
    N, M = rows.shape
    rank = np.zeros(N, dtype=np.int64)
    used = np.zeros((N, M), dtype=bool)
    idx = np.arange(N)
    for c in range(ncols - 1, -1, -1):
        bit = ((rows >> np.uint64(c)) & np.uint64(1)).astype(bool)
        cand = bit & ~used
        has = cand.any(axis=1)
        if not has.any():
            continue
        piv = np.argmax(cand, axis=1)
        pivrow = rows[idx, piv]
        elim = bit & has[:, None]
        elim[idx, piv] = False
        rows ^= np.where(elim, pivrow[:, None], np.uint64(0))
        used[idx[has], piv[has]] = True
        rank += has
    return rank


def rank_probabilities(M: int, Q: int) -> np.ndarray:
    """Вероятности P(rank = r), r = 0..min(M,Q), для случайной M×Q матрицы над GF(2)."""
    R = min(M, Q)
    probs = np.zeros(R + 1)
    for r in range(R + 1):
        lp = (r * (Q + M - r) - M * Q) * log(2.0)
        for i in range(r):
            lp += log(1 - 2.0 ** (i - Q)) + log(1 - 2.0 ** (i - M)) - log(1 - 2.0 ** (i - r))
        probs[r] = np.exp(lp)
    return probs


# ---------------------------------------------------------------- преобразования

def bits_to_words(bits: np.ndarray, w: int = 32) -> np.ndarray:
    """Нарезать поток бит на w-битные беззнаковые слова (старший бит первым)."""
    n = bits.size // w
    if n == 0:
        return np.zeros(0, dtype=np.uint64)
    b = bits[: n * w].reshape(n, w).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(w - 1, -1, -1, dtype=np.uint64))
    return (b * weights).sum(axis=1, dtype=np.uint64)


def bits_to_uniforms(bits: np.ndarray) -> np.ndarray:
    """32-битные слова -> числа из [0, 1) (как в DieHard/TestU01)."""
    return bits_to_words(bits, 32).astype(np.float64) / 4294967296.0


def window_values(bits: np.ndarray, m: int, cyclic: bool = False) -> np.ndarray:
    """Значения всех перекрывающихся m-битных окон как целые числа."""
    x = np.concatenate([bits, bits[: m - 1]]) if cyclic and m > 1 else bits
    L = x.size - m + 1
    if L <= 0:
        return np.zeros(0, dtype=np.int64)
    v = np.zeros(L, dtype=np.int64)
    for j in range(m):
        v = (v << 1) | x[j: j + L]
    return v


def stirling2_row(k: int) -> np.ndarray:
    """Числа Стирлинга второго рода S(k, r), r = 0..k."""
    S = np.zeros((k + 1, k + 1))
    S[0, 0] = 1
    for i in range(1, k + 1):
        for r in range(1, i + 1):
            S[i, r] = r * S[i - 1, r] + S[i - 1, r - 1]
    return S[k]
