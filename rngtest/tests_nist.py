"""Батарея NIST SP 800-22 rev.1a — все 15 статистических тестов.

Реализация следует документу NIST SP 800-22 Rev. 1a (2010) и эталонному коду
sts-2.1.2. Корректность сверяется с контрольными значениями Приложения B
(двоичное разложение числа e, n = 10^6) — см. :mod:`rngtest.verify`.
"""
from __future__ import annotations

from math import floor, log, log2, sqrt

import numpy as np
from scipy import stats

from . import native
from .core import InsufficientData, register_test
from .mathutil import erfc, gf2_rank_batch, igamc, rank_probabilities, window_values

B = "NIST SP 800-22"


# 1 ------------------------------------------------------------------------
@register_test("nist.frequency", "Частотный (монобитный) тест", B, min_bits=100,
               description="Доля единиц в последовательности должна быть близка к 1/2. "
                           "Выявляет смещение (bias) генератора.")
def frequency(bits):
    n = bits.size
    s = 2 * int(bits.sum()) - n
    s_obs = abs(s) / sqrt(n)
    return {"statistic": s_obs, "p_values": [erfc(s_obs / sqrt(2))], "details": {"S_n": s}}


# 2 ------------------------------------------------------------------------
@register_test("nist.block_frequency", "Частотный тест в блоках", B, min_bits=100, M=128,
               description="Доля единиц в каждом M-битном блоке должна быть близка к 1/2. "
                           "Выявляет локальные смещения.")
def block_frequency(bits, M=128):
    n = bits.size
    N = n // M
    if N < 1:
        raise InsufficientData(f"нужно хотя бы M={M} бит")
    pi = bits[: N * M].reshape(N, M).mean(axis=1)
    chi2 = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return {"statistic": chi2, "p_values": [igamc(N / 2, chi2 / 2)], "details": {"N": N}}


# 3 ------------------------------------------------------------------------
@register_test("nist.runs", "Тест на серии (Runs)", B, min_bits=100,
               description="Число серий (максимальных блоков одинаковых бит) должно "
                           "соответствовать случайной последовательности. Выявляет слишком "
                           "быстрое или медленное чередование 0 и 1.")
def runs(bits):
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / sqrt(n):
        return {"statistic": float("nan"), "p_values": [0.0],
                "details": {"note": "не пройден предварительный частотный критерий"}}
    v = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    p = erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * sqrt(2 * n) * pi * (1 - pi)))
    return {"statistic": v, "p_values": [p]}


# 4 ------------------------------------------------------------------------
_LR_TABLE = [  # (min n, M, границы категорий v_lo..v_hi, вероятности)
    (750000, 10000, 10, 16, [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]),
    (6272, 128, 4, 9, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]),
    (128, 8, 1, 4, [0.2148, 0.3672, 0.2305, 0.1875]),
]


def _longest_runs(blocks_bits: np.ndarray, N: int, M: int) -> np.ndarray:
    if native.available():
        return native.longest_runs(blocks_bits, N, M)
    blk = blocks_bits[: N * M].reshape(N, M)
    cur = np.zeros(N, dtype=np.int32)
    best = np.zeros(N, dtype=np.int32)
    for j in range(M):
        cur = (cur + 1) * blk[:, j]
        np.maximum(best, cur, out=best)
    return best


@register_test("nist.longest_run", "Самая длинная серия единиц в блоке", B, min_bits=128,
               description="Распределение длины самой длинной серии единиц в M-битных блоках.")
def longest_run(bits):
    n = bits.size
    for min_n, M, lo, hi, pi in _LR_TABLE:
        if n >= min_n:
            break
    N = n // M
    longest = _longest_runs(bits, N, M)
    v = np.bincount(np.clip(longest, lo, hi) - lo, minlength=hi - lo + 1).astype(float)
    exp = N * np.array(pi)
    chi2 = float(((v - exp) ** 2 / exp).sum())
    K = len(pi) - 1
    return {"statistic": chi2, "p_values": [igamc(K / 2, chi2 / 2)],
            "details": {"M": M, "N": N, "counts": v.astype(int).tolist()}}


# 5 ------------------------------------------------------------------------
def matrix_rows(bits: np.ndarray, n_mat: int, M: int, Q: int) -> np.ndarray:
    """Нарезать биты на n_mat матриц M×Q, строки — целые числа."""
    if Q in (8, 16, 32, 64):  # строка = машинное слово: упаковка без промежуточных массивов
        w = np.packbits(bits[: n_mat * M * Q]).view(f">u{Q // 8}")
        return w.astype(np.uint64).reshape(n_mat, M)
    b = bits[: n_mat * M * Q].reshape(n_mat, M, Q).astype(np.uint64)
    w = np.uint64(1) << np.arange(Q - 1, -1, -1, dtype=np.uint64)
    return (b * w).sum(axis=2, dtype=np.uint64)


@register_test("nist.rank", "Тест рангов двоичных матриц", B, min_bits=38 * 1024,
               description="Ранги 32×32 матриц над GF(2). Выявляет линейные зависимости "
                           "между подстроками фиксированной длины.", complexity="O(n·32)")
def rank(bits, M=32, Q=32):
    N = bits.size // (M * Q)
    if N < 38:
        raise InsufficientData("нужно не менее 38 матриц")
    r = gf2_rank_batch(matrix_rows(bits, N, M, Q), Q)
    probs = rank_probabilities(M, Q)
    R = min(M, Q)
    p_full, p_m1 = probs[R], probs[R - 1]
    p_rest = 1 - p_full - p_m1
    f_full = int((r == R).sum())
    f_m1 = int((r == R - 1).sum())
    f_rest = N - f_full - f_m1
    chi2 = ((f_full - p_full * N) ** 2 / (p_full * N) + (f_m1 - p_m1 * N) ** 2 / (p_m1 * N)
            + (f_rest - p_rest * N) ** 2 / (p_rest * N))
    return {"statistic": chi2, "p_values": [float(np.exp(-chi2 / 2))],
            "details": {"N": N, "F_M": f_full, "F_M-1": f_m1, "rest": f_rest}}


# 6 ------------------------------------------------------------------------
@register_test("nist.dft", "Спектральный тест (ДПФ)", B, min_bits=1000,
               description="Пики дискретного преобразования Фурье. Выявляет периодические "
                           "структуры.", complexity="O(n log n)")
def dft(bits):
    n = bits.size
    x = 2.0 * bits - 1.0
    s = np.fft.rfft(x)[: n // 2]
    mod = np.abs(s)
    T = sqrt(log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = int((mod < T).sum())
    d = (N1 - N0) / sqrt(n * 0.95 * 0.05 / 4)
    return {"statistic": d, "p_values": [erfc(abs(d) / sqrt(2))], "details": {"N1": N1, "N0": N0}}


# 7 ------------------------------------------------------------------------
def aperiodic_templates(m: int) -> list:
    """Все апериодические шаблоны длины m (не перекрываются сами с собой), по возрастанию."""
    out = []
    for t in range(1 << m):
        s = format(t, f"0{m}b")
        if all(s[:k] != s[m - k:] for k in range(1, m)):
            out.append(t)
    return out


@register_test("nist.non_overlapping_template", "Тест неперекрывающихся шаблонов", B,
               min_bits=8 * 512, m=9, N=8,
               description="Частота появления каждого из апериодических m-битных шаблонов "
                           "(148 шаблонов при m=9) в N блоках. Выявляет избыток/недостаток "
                           "конкретных паттернов.")
def non_overlapping_template(bits, m=9, N=8):
    n = bits.size
    M = n // N
    if M < 2 ** m:
        raise InsufficientData("слишком короткие блоки")
    templates = np.array(aperiodic_templates(m))
    mu = (M - m + 1) / 2 ** m
    var = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    W = np.empty((N, 1 << m))
    for j in range(N):
        # шаблоны апериодичны => совпадения не перекрываются, можно считать все окна сразу
        W[j] = np.bincount(window_values(bits[j * M:(j + 1) * M], m), minlength=1 << m)
    W = W[:, templates]
    chi2 = ((W - mu) ** 2 / var).sum(axis=0)
    ps = [igamc(N / 2, c / 2) for c in chi2]
    return {"statistic": float(chi2[0]), "p_values": ps,
            "details": {"templates": len(ps), "first_template": format(int(templates[0]), f"0{m}b"),
                        "p_first_template": ps[0]}}


# 8 ------------------------------------------------------------------------
_OT_PI = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
# старые (неточные) вероятности, по которым посчитаны значения Приложения B SP 800-22
_OT_PI_LEGACY = [0.367879, 0.183940, 0.137955, 0.099634, 0.069935, 0.140657]


@register_test("nist.overlapping_template", "Тест перекрывающихся шаблонов", B,
               min_bits=1032 * 50, m=9,
               description="Число перекрывающихся вхождений шаблона из m единиц в блоках "
                           "по 1032 бита.")
def overlapping_template(bits, m=9, M=1032, K=5, legacy=False):
    N = bits.size // M
    if N < 1:
        raise InsufficientData("короткая последовательность")
    target = (1 << m) - 1
    blk = bits[: N * M].reshape(N, M)
    vals = np.zeros((N, M - m + 1), dtype=np.int64)
    for j in range(m):
        vals = (vals << 1) | blk[:, j: j + M - m + 1]
    counts = np.minimum((vals == target).sum(axis=1), K)
    v = np.bincount(counts, minlength=K + 1).astype(float)
    exp = N * np.array(_OT_PI_LEGACY if legacy else _OT_PI)
    chi2 = float(((v - exp) ** 2 / exp).sum())
    return {"statistic": chi2, "p_values": [igamc(K / 2, chi2 / 2)],
            "details": {"N": N, "counts": v.astype(int).tolist()}}


# 9 ------------------------------------------------------------------------
_UNIV = {1: (0.7326495, 0.690), 2: (1.5374383, 1.338), 3: (2.4016068, 1.901),
         4: (3.3112247, 2.358), 5: (4.2534266, 2.705), 6: (5.2177052, 2.954),
         7: (6.1962507, 3.125), 8: (7.1836656, 3.238), 9: (8.1764248, 3.311),
         10: (9.1723243, 3.356), 11: (10.170032, 3.384), 12: (11.168765, 3.401),
         13: (12.168070, 3.410), 14: (13.167693, 3.416), 15: (14.167488, 3.419),
         16: (15.167379, 3.421)}


@register_test("nist.universal", "Универсальный тест Маурера", B, min_bits=8080,
               description="Сжимаемость последовательности: расстояния между повторами "
                           "L-битных блоков. Сжимаемая последовательность неслучайна.")
def universal(bits, L=None):
    n = bits.size
    if L is None:  # порог NIST: n >= 1010 * 2^L * L
        L = max([l for l in range(2, 17) if n >= 1010 * (1 << l) * l] or [0])
        if L == 0:
            raise InsufficientData("нужно не менее 8080 бит")
    Q = 10 * (1 << L)
    K = n // L - Q
    if K <= 0:
        raise InsufficientData("короткая последовательность")
    vals = window_values(bits, L)[::L][: Q + K]
    idx = np.arange(1, Q + K + 1)  # 1-индексация как в NIST
    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    prev = np.zeros(Q + K, dtype=np.int64)
    same = np.r_[False, sv[1:] == sv[:-1]]
    prev_sorted = np.where(same, np.r_[0, idx[order][:-1]], 0)
    prev[order] = prev_sorted
    dist = idx[Q:] - prev[Q:]
    fn = float(np.log2(dist).sum() / K)
    ev, var = _UNIV[L]
    c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15
    sigma = c * sqrt(var / K)
    p = erfc(abs((fn - ev) / (sqrt(2) * sigma)))
    return {"statistic": fn, "p_values": [p], "details": {"L": L, "Q": Q, "K": K}}


# 10 -----------------------------------------------------------------------
def berlekamp_massey_batch(S: np.ndarray) -> np.ndarray:
    """Линейная сложность для каждой строки S (N×M), векторизовано по блокам."""
    N, M = S.shape
    S = S.astype(np.uint8)
    C = np.zeros((N, M + 1), dtype=np.uint8)
    Bp = np.zeros((N, M + 1), dtype=np.uint8)
    C[:, 0] = Bp[:, 0] = 1
    L = np.zeros(N, dtype=np.int64)
    m = -np.ones(N, dtype=np.int64)
    cols = np.arange(M + 1)
    for i in range(M):
        if i == 0:
            d = S[:, 0].copy()
        else:
            d = (S[:, i] + np.einsum("ij,ij->i", C[:, 1:i + 1], S[:, i - 1::-1][:, :i],
                                     dtype=np.int64)) & 1
        idx = np.nonzero(d)[0]
        if idx.size == 0:
            continue
        T = C[idx].copy()
        src = cols[None, :] - (i - m[idx])[:, None]
        Bs = np.where(src >= 0, np.take_along_axis(Bp[idx], np.clip(src, 0, M), axis=1), 0)
        C[idx] ^= Bs.astype(np.uint8)
        upd = L[idx] <= i // 2
        rows = idx[upd]
        L[rows] = i + 1 - L[rows]
        m[rows] = i
        Bp[rows] = T[upd]
    return L


def linear_complexity_blocks(bits: np.ndarray, N: int, M: int) -> np.ndarray:
    if native.available():
        return native.lc_blocks(bits, N, M).astype(np.int64)
    return berlekamp_massey_batch(bits[: N * M].reshape(N, M))


# значения как в эталонном коде sts-2.1.2 (в тексте стандарта π0 = 0.010417)
_LC_PI = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]


@register_test("nist.linear_complexity", "Тест линейной сложности", B, min_bits=200 * 500,
               M=500, complexity="O(n·M)",
               description="Длина минимального РСЛОС, порождающего блок (алгоритм "
                           "Берлекэмпа–Мэсси). Выявляет линейные генераторы (LFSR).")
def linear_complexity(bits, M=500):
    N = bits.size // M
    if N < 200:
        raise InsufficientData(f"нужно не менее 200 блоков по {M} бит")
    L = linear_complexity_blocks(bits, N, M)
    mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    T = (-1) ** M * (L - mu) + 2 / 9
    v = np.histogram(T, bins=[-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf])[0].astype(float)
    exp = N * np.array(_LC_PI)
    chi2 = float(((v - exp) ** 2 / exp).sum())
    return {"statistic": chi2, "p_values": [igamc(3, chi2 / 2)],
            "details": {"N": N, "counts": v.astype(int).tolist(),
                        "backend": "C" if native.available() else "NumPy"}}


# 11 -----------------------------------------------------------------------
def _psi2(bits, m):
    if m <= 0:
        return 0.0
    n = bits.size
    c = np.bincount(window_values(bits, m, cyclic=True), minlength=1 << m).astype(float)
    return (1 << m) / n * float((c * c).sum()) - n


@register_test("nist.serial", "Последовательный тест (Serial)", B, min_bits=64,
               description="Равномерность распределения всех перекрывающихся m-битных "
                           "шаблонов. Даёт 2 p-значения.")
def serial(bits, m=16):
    n = bits.size
    m = min(m, int(floor(log2(n))) - 3)
    if m < 2:
        raise InsufficientData("короткая последовательность")
    p0, p1, p2 = _psi2(bits, m), _psi2(bits, m - 1), _psi2(bits, m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    return {"statistic": d1,
            "p_values": [igamc(2 ** (m - 2), d1 / 2), igamc(2 ** (m - 3), d2 / 2)],
            "details": {"m": m, "del_psi2": d1, "del2_psi2": d2}}


# 12 -----------------------------------------------------------------------
def _phi(bits, m):
    if m == 0:
        return 0.0
    n = bits.size
    c = np.bincount(window_values(bits, m, cyclic=True), minlength=1 << m)
    c = c[c > 0] / n
    return float((c * np.log(c)).sum())


@register_test("nist.approximate_entropy", "Тест приближённой энтропии", B, min_bits=64,
               description="Сравнивает частоты перекрывающихся блоков длины m и m+1.")
def approximate_entropy(bits, m=10):
    n = bits.size
    m = min(m, int(floor(log2(n))) - 6)
    if m < 1:
        raise InsufficientData("короткая последовательность")
    apen = _phi(bits, m) - _phi(bits, m + 1)
    chi2 = 2 * n * (log(2) - apen)
    return {"statistic": chi2, "p_values": [igamc(2 ** (m - 1), chi2 / 2)],
            "details": {"m": m, "ApEn": apen}}


# 13 -----------------------------------------------------------------------
def _cusum_p(n, z):
    Phi = stats.norm.cdf
    sq = sqrt(n)
    k1 = np.arange(int((-n / z + 1) / 4), floor((n / z - 1) / 4) + 1)
    k2 = np.arange(int((-n / z - 3) / 4), floor((n / z - 1) / 4) + 1)
    s1 = np.sum(Phi((4 * k1 + 1) * z / sq) - Phi((4 * k1 - 1) * z / sq))
    s2 = np.sum(Phi((4 * k2 + 3) * z / sq) - Phi((4 * k2 + 1) * z / sq))
    return float(min(max(1 - s1 + s2, 0.0), 1.0))


@register_test("nist.cumulative_sums", "Тест кумулятивных сумм", B, min_bits=100,
               description="Максимальное отклонение случайного блуждания (±1) от нуля, "
                           "в прямом и обратном направлении. Даёт 2 p-значения.")
def cumulative_sums(bits):
    n = bits.size
    x = 2 * bits.astype(np.int64) - 1
    z_f = int(np.abs(np.cumsum(x)).max())
    z_b = int(np.abs(np.cumsum(x[::-1])).max())
    return {"statistic": z_f, "p_values": [_cusum_p(n, z_f), _cusum_p(n, z_b)],
            "details": {"z_forward": z_f, "z_backward": z_b}}


# 14, 15 -------------------------------------------------------------------
def _excursions(bits):
    S = np.cumsum(2 * bits.astype(np.int64) - 1)
    zeros = S == 0
    J = int(zeros.sum()) + (1 if S[-1] != 0 else 0)
    # номер цикла для позиции k = число нулей среди S_1..S_{k-1}
    cid = np.concatenate([[0], np.cumsum(zeros)[:-1]])
    return S, J, cid


def _ex_pi(x):
    a = 1 / (2 * abs(x))
    p = [1 - a] + [1 / (4 * x * x) * (1 - a) ** (k - 1) for k in range(1, 5)] + [a * (1 - a) ** 4]
    return np.array(p)


@register_test("nist.random_excursions", "Тест случайных отклонений", B, min_bits=10 ** 6,
               description="Число посещений состояний x∈{±1..±4} в циклах случайного "
                           "блуждания. Даёт 8 p-значений. Неприменим, если циклов < 500.")
def random_excursions(bits):
    n = bits.size
    S, J, cid = _excursions(bits)
    if J < max(0.005 * sqrt(n), 500):
        raise InsufficientData(f"число циклов J={J} < 500 — тест неприменим")
    ps, chis = [], []
    for x in (-4, -3, -2, -1, 1, 2, 3, 4):
        visits = np.bincount(cid[S == x], minlength=J)[:J]
        v = np.bincount(np.minimum(visits, 5), minlength=6).astype(float)
        exp = J * _ex_pi(x)
        chi2 = float(((v - exp) ** 2 / exp).sum())
        chis.append(chi2)
        ps.append(igamc(2.5, chi2 / 2))
    return {"statistic": chis[4], "p_values": ps,
            "details": {"J": J, "states": [-4, -3, -2, -1, 1, 2, 3, 4], "p_x=+1": ps[4]}}


@register_test("nist.random_excursions_variant", "Тест случайных отклонений (вариант)", B,
               min_bits=10 ** 6,
               description="Общее число посещений состояний x∈{±1..±9}. Даёт 18 p-значений.")
def random_excursions_variant(bits):
    n = bits.size
    S, J, _ = _excursions(bits)
    if J < max(0.005 * sqrt(n), 500):
        raise InsufficientData(f"число циклов J={J} < 500 — тест неприменим")
    states = [x for x in range(-9, 10) if x != 0]
    cnt = np.bincount(S[np.abs(S) <= 9] + 9, minlength=19)
    ps = [erfc(abs(int(cnt[x + 9]) - J) / sqrt(2 * J * (4 * abs(x) - 2))) for x in states]
    return {"statistic": float(cnt[-1 + 9]), "p_values": ps,
            "details": {"J": J, "states": states, "p_x=-1": ps[8]}}
