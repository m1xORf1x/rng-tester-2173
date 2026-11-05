"""Батарея по мотивам TestU01 SmallCrush (P. L'Ecuyer, R. Simard, 2007).

Набор повторяет состав SmallCrush (10 тестов) с параметрами, масштабируемыми под
длину входной последовательности. Числа U ∈ [0,1) получаются из 32-битных слов.
HammingIndep, RandomWalk1 и MatrixRank работают непосредственно с битами.
"""
from __future__ import annotations

from math import comb, factorial, sqrt

import numpy as np
from scipy import stats

from .core import InsufficientData, register_test
from .mathutil import (bits_to_uniforms, chi2_test, gf2_rank_batch, rank_probabilities,
                       stirling2_row)
from .tests_nist import matrix_rows

B = "TestU01 SmallCrush"


def _poisson_two_sided(y, lam):
    return float(min(1.0, 2 * min(stats.poisson.sf(y - 1, lam), stats.poisson.cdf(y, lam))))


def _uniforms(bits, need):
    u = bits_to_uniforms(bits)
    if u.size < need:
        raise InsufficientData(f"нужно не менее {need} чисел (×32 бита)")
    return u


@register_test("testu01.birthday_spacings", "Интервалы дней рождения, t=2 (BirthdaySpacings)", B,
               min_bits=10 * 1024 * 32,
               description="Двумерные 'дни рождения': точки (U1,U2) в сетке 2^12×2^12. Суммарное "
                           "число совпадающих интервалов ~ Пуассон.")
def birthday_spacings(bits, n=512, d=4096):
    u = _uniforms(bits, 10 * 2 * n)
    R = u.size // (2 * n)
    pts = np.floor(u[: R * 2 * n] * d).astype(np.int64).reshape(R, n, 2)
    cells = np.sort(pts[:, :, 0] * d + pts[:, :, 1], axis=1)
    sp = np.sort(np.diff(cells, axis=1, prepend=0), axis=1)
    y = int((sp[:, 1:] == sp[:, :-1]).sum())
    lam = R * n ** 3 / (4 * d * d)
    return {"statistic": y, "p_values": [_poisson_two_sided(y, lam)],
            "details": {"replications": R, "expected": lam}}


@register_test("testu01.collision", "Тест коллизий (Collision)", B, min_bits=4096 * 32,
               description="n точек размещаются в k=2^20 ячеек (t=2, d=2^10). Число коллизий "
                           "сравнивается с ожидаемым (Пуассон).")
def collision(bits, d=1024):
    u = _uniforms(bits, 4096)
    k = d * d
    n = min(u.size // 2, 1 << 14)
    R = min(u.size // (2 * n), 100)
    total, lam = 0, 0.0
    for r in range(R):
        x = np.floor(u[r * 2 * n:(r + 1) * 2 * n] * d).astype(np.int64)
        cells = x[0::2] * d + x[1::2]
        total += n - np.unique(cells).size
        lam += n - k + k * (1 - 1 / k) ** n
    return {"statistic": total, "p_values": [_poisson_two_sided(total, lam)],
            "details": {"n": n, "replications": R, "expected": lam}}


@register_test("testu01.gap", "Тест интервалов (Gap)", B, min_bits=2000 * 32,
               description="Длины промежутков между попаданиями U в интервал [0, 1/8) — "
                           "геометрическое распределение.")
def gap(bits, alpha=0.0, beta=0.125):
    u = _uniforms(bits, 2000)
    p = beta - alpha
    hits = np.flatnonzero((u >= alpha) & (u < beta))
    gaps = np.diff(hits) - 1
    t = int(np.ceil(np.log(5 / max(gaps.size, 1)) / np.log(1 - p))) if gaps.size else 1
    t = max(min(t, 200), 2)
    obs = np.bincount(np.minimum(gaps, t), minlength=t + 1)
    exp = gaps.size * np.r_[p * (1 - p) ** np.arange(t), (1 - p) ** t]
    chi2, df, pv = chi2_test(obs, exp)
    return {"statistic": chi2, "p_values": [pv], "details": {"gaps": int(gaps.size), "df": df}}


@register_test("testu01.simple_poker", "Покер-тест Кнута (SimpPoker)", B, min_bits=2000 * 32,
               description="Покер-тест Кнута: группы по k=8 чисел из d=8 значений; число "
                           "различных значений в группе.")
def simple_poker(bits, d=8, k=8):
    u = _uniforms(bits, 2000)
    G = u.size // k
    v = np.floor(u[: G * k] * d).astype(np.int64).reshape(G, k)
    sv = np.sort(v, axis=1)
    distinct = 1 + (sv[:, 1:] != sv[:, :-1]).sum(axis=1)
    S = stirling2_row(k)
    probs = np.array([np.prod(np.arange(d - r + 1, d + 1, dtype=float)) * S[r] / d ** k
                      for r in range(1, min(k, d) + 1)])
    obs = np.bincount(distinct, minlength=min(k, d) + 1)[1:]
    chi2, df, pv = chi2_test(obs, G * probs)
    return {"statistic": chi2, "p_values": [pv], "details": {"groups": G}}


@register_test("testu01.coupon_collector", "Собиратель купонов (CouponCollector)", B, min_bits=4000 * 32,
               description="Длина отрезка, необходимого для сбора всех d=8 'купонов'.")
def coupon_collector(bits, d=8, t=40):
    u = _uniforms(bits, 4000)
    v = np.floor(u * d).astype(np.int64).tolist()
    lengths, seen, cnt, L = [], 0, 0, 0
    for x in v:
        L += 1
        if not seen >> x & 1:
            seen |= 1 << x
            cnt += 1
            if cnt == d:
                lengths.append(L)
                seen = cnt = L = 0
    lengths = np.array(lengths, dtype=np.int64)
    if lengths.size < 10:  # почти никогда не встречаются все d значений — явный дефект
        return {"statistic": float(lengths.size), "p_values": [0.0],
                "details": {"segments": int(lengths.size), "note": "не собраны все купоны"}}
    S = [stirling2_row(s - 1)[d - 1] if s - 1 >= d - 1 else 0 for s in range(d, t)]
    probs = np.array([factorial(d) / d ** s * S[s - d] for s in range(d, t)])
    probs = np.r_[probs, 1 - probs.sum()]
    obs = np.bincount(np.clip(lengths, d, t) - d, minlength=t - d + 1)
    chi2, df, pv = chi2_test(obs, lengths.size * probs)
    return {"statistic": chi2, "p_values": [pv], "details": {"segments": int(lengths.size)}}


@register_test("testu01.max_of_t", "Максимум из t (MaxOft)", B, min_bits=3000 * 32,
               description="Максимум из t=6 чисел: величина max^t равномерна. Критерии χ² и "
                           "Колмогорова–Смирнова. Даёт 2 p-значения.")
def max_of_t(bits, t=6):
    u = _uniforms(bits, 3000)
    G = u.size // t
    m = u[: G * t].reshape(G, t).max(axis=1) ** t
    bins = int(min(max(G // 50, 5), 100))
    obs = np.histogram(m, bins=bins, range=(0, 1))[0]
    chi2, df, p1 = chi2_test(obs, np.full(bins, G / bins))
    p2 = float(stats.kstest(m, "uniform").pvalue)
    return {"statistic": chi2, "p_values": [p1, p2], "details": {"groups": G}}


@register_test("testu01.weight_distrib", "Распределение весов (WeightDistrib)", B, min_bits=6400 * 32,
               description="Число U < 1/4 в группах по k=64 — биномиальное распределение.")
def weight_distrib(bits, k=64, beta=0.25):
    u = _uniforms(bits, 6400)
    G = u.size // k
    w = (u[: G * k].reshape(G, k) < beta).sum(axis=1)
    obs = np.bincount(w, minlength=k + 1)
    exp = G * stats.binom.pmf(np.arange(k + 1), k, beta)
    chi2, df, pv = chi2_test(obs, exp)
    return {"statistic": chi2, "p_values": [pv], "details": {"groups": G}}


@register_test("testu01.matrix_rank", "Ранги матриц 60×60 (MatrixRank)", B, min_bits=100 * 3600,
               description="Ранги 60×60 двоичных матриц над GF(2).")
def matrix_rank(bits, L=60):
    N = bits.size // (L * L)
    if N < 100:
        raise InsufficientData("нужно не менее 100 матриц")
    r = gf2_rank_batch(matrix_rows(bits, N, L, L), L)
    probs = rank_probabilities(L, L)
    obs = [int((r <= L - 2).sum()), int((r == L - 1).sum()), int((r == L).sum())]
    exp = N * np.array([probs[: L - 1].sum(), probs[L - 1], probs[L]])
    chi2, df, pv = chi2_test(obs, exp)
    return {"statistic": chi2, "p_values": [pv], "details": {"matrices": N}}


_HW_EDGES = [14, 16, 17, 19]  # классы весов 32-битного блока: ≤13, 14–15, 16, 17–18, ≥19


@register_test("testu01.hamming_indep", "Независимость весов Хэмминга (HammingIndep)", B, min_bits=5000 * 64,
               description="Независимость весов Хэмминга соседних 32-битных блоков "
                           "(таблица сопряжённости 5×5, χ²).")
def hamming_indep(bits, L=32):
    N = bits.size // (2 * L)
    if N < 5000:
        raise InsufficientData("мало данных")
    w = bits[: N * 2 * L].reshape(N, 2, L).sum(axis=2)
    cls = np.digitize(w, _HW_EDGES)
    pmf = stats.binom.pmf(np.arange(L + 1), L, 0.5)
    cp = np.array([pmf[: 14].sum(), pmf[14:16].sum(), pmf[16], pmf[17:19].sum(), pmf[19:].sum()])
    table = np.zeros((5, 5))
    np.add.at(table, (cls[:, 0], cls[:, 1]), 1)
    exp = N * np.outer(cp, cp)
    chi2 = float(((table - exp) ** 2 / exp).sum())
    return {"statistic": chi2, "p_values": [float(stats.chi2.sf(chi2, 24))], "details": {"pairs": N}}


@register_test("testu01.random_walk", "Случайное блуждание (RandomWalk1)", B, min_bits=1000 * 64,
               description="Блуждания длины L=64 по битам: распределение конечного положения H "
                           "и максимума M (критерий χ²). Даёт 2 p-значения.")
def random_walk(bits, L=64):
    N = bits.size // L
    if N < 1000:
        raise InsufficientData("мало данных")
    steps = 2 * bits[: N * L].reshape(N, L).astype(np.int64) - 1
    S = np.cumsum(steps, axis=1)
    H = S[:, -1]
    k = (H + L) // 2
    obs_h = np.bincount(k, minlength=L + 1)
    pk = np.array([comb(L, i) for i in range(L + 1)], dtype=float) / 2.0 ** L
    _, _, p1 = chi2_test(obs_h, N * pk)
    Mx = np.maximum(S.max(axis=1), 0)
    # принцип отражения: P(M >= m) = P(S_L >= m) + P(S_L > m)
    pos = np.arange(-L, L + 1, 2)
    ps_l = dict(zip(pos, pk))
    tail = lambda m: sum(v for x, v in ps_l.items() if x >= m) + sum(v for x, v in ps_l.items() if x > m)
    pm = np.array([1 - tail(1)] + [tail(m) - tail(m + 1) for m in range(1, L + 1)])
    obs_m = np.bincount(Mx, minlength=L + 1)
    _, _, p2 = chi2_test(obs_m, N * pm)
    return {"statistic": float(H.mean()), "p_values": [p1, p2], "details": {"walks": N}}
