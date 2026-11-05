"""Дополнительные батареи:

* **FIPS 140-2** — четыре "быстрых" теста на 20 000 бит, исторически применявшиеся
  для самотестирования криптомодулей при включении. Решение принимается по
  фиксированным границам стандарта (а не по α), p-значения приводятся справочно.
* **Базовые тесты Menezes (HAC, гл. 5.4.4)** — частотный, двухбитовый
  (serial), покер, серий и автокорреляционный тесты. Автокорреляционный тест
  имеет оптимизированную C-реализацию на упакованных битах.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats

from . import native
from .core import InsufficientData, register_test
from .mathutil import erfc, normal_two_sided, window_values

FIPS = "FIPS 140-2"
HAC = "Menezes (HAC)"
FIPS_N = 20000


def _run_lengths(bits):
    """Длины серий и значения бит, которыми они образованы."""
    edges = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    starts = np.r_[0, edges]
    lens = np.diff(np.r_[starts, bits.size])
    return lens, bits[starts]


# ----------------------------------------------------------------- FIPS 140-2
@register_test("fips.monobit", "FIPS: монобитный тест", FIPS, min_bits=FIPS_N,
               description="Число единиц в 20 000 бит должно лежать в (9725, 10275).")
def fips_monobit(bits):
    x = bits[:FIPS_N]
    ones = int(x.sum())
    return {"statistic": ones, "passed": 9725 < ones < 10275,
            "p_values": [erfc(abs(2 * ones - FIPS_N) / sqrt(FIPS_N) / sqrt(2))],
            "details": {"bounds": "(9725, 10275)"}}


@register_test("fips.poker", "FIPS: покер-тест", FIPS, min_bits=FIPS_N,
               description="5000 4-битных блоков; статистика X должна лежать в (2.16, 46.17).")
def fips_poker(bits):
    x = bits[:FIPS_N].reshape(-1, 4)
    v = x[:, 0] * 8 + x[:, 1] * 4 + x[:, 2] * 2 + x[:, 3]
    f = np.bincount(v, minlength=16)
    X = 16 / 5000 * float((f.astype(float) ** 2).sum()) - 5000
    return {"statistic": X, "passed": 2.16 < X < 46.17,
            "p_values": [float(stats.chi2.sf(X, 15))], "details": {"bounds": "(2.16, 46.17)"}}


_FIPS_RUNS = [(2315, 2685), (1114, 1386), (527, 723), (240, 384), (103, 209), (103, 209)]


@register_test("fips.runs", "FIPS: тест серий", FIPS, min_bits=FIPS_N,
               description="Число серий длины 1..5 и ≥6 (отдельно для 0 и 1) в допустимых "
                           "интервалах.")
def fips_runs(bits):
    lens, vals = _run_lengths(bits[:FIPS_N])
    ok = True
    obs, exp = [], []
    counts = {}
    for b in (0, 1):
        c = np.bincount(np.minimum(lens[vals == b], 6), minlength=7)[1:]
        counts[str(b)] = c.tolist()
        for i, (lo, hi) in enumerate(_FIPS_RUNS):
            ok &= lo <= c[i] <= hi
        obs.extend(c)
        e = [(FIPS_N - i + 3) / 2 ** (i + 2) for i in range(1, 6)]
        e.append(FIPS_N / 2 ** 7)  # ≈ Σ_{i≥6} n / 2^(i+2)
        exp.extend(e)
    obs, exp = np.array(obs, float), np.array(exp)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    return {"statistic": chi2, "passed": bool(ok), "p_values": [float(stats.chi2.sf(chi2, 10))],
            "details": {"runs": counts}}


@register_test("fips.long_run", "FIPS: тест длинной серии", FIPS, min_bits=FIPS_N,
               description="В 20 000 бит не должно быть серии длины ≥ 26.")
def fips_long_run(bits):
    lens, _ = _run_lengths(bits[:FIPS_N])
    longest = int(lens.max())
    # P(max run >= L) ≈ 1 - exp(-n / 2^L) (число серий длины >= L ~ Пуассон)
    p = float(-np.expm1(-FIPS_N / 2 ** longest))
    return {"statistic": longest, "passed": longest < 26, "p_values": [min(p, 1.0)]}


# ----------------------------------------------------------------- HAC (Menezes)
@register_test("hac.frequency", "HAC: частотный тест", HAC, min_bits=100,
               description="X1 = (n0 − n1)² / n ~ χ²(1).")
def hac_frequency(bits):
    n = bits.size
    n1 = int(bits.sum())
    X = (n - 2 * n1) ** 2 / n
    return {"statistic": X, "p_values": [float(stats.chi2.sf(X, 1))]}


@register_test("hac.serial", "HAC: двухбитовый тест (serial)", HAC, min_bits=100,
               description="Частоты пар 00, 01, 10, 11 (перекрывающихся). X2 ~ χ²(2).")
def hac_serial(bits):
    n = bits.size
    n1 = int(bits.sum())
    n0 = n - n1
    c = np.bincount(window_values(bits, 2), minlength=4).astype(float)
    X = 4 / (n - 1) * float((c ** 2).sum()) - 2 / n * (n0 ** 2 + n1 ** 2) + 1
    return {"statistic": X, "p_values": [float(stats.chi2.sf(X, 2))]}


@register_test("hac.poker", "HAC: покер-тест", HAC, min_bits=1000,
               description="Неперекрывающиеся m-битные блоки (m выбирается из условия "
                           "k ≥ 5·2^m). X3 ~ χ²(2^m − 1).")
def hac_poker(bits, m=None):
    n = bits.size
    if m is None:
        m = max(mm for mm in range(1, 17) if n // mm >= 5 * 2 ** mm)
        m = min(m, 8)
    k = n // m
    w = 1 << np.arange(m - 1, -1, -1)
    v = (bits[: k * m].reshape(k, m) * w).sum(axis=1)
    c = np.bincount(v, minlength=1 << m).astype(float)
    X = (1 << m) / k * float((c ** 2).sum()) - k
    return {"statistic": X, "p_values": [float(stats.chi2.sf(X, (1 << m) - 1))],
            "details": {"m": m}}


@register_test("hac.runs", "HAC: тест серий (блоки и пропуски)", HAC, min_bits=1000,
               description="Число блоков (серий 1) и пропусков (серий 0) длины i = 1..k. "
                           "X4 ~ χ²(2k − 2).")
def hac_runs(bits):
    n = bits.size
    lens, vals = _run_lengths(bits)
    e = lambda i: (n - i + 3) / 2 ** (i + 2)
    k = max(i for i in range(1, 40) if e(i) >= 5)
    B = np.bincount(np.minimum(lens[vals == 1], k + 1), minlength=k + 2)[1:k + 1]
    G = np.bincount(np.minimum(lens[vals == 0], k + 1), minlength=k + 2)[1:k + 1]
    E = np.array([e(i) for i in range(1, k + 1)])
    X = float(((B - E) ** 2 / E).sum() + ((G - E) ** 2 / E).sum())
    return {"statistic": X, "p_values": [float(stats.chi2.sf(X, 2 * k - 2))], "details": {"k": k}}


def autocorr_count(bits: np.ndarray, d: int, packed: np.ndarray = None) -> int:
    """A(d) = Σ s_i ⊕ s_{i+d}. C-версия на упакованных битах, иначе NumPy."""
    if packed is not None and native.available():
        return native.packed_autocorr(packed, bits.size, d)
    return int(np.count_nonzero(bits[:-d] != bits[d:]))


@register_test("hac.autocorrelation", "Автокорреляционный тест", HAC, min_bits=1000,
               lags=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32),
               description="Корреляция последовательности со своим сдвигом на d бит: "
                           "Z = 2(A(d) − (n−d)/2)/√(n−d) ~ N(0,1). Проверяется набор сдвигов d.")
def autocorrelation(bits, lags=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32)):
    n = bits.size
    packed = native.pack(bits) if native.available() else None
    ps, zs = [], []
    for d in lags:
        if d >= n // 2:
            continue
        A = autocorr_count(bits, d, packed)
        z = 2 * (A - (n - d) / 2) / sqrt(n - d)
        zs.append(z)
        ps.append(normal_two_sided(z))
    if not ps:
        raise InsufficientData("короткая последовательность")
    worst = int(np.argmin(ps))
    return {"statistic": float(zs[worst]), "p_values": ps,
            "details": {"lags": list(lags)[: len(ps)], "z": [round(z, 4) for z in zs]}}
