"""Батарея DieHard (G. Marsaglia, 1995) — реализация на Python/NumPy.

Классический DieHard рассчитан на файл ~80 Мбит и 32-битные слова. Здесь тесты
реализованы по оригинальным описаниям, а число повторений автоматически
подстраивается под длину последовательности (не меньше заданного минимума).
Тесты, требующие десятков мегабит при любом разумном числе повторений
(OPSO/OQSO/DNA, OPERM5, Squeeze, Overlapping sums), не включены; для полной
сверки предусмотрен адаптер к внешней утилите ``dieharder`` (см. verify.py).
"""
from __future__ import annotations

from math import comb, sqrt

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from .core import InsufficientData, register_test
from .mathutil import (bits_to_uniforms, bits_to_words, chi2_test, ks_uniform,
                       normal_two_sided, rank_probabilities, gf2_rank_batch)
from .tests_nist import matrix_rows

B = "DieHard"


# ------------------------------------------------------------------ дни рождения
def _birthday_dups(days: np.ndarray, n: int) -> np.ndarray:
    """Число повторяющихся значений среди отсортированных интервалов для каждой выборки."""
    d = np.sort(days.reshape(-1, n), axis=1)
    sp = np.sort(np.diff(d, axis=1, prepend=0), axis=1)
    return (sp[:, 1:] == sp[:, :-1]).sum(axis=1)


@register_test("diehard.birthday_spacings", "Интервалы между днями рождения", B,
               min_bits=20 * 512 * 32,
               description="n=512 'дней рождения' в году из 2^24 дней (24 бита слова). Число "
                           "совпадающих интервалов ~ Пуассон(λ=2). Проверяются старшие и "
                           "младшие 24 бита слов.")
def birthday_spacings(bits, n=512):
    w = bits_to_words(bits, 32)
    k = min(w.size // n, 500)
    if k < 20:
        raise InsufficientData("нужно не менее 20 выборок по 512 слов")
    lam = n ** 3 / (4 * 2 ** 24)
    ps = []
    for shift in (8, 0):  # старшие 24 бита, младшие 24 бита
        days = ((w[: k * n] >> np.uint64(shift)) & np.uint64(0xFFFFFF)).astype(np.int64)
        dups = _birthday_dups(days, n)
        maxj = 10
        obs = np.bincount(np.minimum(dups, maxj), minlength=maxj + 1)
        exp = k * np.r_[stats.poisson.pmf(np.arange(maxj), lam), stats.poisson.sf(maxj - 1, lam)]
        ps.append(chi2_test(obs, exp)[2])
    return {"statistic": float(lam), "p_values": ps, "details": {"samples": k, "lambda": lam}}


# ------------------------------------------------------------------ ранги матриц
def _rank_test(rows, ncols, M, Q, low_merge):
    r = gf2_rank_batch(rows, ncols)
    N = r.size
    probs = rank_probabilities(M, Q)
    R = min(M, Q)
    cats = list(range(low_merge, R + 1))
    obs = [int((r <= low_merge).sum())] + [int((r == c).sum()) for c in cats[1:]]
    exp = [N * probs[: low_merge + 1].sum()] + [N * probs[c] for c in cats[1:]]
    chi2, df, p = chi2_test(obs, exp)
    return chi2, p, dict(zip([f"≤{low_merge}"] + [str(c) for c in cats[1:]], obs))


@register_test("diehard.rank_32x32", "Ранги матриц 32×32", B, min_bits=100 * 1024,
               description="Ранг над GF(2) матриц из 32 подряд идущих 32-битных слов.")
def rank_32x32(bits):
    N = min(bits.size // 1024, 40000)
    chi2, p, d = _rank_test(matrix_rows(bits, N, 32, 32), 32, 32, 32, 29)
    return {"statistic": chi2, "p_values": [p], "details": {"matrices": N, "ranks": d}}


@register_test("diehard.rank_31x31", "Ранги матриц 31×31", B, min_bits=100 * 1024,
               description="Ранг матриц из старших 31 бита 31 подряд идущего слова.")
def rank_31x31(bits):
    w = bits_to_words(bits, 32)
    N = min(w.size // 31, 40000)
    if N < 100:
        raise InsufficientData("мало данных")
    rows = (w[: N * 31] >> np.uint64(1)).reshape(N, 31)
    chi2, p, d = _rank_test(rows, 31, 31, 31, 28)
    return {"statistic": chi2, "p_values": [p], "details": {"matrices": N, "ranks": d}}


@register_test("diehard.rank_6x8", "Ранги матриц 6×8", B, min_bits=6 * 32 * 500,
               description="Из каждого из 6 слов берётся один байт; ранг 6×8 матрицы. "
                           "Проверяются все 4 байтовые позиции слова.")
def rank_6x8(bits):
    w = bits_to_words(bits, 32)
    N = min(w.size // 6, 100000)
    ps, chis = [], []
    for shift in (24, 16, 8, 0):
        rows = ((w[: N * 6] >> np.uint64(shift)) & np.uint64(0xFF)).reshape(N, 6)
        chi2, p, _ = _rank_test(rows, 8, 6, 8, 4)
        ps.append(p)
        chis.append(chi2)
    return {"statistic": chis[0], "p_values": ps, "details": {"matrices": N}}


# ------------------------------------------------------------------ bitstream
@register_test("diehard.bitstream", "Тест битового потока (Bitstream)", B,
               min_bits=(1 << 21) + 19,
               description="Перекрывающиеся 20-битные слова в потоке из 2^21 бит. Число "
                           "отсутствующих слов ~ N(141909, 428).")
def bitstream(bits):
    seg = (1 << 21) + 19
    k = min(bits.size // seg, 20)
    ps, zs = [], []
    for i in range(k):
        x = bits[i * seg:(i + 1) * seg]
        v = np.zeros(1 << 21, dtype=np.int64)
        for j in range(20):
            v = (v << 1) | x[j: j + (1 << 21)]
        missing = (1 << 20) - np.unique(v).size
        z = (missing - 141909) / 428
        zs.append(z)
        ps.append(normal_two_sided(z))
    return {"statistic": float(zs[0]), "p_values": ps, "details": {"samples": k}}


# ------------------------------------------------------------------ count the 1s
_LETTER_P = np.array([37, 56, 70, 56, 37]) / 256.0


@register_test("diehard.count_ones_stream", "Подсчёт единиц в потоке байт", B,
               min_bits=8 * 64000,
               description="Байты отображаются в 'буквы' A..E по числу единиц; частоты "
                           "перекрывающихся 5-буквенных слов. Статистика Q5−Q4 ~ χ²(2500).")
def count_ones_stream(bits):
    nbytes = bits.size // 8
    by = np.packbits(bits[: nbytes * 8])
    ones = np.unpackbits(by.reshape(-1, 1), axis=1).sum(axis=1)
    letters = np.select([ones <= 2, ones == 3, ones == 4, ones == 5], [0, 1, 2, 3], 4)

    def q(k):
        x = np.concatenate([letters, letters[: k - 1]])
        L = letters.size
        v = np.zeros(L, dtype=np.int64)
        for j in range(k):
            v = v * 5 + x[j: j + L]
        cnt = np.bincount(v, minlength=5 ** k).astype(float)
        # ожидаемая частота каждого из 5^k слов: L · Π P(буква)
        pw = _LETTER_P
        for _ in range(k - 1):
            pw = np.multiply.outer(pw, _LETTER_P).ravel()
        exp = L * pw
        return float(((cnt - exp) ** 2 / exp).sum())

    stat = q(5) - q(4)
    z = (stat - 2500) / sqrt(5000)
    return {"statistic": stat, "p_values": [normal_two_sided(z)], "details": {"letters": int(nbytes)}}


# ------------------------------------------------------------------ геометрические
@register_test("diehard.parking_lot", "Парковка (Parking lot)", B, min_bits=24000 * 32,
               complexity="O(n·k)",
               description="12000 попыток 'припарковать' квадраты на площадке 100×100. "
                           "Число успешных ~ N(3523, 21.9).")
def parking_lot(bits, attempts=12000):
    u = bits_to_uniforms(bits)
    trials = min(u.size // (2 * attempts), 10)
    ks = []
    for t in range(trials):
        pts = u[t * 2 * attempts:(t + 1) * 2 * attempts].reshape(-1, 2) * 100
        px = np.empty(attempts)
        py = np.empty(attempts)
        k = 0
        for x, y in pts:
            if k == 0 or not np.any((np.abs(px[:k] - x) <= 1) & (np.abs(py[:k] - y) <= 1)):
                px[k], py[k] = x, y
                k += 1
        ks.append(k)
    z = (np.mean(ks) - 3523) / (21.9 / sqrt(trials))
    return {"statistic": float(np.mean(ks)), "p_values": [normal_two_sided(z)],
            "details": {"trials": trials, "parked": ks}}


@register_test("diehard.minimum_distance", "Минимальное расстояние (2D)", B,
               min_bits=16000 * 32,
               description="8000 точек в квадрате 10000×10000; квадрат минимального расстояния "
                           "~ Exp(среднее 0.995).")
def minimum_distance(bits, npts=8000):
    u = bits_to_uniforms(bits)
    trials = min(u.size // (2 * npts), 100)
    ps = []
    for t in range(trials):
        pts = u[t * 2 * npts:(t + 1) * 2 * npts].reshape(-1, 2) * 10000
        d, _ = cKDTree(pts).query(pts, k=2)
        d2 = d[:, 1].min() ** 2
        ps.append(1 - np.exp(-d2 / 0.995))
    p = ks_uniform(ps) if trials > 1 else float(min(2 * ps[0], 2 * (1 - ps[0]), 1))
    return {"statistic": float(np.mean(ps)), "p_values": [p], "details": {"trials": trials}}


@register_test("diehard.spheres_3d", "3D-сферы", B, min_bits=12000 * 32,
               description="4000 точек в кубе со стороной 1000; куб минимального расстояния "
                           "~ Exp(среднее 30).")
def spheres_3d(bits, npts=4000):
    u = bits_to_uniforms(bits)
    trials = min(u.size // (3 * npts), 20)
    ps = []
    for t in range(trials):
        pts = u[t * 3 * npts:(t + 1) * 3 * npts].reshape(-1, 3) * 1000
        d, _ = cKDTree(pts).query(pts, k=2)
        r3 = d[:, 1].min() ** 3
        ps.append(1 - np.exp(-r3 / 30))
    p = ks_uniform(ps) if trials > 1 else float(min(2 * ps[0], 2 * (1 - ps[0]), 1))
    return {"statistic": float(np.mean(ps)), "p_values": [p], "details": {"trials": trials}}


# ------------------------------------------------------------------ крэпс
def _craps_length_probs(maxlen=21):
    ways = {s: 6 - abs(s - 7) for s in range(2, 13)}
    p = np.zeros(maxlen + 1)
    p[1] = sum(ways[s] for s in (2, 3, 7, 11, 12)) / 36
    for pt in (4, 5, 6, 8, 9, 10):
        q = ways[pt] / 36
        r = (ways[pt] + 6) / 36
        for k in range(2, maxlen):
            p[k] += q * (1 - r) ** (k - 2) * r
    p[maxlen] = 1 - p[1:maxlen].sum()
    return p[1:]


@register_test("diehard.craps", "Игра в кости (Craps)", B, min_bits=20000 * 32,
               complexity="O(n), последовательный",
               description="Моделирование игры в крэпс: число выигрышей ~ Bin(n, 244/495) и "
                           "распределение длины партий. Даёт 2 p-значения.")
def craps(bits):
    u = bits_to_uniforms(bits)
    rolls = (np.floor(u * 6).astype(np.int64) + 1)
    rolls = rolls[: (rolls.size // 2) * 2]
    dice = (rolls[0::2] + rolls[1::2]).tolist()
    wins, lengths = 0, []
    i, n = 0, len(dice)
    while i < n:
        s = dice[i]
        i += 1
        if s in (7, 11):
            wins += 1
            lengths.append(1)
            continue
        if s in (2, 3, 12):
            lengths.append(1)
            continue
        k = 1
        while i < n:
            t = dice[i]
            i += 1
            k += 1
            if t == s or t == 7:
                wins += t == s
                lengths.append(k)
                break
        else:
            break
    games = len(lengths)
    pw = 244 / 495
    z = (wins - games * pw) / sqrt(games * pw * (1 - pw))
    obs = np.bincount(np.minimum(lengths, 21), minlength=22)[1:]
    _, _, p_len = chi2_test(obs, games * _craps_length_probs())
    return {"statistic": float(z), "p_values": [normal_two_sided(z), p_len],
            "details": {"games": games, "wins": wins}}


# ------------------------------------------------------------------ серии вверх/вниз
_KNUTH_A = np.array([
    [4529.4, 9044.9, 13568, 18091, 22615, 27892],
    [9044.9, 18097, 27139, 36187, 45234, 55789],
    [13568, 27139, 40721, 54281, 67852, 83685],
    [18091, 36187, 54281, 72414, 90470, 111580],
    [22615, 45234, 67852, 90470, 113262, 139476],
    [27892, 55789, 83685, 111580, 139476, 172860]])
_KNUTH_B = np.array([1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840])


def _runs_up_stat(u):
    n = u.size
    up = u[1:] > u[:-1]
    # границы серий: позиция, где рост прерывается
    breaks = np.flatnonzero(~up) + 1
    edges = np.r_[0, breaks, n]
    lens = np.diff(edges)
    c = np.bincount(np.minimum(lens, 6), minlength=7)[1:].astype(float)
    d = c - n * _KNUTH_B
    return float(d @ _KNUTH_A @ d / (n - 6))


@register_test("diehard.runs_updown", "Серии вверх и вниз (Кнут)", B, min_bits=4000 * 32,
               description="Длины монотонно возрастающих/убывающих серий чисел; статистика "
                           "Кнута с ковариационной матрицей, ~ χ²(6). Даёт 2 p-значения.")
def runs_updown(bits):
    u = bits_to_uniforms(bits)
    v_up, v_dn = _runs_up_stat(u), _runs_up_stat(-u)
    return {"statistic": v_up, "p_values": [float(stats.chi2.sf(v_up, 6)), float(stats.chi2.sf(v_dn, 6))],
            "details": {"V_up": v_up, "V_down": v_dn, "numbers": int(u.size)}}
