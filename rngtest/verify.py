"""Модуль верификации (ТЗ п. 5.1, 6.5.2, этап 2.3).

1. **NIST SP 800-22** — сравнение с контрольными p-значениями из Приложения B
   стандарта (двоичное разложение e, n = 10^6). Разложение e вычисляется
   точно, поэтому сверка воспроизводима без внешних файлов.
2. **DieHard** — адаптер к утилите ``dieharder`` (пакет ``dieharder`` в
   Debian/Ubuntu). Если утилита установлена, одна и та же последовательность
   прогоняется через неё и через нашу реализацию, результаты сводятся в таблицу.
3. **Функциональная проверка** — заведомо случайные и заведомо неслучайные
   последовательности (ТЗ п. 6.5.1).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .constants import e_bits
from .runner import run_test

# (id теста, параметры, индекс p-значения, эталон из SP 800-22 Appendix B, допуск)
NIST_REFERENCE = [
    ("nist.frequency", {}, 0, 0.953749, 1e-6),
    ("nist.block_frequency", {"M": 128}, 0, 0.211072, 1e-6),
    ("nist.cumulative_sums", {}, 0, 0.669887, 2e-6),
    ("nist.cumulative_sums", {}, 1, 0.724266, 2e-6),
    ("nist.runs", {}, 0, 0.561917, 1e-6),
    ("nist.longest_run", {}, 0, 0.718945, 1e-6),
    ("nist.rank", {}, 0, 0.306156, 1e-6),
    ("nist.dft", {}, 0, 0.847187, 1e-6),
    ("nist.non_overlapping_template", {"m": 9}, 0, 0.078790, 1e-6),
    ("nist.overlapping_template", {"m": 9, "legacy": True}, 0, 0.110434, 1e-5),
    ("nist.universal", {}, 0, 0.282568, 1e-6),
    ("nist.approximate_entropy", {"m": 10}, 0, 0.700073, 1e-6),
    ("nist.random_excursions", {}, 4, 0.786868, 1e-6),
    ("nist.random_excursions_variant", {}, 8, 0.826009, 1e-6),
    ("nist.linear_complexity", {"M": 500}, 0, 0.826335, 1e-6),
    ("nist.serial", {"m": 16}, 0, 0.766182, 1e-6),
    ("nist.serial", {"m": 16}, 1, 0.462921, 1e-6),
]

_NOTES = {
    "nist.overlapping_template": "в Приложении B использованы старые вероятности π_i (параметр legacy=True)",
    "nist.random_excursions": "состояние x = +1",
    "nist.random_excursions_variant": "состояние x = −1",
    "nist.cumulative_sums": "",
}


@dataclass
class RefRow:
    test_id: str
    params: dict
    p_index: int
    expected: float
    actual: Optional[float]
    tol: float
    note: str

    @property
    def ok(self) -> bool:
        return self.actual is not None and abs(self.actual - self.expected) <= self.tol

    @property
    def diff(self) -> Optional[float]:
        return None if self.actual is None else self.actual - self.expected


def verify_nist(n: int = 1_000_000) -> List[RefRow]:
    bits = e_bits(n)
    rows = []
    cache = {}
    for tid, params, idx, exp, tol in NIST_REFERENCE:
        key = (tid, tuple(sorted(params.items())))
        if key not in cache:
            cache[key] = run_test(tid, bits, params=params, measure_memory=False)
        r = cache[key]
        act = r.p_values[idx] if len(r.p_values) > idx else None
        note = _NOTES.get(tid, "")
        if tid == "nist.cumulative_sums":
            note = "прямое направление" if idx == 0 else "обратное направление"
        if tid == "nist.serial":
            note = f"p-значение {idx + 1}"
        rows.append(RefRow(tid, params, idx, exp, act, tol, note))
    return rows


# ------------------------------------------------------------------ dieharder
# соответствие наших тестов номерам тестов dieharder (-d N)
DIEHARDER_MAP = {
    "diehard.birthday_spacings": 0,
    "diehard.rank_32x32": 2,
    "diehard.rank_6x8": 3,
    "diehard.bitstream": 4,
    "diehard.count_ones_stream": 8,
    "diehard.parking_lot": 10,
    "diehard.minimum_distance": 11,
    "diehard.spheres_3d": 12,
    "diehard.runs_updown": 15,
    "diehard.craps": 16,
}


def dieharder_available() -> Optional[str]:
    return shutil.which("dieharder")


def run_dieharder(bits: np.ndarray, test_num: int, timeout: int = 600) -> List[float]:
    """Запустить dieharder на файле с последовательностью (raw, генератор 201)."""
    exe = dieharder_available()
    if not exe:
        raise RuntimeError("dieharder не найден (Debian/Ubuntu: sudo apt install dieharder)")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(np.packbits(bits).tobytes())
        path = f.name
    try:
        out = subprocess.run([exe, "-g", "201", "-f", path, "-d", str(test_num)],
                             capture_output=True, text=True, timeout=timeout).stdout
    finally:
        os.unlink(path)
    ps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6 and re.fullmatch(r"[0-9.eE-]+", parts[4] or "x"):
            ps.append(float(parts[4]))
    return ps


def compare_dieharder(bits: np.ndarray, alpha: float = 0.01) -> List[dict]:
    rows = []
    for tid, num in DIEHARDER_MAP.items():
        ours = run_test(tid, bits, alpha, measure_memory=False)
        try:
            ref = run_dieharder(bits, num)
            ref_p = min(ref) if ref else None
            err = ""
        except Exception as e:
            ref_p, err = None, str(e)
        rows.append({"test": tid, "dieharder_test": num,
                     "our_status": ours.status, "our_p": ours.p_value,
                     "dieharder_p": ref_p,
                     "dieharder_status": None if ref_p is None else ("pass" if ref_p >= alpha else "fail"),
                     "agree": None if ref_p is None else ((ref_p >= alpha) == (ours.status == "pass")),
                     "error": err})
    return rows


# ------------------------------------------------------------------ функциональная проверка
def functional_check(n: int = 1_000_000, tests: Optional[List[str]] = None, alpha: float = 0.01):
    """Каждый тест на заведомо случайной и заведомо неслучайных последовательностях.

    Критерий корректности: тест проходит на эталонной последовательности (PCG64 с
    фиксированным seed — результат воспроизводим) и отвергает "все нули".
    Чередование 0101… и счётчик — справочные: например, частотный тест их
    закономерно пропускает (единиц ровно половина), это не ошибка, а
    ограничение конкретного критерия, которое и компенсирует набор тестов."""
    from .core import REGISTRY
    from .sources import generate
    tests = tests or list(REGISTRY)
    seqs = {
        "PCG64 (seed 1)": generate("pcg64", n, 1),
        "все нули": np.zeros(n, dtype=np.uint8),
        "0101…": np.tile(np.array([0, 1], dtype=np.uint8), n // 2),
        "счётчик": generate("counter", n),
    }
    rows = []
    for tid in tests:
        row = {"test": tid}
        for name, bits in seqs.items():
            row[name] = run_test(tid, bits, alpha, measure_memory=False).status
        row["ok"] = row["PCG64 (seed 1)"] in ("pass", "skipped") and row["все нули"] in ("fail", "skipped")
        rows.append(row)
    return rows
