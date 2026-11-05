"""Запуск тестов и батарей со сбором метрик (p-значения, время, память, скорость)."""
from __future__ import annotations

import time
import tracemalloc
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats

from .core import REGISTRY, InsufficientData, TestResult, as_bits, combined_p
from . import tests_nist, tests_diehard, tests_testu01, tests_extra  # noqa: F401 (регистрация)
from .mathutil import igamc

# ------------------------------------------------------------------ батареи
def _ids(prefix):
    return [k for k in REGISTRY if k.startswith(prefix + ".")]


# Оптимальный набор (регламентный режим). Обоснован экспериментально
# (rngtest.optimize: мощность против 14 сценариев дефектов и стоимость каждого
# теста) и теоретически (docs/TESTS.md, раздел "Оптимальный набор"). Каждый тест
# закрывает свой класс дефектов; дублирующие критерии (кумулятивные суммы ≈
# частотный, шаблоны ≈ serial/ApEn) исключены, т.к. каждый лишний тест повышает
# вероятность ложной отбраковки: 1 − (1 − α)^T.
OPTIMAL = [
    "nist.frequency",            # смещение
    "nist.block_frequency",      # локальное смещение, дрейф
    "nist.runs",                 # корреляция соседних бит
    "hac.autocorrelation",       # корреляции на сдвигах 1..32
    "nist.longest_run",          # залипание, короткий период младших бит
    "testu01.hamming_indep",     # зависимость весов соседних слов, дрейф
    "nist.dft",                  # периодичность
    "nist.rank",                 # линейность над GF(2) (xorshift, ЛКГ, РСЛОС малой степени)
    "nist.linear_complexity",    # РСЛОС степени до M/2
]

# Облегчённый набор для динамического контроля на IoT: только O(n) по времени,
# без БПФ и алгоритма Берлекэмпа–Мэсси. Ранговый тест оставлен: он дешёвый
# (строки матриц = машинные слова) и единственный из быстрых ловит линейные генераторы.
IOT_LIGHT = ["nist.frequency", "nist.block_frequency", "nist.runs", "hac.autocorrelation",
             "nist.longest_run", "testu01.hamming_indep", "nist.rank"]

BATTERIES: Dict[str, Dict] = {
    "nist": {"name": "NIST SP 800-22", "tests": _ids("nist")},
    "diehard": {"name": "DieHard", "tests": _ids("diehard")},
    "testu01": {"name": "TestU01 SmallCrush", "tests": _ids("testu01")},
    "fips": {"name": "FIPS 140-2", "tests": _ids("fips")},
    "hac": {"name": "Menezes (HAC)", "tests": _ids("hac")},
    "optimal": {"name": "Оптимальный набор", "tests": OPTIMAL},
    "iot": {"name": "IoT-набор", "tests": IOT_LIGHT},
}


def battery_tests(key_or_list) -> List[str]:
    if isinstance(key_or_list, str):
        if key_or_list in BATTERIES:
            return list(BATTERIES[key_or_list]["tests"])
        return [t.strip() for t in key_or_list.split(",") if t.strip()]
    return list(key_or_list)


# ------------------------------------------------------------------ один тест
def run_test(test_id: str, bits: np.ndarray, alpha: float = 0.01,
             params: Optional[dict] = None, measure_memory: bool = True) -> TestResult:
    spec = REGISTRY[test_id]
    p = dict(spec.defaults)
    p.update(params or {})
    n = int(bits.size)
    base = dict(test_id=test_id, name=spec.name, battery=spec.battery, alpha=alpha,
                n_bits=n, params=p)
    if n < spec.min_bits:
        return TestResult(status="skipped", p_value=None, p_values=[], statistic=None,
                          time_s=0.0, peak_mem_bytes=0,
                          message=f"нужно не менее {spec.min_bits:,} бит".replace(",", " "), **base)
    if measure_memory:
        tracemalloc.start()
        tracemalloc.reset_peak() if hasattr(tracemalloc, "reset_peak") else None
    t0 = time.perf_counter()
    try:
        out = spec.func(bits, **p)
        status, msg = None, ""
    except InsufficientData as e:
        out, status, msg = None, "skipped", str(e)
    except Exception as e:  # pragma: no cover - защита интерфейса от падений
        out, status, msg = None, "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    dt = time.perf_counter() - t0
    peak = 0
    if measure_memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    if out is None:
        return TestResult(status=status, p_value=None, p_values=[], statistic=None,
                          time_s=dt, peak_mem_bytes=peak, message=msg, **base)
    ps = [float(x) for x in out["p_values"]]
    pv = combined_p(ps)
    if "passed" in out:          # тесты с фиксированными границами (FIPS 140-2)
        passed = bool(out["passed"])
    else:
        passed = bool(np.isfinite(pv) and pv >= alpha)
    stat = out.get("statistic")
    details = _jsonable(out.get("details", {}))
    if "passed" in out:
        details["decision"] = "bounds"
    return TestResult(status="pass" if passed else "fail", p_value=pv, p_values=ps,
                      statistic=None if stat is None else float(stat), time_s=dt,
                      peak_mem_bytes=peak, details=details, **base)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ------------------------------------------------------------------ батарея
@dataclass
class SecondLevel:
    """Анализ второго уровня NIST по m подпоследовательностям."""
    test_id: str
    name: str
    proportion: float
    proportion_min: float
    uniformity_p: float
    passed: bool


@dataclass
class BatteryReport:
    battery: str
    n_bits: int
    alpha: float
    results: List[TestResult]
    total_time_s: float
    peak_mem_bytes: int
    source: str = ""
    subsequences: int = 1
    second_level: List[SecondLevel] = field(default_factory=list)

    @property
    def counted(self):
        return [r for r in self.results if r.status in ("pass", "fail")]

    @property
    def n_pass(self):
        return sum(r.status == "pass" for r in self.results)

    @property
    def n_fail(self):
        return sum(r.status == "fail" for r in self.results)

    @property
    def n_skipped(self):
        return sum(r.status in ("skipped", "error") for r in self.results)

    @property
    def throughput_mbit_s(self):
        return self.n_bits * self.subsequences / self.total_time_s / 1e6 if self.total_time_s else None

    @property
    def fail_count_p(self) -> float:
        """P(число отказов ≥ наблюдаемого | последовательность случайна) — Bin(T, α)."""
        T = len(self.counted)
        if T == 0:
            return float("nan")
        return float(stats.binom.sf(self.n_fail - 1, T, self.alpha))

    @property
    def p_value(self) -> float:
        """p-значение батареи: 1 − (1 − p_min)^T — поправка Шидака на число тестов T.

        Согласовано с вердиктом: p < α  ⇔  какой-либо тест дал p < strong_threshold.
        """
        ps = [r.p_value for r in self.counted if r.p_value is not None and np.isfinite(r.p_value)]
        if not ps:
            return float("nan")
        T = len({r.test_id for r in self.counted})
        return float(-np.expm1(T * np.log1p(-min(min(ps), 1 - 1e-16))))

    @property
    def strong_threshold(self) -> float:
        """Порог "явного" отказа: 1 − (1 − α)^(1/T) ≈ α/T (поправка Шидака на число тестов)."""
        T = max(len({r.test_id for r in self.counted}), 1)
        return float(-np.expm1(np.log1p(-self.alpha) / T))

    @property
    def verdict(self) -> str:
        """Итоговое решение по батарее.

        Даже идеальный генератор в среднем не проходит долю α тестов, поэтому один
        отказ сам по себе не означает неслучайность. Последовательность отвергается, если
        (а) хотя бы один тест дал p < 1 − (1 − α)^(1/T) (Шидак) или нарушил границы FIPS 140-2, или
        (б) число отказов неправдоподобно велико для Bin(T, α) (P < 0.001), или
        (в) в режиме подпоследовательностей не пройден анализ второго уровня NIST.
        """
        if self.second_level:
            ok = all(s.passed for s in self.second_level)
        elif not self.counted:
            return "Нет данных"
        else:
            strong = any(r.status == "fail" and (r.details.get("decision") == "bounds"
                                                 or (r.p_value or 0) < self.strong_threshold)
                         for r in self.counted)
            ok = not strong and self.fail_count_p >= 0.001
            if ok and self.n_fail:
                return "Гипотеза о случайности не отвергается (отказы в пределах статистической нормы)"
        return ("Гипотеза о случайности не отвергается" if ok
                else "Последовательность НЕ случайна")

    @property
    def ok(self) -> bool:
        return self.verdict.startswith("Гипотеза")

    def to_dict(self):
        return {
            "battery": self.battery, "source": self.source, "n_bits": self.n_bits,
            "alpha": self.alpha, "subsequences": self.subsequences,
            "verdict": self.verdict, "passed": self.n_pass, "failed": self.n_fail,
            "skipped": self.n_skipped, "p_value": self.p_value, "fail_count_p": self.fail_count_p,
            "total_time_s": self.total_time_s, "peak_mem_bytes": self.peak_mem_bytes,
            "throughput_mbit_s": self.throughput_mbit_s,
            "second_level": [s.__dict__ for s in self.second_level],
            "results": [r.to_dict() for r in self.results],
        }


ProgressCb = Callable[[int, int, str], None]


def run_battery(tests, bits, alpha: float = 0.01, params: Optional[Dict[str, dict]] = None,
                progress: Optional[ProgressCb] = None, subsequences: int = 1,
                measure_memory: bool = True, battery_name: str = "", source: str = "") -> BatteryReport:
    """Запустить набор тестов.

    tests          — ключ батареи из BATTERIES, строка "id1,id2" или список id;
    params         — переопределение параметров: {test_id: {param: value}};
    subsequences   — разбить последовательность на m частей и выполнить анализ
                     второго уровня NIST (доля прошедших + равномерность p-значений).
    """
    bits = as_bits(bits)
    ids = battery_tests(tests)
    if not battery_name:
        battery_name = BATTERIES[tests]["name"] if isinstance(tests, str) and tests in BATTERIES else "Пользовательская"
    params = params or {}
    m = max(1, int(subsequences))
    seg = bits.size // m
    parts = [bits[i * seg:(i + 1) * seg] for i in range(m)]
    results: List[TestResult] = []
    total = len(ids) * m
    done = 0
    t0 = time.perf_counter()
    for tid in ids:
        for part in parts:
            if progress:
                progress(done, total, REGISTRY[tid].name)
            results.append(run_test(tid, part, alpha, params.get(tid), measure_memory))
            done += 1
    if progress:
        progress(total, total, "готово")
    elapsed = time.perf_counter() - t0
    rep = BatteryReport(battery=battery_name, n_bits=seg, alpha=alpha, results=results,
                        total_time_s=elapsed,
                        peak_mem_bytes=max([r.peak_mem_bytes for r in results] or [0]),
                        source=source, subsequences=m)
    if m > 1:
        rep.second_level = second_level_analysis(results, ids, alpha, m)
    return rep


def second_level_analysis(results: List[TestResult], ids: List[str], alpha: float,
                          m: int) -> List[SecondLevel]:
    """NIST SP 800-22, раздел 4.2: доля прошедших и равномерность p-значений."""
    out = []
    for tid in ids:
        rs = [r for r in results if r.test_id == tid and r.status in ("pass", "fail")]
        if not rs:
            continue
        k = len(rs)
        prop = sum(r.status == "pass" for r in rs) / k
        ph = 1 - alpha
        pmin = ph - 3 * np.sqrt(ph * alpha / k)
        pv = [r.p_value for r in rs]
        F = np.histogram(pv, bins=10, range=(0, 1))[0]
        chi2 = float(((F - k / 10) ** 2 / (k / 10)).sum())
        unif = igamc(9 / 2, chi2 / 2) if k >= 10 else float("nan")
        passed = prop >= pmin and (not np.isfinite(unif) or unif >= 0.0001)
        out.append(SecondLevel(tid, REGISTRY[tid].name, prop, pmin, unif, passed))
    return out
