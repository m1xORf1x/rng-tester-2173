"""Экспериментальный подбор оптимального набора тестов (ТЗ, этапы 1.2 и 2.2).

Идея: набор тестов должен (1) обнаруживать все известные классы дефектов,
(2) иметь минимальную вычислительную стоимость и (3) не давать лишних ложных
отказов (каждый лишний тест увеличивает вероятность ложной отбраковки
хорошей последовательности: 1 − (1 − α)^T).

Алгоритм:
1. Для каждого сценария дефекта (генератор + сила дефекта) и каждого теста
   оценивается мощность — доля отвергнутых последовательностей из R повторов.
2. Оценивается стоимость теста — время на 1 Мбит.
3. Жадный алгоритм взвешенного покрытия множеств: на каждом шаге добавляется
   тест с максимальным приростом "покрытия" (Σ по сценариям прироста
   вероятности обнаружения хотя бы одним тестом набора), делённым на стоимость
   (с показателем cost_weight). Остановка, когда прирост < min_gain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .core import REGISTRY
from .runner import run_test
from .sources import GENERATORS


@dataclass
class Scenario:
    key: str          # ключ генератора
    label: str
    kwargs: dict = field(default_factory=dict)


DEFAULT_SCENARIOS = [
    Scenario("biased", "Смещение p=0.505", {"p": 0.505}),
    Scenario("biased", "Смещение p=0.51", {"p": 0.51}),
    Scenario("markov", "Корреляция 0.505", {"stay": 0.505}),
    Scenario("markov", "Корреляция 0.51", {"stay": 0.51}),
    Scenario("periodic", "Период 2^17", {"period": 1 << 17}),
    Scenario("periodic", "Период 2^12", {"period": 1 << 12}),
    Scenario("lfsr", "РСЛОС степени 31"),
    Scenario("xorshift32", "Xorshift32"),
    Scenario("randu", "RANDU"),
    Scenario("lcg_low", "ЛКГ, младшие биты"),
    Scenario("sticky", "Залипший бит"),
    Scenario("drift", "Дрейф ±1%", {"amp": 0.01}),
    Scenario("drift", "Дрейф ±3%", {"amp": 0.03}),
    Scenario("counter", "Счётчик"),
]


@dataclass
class OptimizationResult:
    tests: List[str]
    scenarios: List[str]
    power: np.ndarray            # [тест, сценарий] — доля обнаружений
    false_alarm: np.ndarray      # [тест] — доля ложных отказов на хорошем генераторе
    cost: np.ndarray             # [тест] — секунд на 1 Мбит
    selected: List[str]
    steps: List[dict]            # история жадного алгоритма

    def coverage(self, subset: List[str]) -> np.ndarray:
        """Вероятность обнаружения каждого сценария хотя бы одним тестом поднабора."""
        idx = [self.tests.index(t) for t in subset]
        if not idx:
            return np.zeros(len(self.scenarios))
        return 1 - np.prod(1 - self.power[idx], axis=0)


def measure(tests: Optional[List[str]] = None, scenarios: Optional[List[Scenario]] = None,
            n_bits: int = 1_000_000, reps: int = 5, alpha: float = 0.01,
            good_key: str = "pcg64", progress: Optional[Callable[[float, str], None]] = None):
    tests = tests or list(REGISTRY)
    scenarios = scenarios or DEFAULT_SCENARIOS
    T, S = len(tests), len(scenarios)
    power = np.zeros((T, S))
    fa = np.zeros(T)
    cost = np.zeros(T)
    applicable = np.zeros(T)
    total = (S + 1) * reps
    step = 0
    for r in range(reps):
        seqs = [(None, GENERATORS[good_key].func(n_bits, 10_000 + r))]
        seqs += [(j, GENERATORS[sc.key].func(n_bits, 20_000 + r, **sc.kwargs))
                 for j, sc in enumerate(scenarios)]
        for j, bits in seqs:
            bits = np.ascontiguousarray(bits, dtype=np.uint8)
            if progress:
                progress(step / total, scenarios[j].label if j is not None else "эталонный генератор")
            step += 1
            for i, t in enumerate(tests):
                res = run_test(t, bits, alpha, measure_memory=False)
                if res.status not in ("pass", "fail"):
                    continue
                rej = res.status == "fail"
                if j is None:
                    fa[i] += rej
                    cost[i] += res.time_s
                    applicable[i] += 1
                else:
                    power[i, j] += rej
    power /= reps
    fa /= reps
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = np.where(applicable > 0, cost / np.maximum(applicable, 1) / (n_bits / 1e6), np.inf)
    if progress:
        progress(1.0, "готово")
    return tests, [s.label for s in scenarios], power, fa, cost


def greedy_select(tests, scenarios, power, fa, cost, cost_weight: float = 0.5,
                  min_gain: float = 0.02, max_tests: int = 12) -> OptimizationResult:
    """Жадный выбор: максимизировать прирост покрытия / стоимость^cost_weight."""
    selected: List[str] = []
    steps = []
    miss = np.ones(len(scenarios))       # вероятность пропуска сценария текущим набором
    usable = [i for i in range(len(tests)) if np.isfinite(cost[i])]
    c = np.maximum(cost, 1e-4)
    while len(selected) < max_tests:
        best, best_score, best_gain = None, 0.0, 0.0
        for i in usable:
            if tests[i] in selected:
                continue
            gain = float((miss * power[i]).sum())
            score = gain / (c[i] ** cost_weight)
            if score > best_score:
                best, best_score, best_gain = i, score, gain
        if best is None or best_gain < min_gain:
            break
        selected.append(tests[best])
        miss = miss * (1 - power[best])
        steps.append({"test": tests[best], "gain": best_gain, "cost_s_per_mbit": float(cost[best]),
                      "coverage": float((1 - miss).mean())})
    return OptimizationResult(tests, scenarios, power, fa, cost, selected, steps)


def optimize(**kw) -> OptimizationResult:
    sel_kw = {k: kw.pop(k) for k in ("cost_weight", "min_gain", "max_tests") if k in kw}
    return greedy_select(*measure(**kw), **sel_kw)
