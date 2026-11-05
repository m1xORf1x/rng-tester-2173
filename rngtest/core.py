"""Базовые сущности: результат теста, описание теста, реестр тестов.

Каждый статистический тест — это функция ``f(bits, **params) -> dict``,
зарегистрированная декоратором :func:`register_test`. Функция получает
последовательность в виде ``numpy.ndarray`` из 0/1 (dtype=uint8) и возвращает
словарь с ключами:

* ``statistic`` — наблюдаемое значение статистики критерия;
* ``p_values`` — список p-значений (у многих тестов их несколько);
* ``details`` — произвольная дополнительная информация (необязательно).

Обвязка (замер времени, памяти, принятие решения) делается в
:mod:`rngtest.runner`, поэтому сами тесты остаются "чистой математикой".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any

import numpy as np


class InsufficientData(Exception):
    """Последовательность слишком коротка (или не выполнено условие применимости теста)."""


@dataclass
class TestSpec:
    id: str                      # уникальный идентификатор, например "nist.frequency"
    name: str                    # человекочитаемое название
    battery: str                 # "NIST SP 800-22", "DieHard", ...
    func: Callable[..., dict]
    min_bits: int                # минимальная длина последовательности
    defaults: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    complexity: str = "O(n)"     # асимптотическая трудоёмкость (для справки)


@dataclass
class TestResult:
    test_id: str
    name: str
    battery: str
    status: str                  # "pass" | "fail" | "skipped" | "error"
    p_value: Optional[float]     # итоговое p-значение (с поправкой Шидака при нескольких)
    p_values: List[float]
    statistic: Optional[float]
    alpha: float
    n_bits: int
    time_s: float
    peak_mem_bytes: int
    params: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def throughput_mbit_s(self) -> Optional[float]:
        if self.time_s <= 0 or self.status in ("skipped", "error"):
            return None
        return self.n_bits / self.time_s / 1e6

    def to_dict(self) -> dict:
        d = asdict(self)
        d["throughput_mbit_s"] = self.throughput_mbit_s
        return d


REGISTRY: Dict[str, TestSpec] = {}


def register_test(id: str, name: str, battery: str, min_bits: int,
                  description: str = "", complexity: str = "O(n)", **defaults):
    """Декоратор регистрации теста в глобальном реестре."""
    def deco(func):
        if id in REGISTRY:
            raise ValueError(f"Тест {id} уже зарегистрирован")
        REGISTRY[id] = TestSpec(id=id, name=name, battery=battery, func=func,
                                min_bits=min_bits, defaults=dict(defaults),
                                description=description or (func.__doc__ or "").strip(),
                                complexity=complexity)
        return func
    return deco


def combined_p(p_values: List[float]) -> float:
    """Итоговое p-значение для теста с несколькими p: min p с поправкой Шидака.

    Если гипотеза верна, все p ~ U(0,1) (приближённо независимы), тогда
    P(min p <= x) = 1 - (1 - x)^k. Так решение по тесту принимается на заданном
    уровне значимости alpha, а не на уровне 1-(1-alpha)^k.
    """
    ps = [p for p in p_values if p is not None and np.isfinite(p)]
    if not ps:
        return float("nan")
    k = len(ps)
    pmin = min(ps)
    if k == 1:
        return float(pmin)
    return float(-np.expm1(k * np.log1p(-pmin))) if pmin < 1 else 1.0


def as_bits(x) -> np.ndarray:
    """Привести вход к массиву 0/1 dtype=uint8."""
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.size and a.max() > 1:
        raise ValueError("Последовательность должна состоять из 0 и 1")
    return a
