"""rngtest — программный комплекс оценки выходных последовательностей датчиков случайных чисел.

Быстрый старт::

    from rngtest import run_battery, generate
    rep = run_battery("nist", generate("urandom", 1_000_000))
    print(rep.verdict)
"""
from .core import REGISTRY, TestResult, register_test
from .runner import BATTERIES, BatteryReport, run_battery, run_test
from .sources import GENERATORS, generate, load_file, parse_bits

__version__ = "1.0.0"
__all__ = ["REGISTRY", "TestResult", "register_test", "BATTERIES", "BatteryReport",
           "run_battery", "run_test", "GENERATORS", "generate", "load_file", "parse_bits"]
