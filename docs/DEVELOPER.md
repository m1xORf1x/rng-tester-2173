# Документация разработчика

## 1. Архитектура

```
             ┌──────────────┐      ┌──────────────┐
             │   app.py     │      │   cli.py     │      интерфейсы
             │ (Streamlit)  │      │  (argparse)  │
             └──────┬───────┘      └──────┬───────┘
                    └──────────┬──────────┘
                               ▼
  sources.py ──bits──▶   runner.py   ──BatteryReport──▶ report.py (JSON/CSV/TXT)
  (файлы, поток,        (батареи, замер времени
   генераторы)           и памяти, вердикт,
                          второй уровень NIST)
                               │ REGISTRY (core.py)
        ┌──────────────┬───────┴──────┬──────────────┐
        ▼              ▼              ▼              ▼
  tests_nist.py  tests_diehard.py tests_testu01.py tests_extra.py     статистические критерии
        └──────────────┴──────┬───────┴──────────────┘
                              ▼
                 mathutil.py, native/ (C через ctypes)               общая математика / ускорение

  optimize.py — подбор набора    verify.py — эталоны и dieharder    bench.py — профилирование
```

Поток данных: источник → `numpy.ndarray[uint8]` из 0/1 → `run_battery` вызывает
`run_test` для каждого теста → функция теста возвращает словарь
`{statistic, p_values, details}` → `run_test` замеряет время (`perf_counter`) и пиковую
память (`tracemalloc`), принимает решение и упаковывает всё в `TestResult` →
`BatteryReport` считает вердикт и сводные метрики.

## 2. Модули

| Модуль | Назначение | Основные объекты |
|---|---|---|
| `core.py` | контракт теста и реестр | `register_test`, `REGISTRY`, `TestSpec`, `TestResult`, `InsufficientData`, `combined_p` |
| `mathutil.py` | общая математика | `igamc`, `chi2_test` (с объединением категорий), `gf2_rank_batch`, `rank_probabilities`, `window_values`, `bits_to_words/uniforms`, `stirling2_row` |
| `tests_*.py` | реализации критериев | функции, зарегистрированные декоратором |
| `runner.py` | батареи и запуск | `BATTERIES`, `OPTIMAL`, `IOT_LIGHT`, `run_test`, `run_battery`, `BatteryReport`, `second_level_analysis` |
| `sources.py` | ввод данных | `parse_bits`, `load_file`, `stream_chunks`, `GENERATORS`, `generate` |
| `constants.py` | эталонные последовательности | `e_bits`, `pi_bits`, `sqrt_bits` (точная целочисленная арифметика) |
| `native/` | C-ускоритель | `fastrng.c`; `available()`, `lc_blocks`, `longest_runs`, `pack`, `packed_popcount`, `packed_autocorr` |
| `report.py` | отчёты | `to_json`, `to_csv`, `to_txt`, `save` |
| `verify.py` | верификация | `verify_nist`, `NIST_REFERENCE`, `compare_dieharder`, `functional_check` |
| `optimize.py` | подбор набора | `Scenario`, `measure`, `greedy_select`, `OptimizationResult` |
| `bench.py` | профилирование | `benchmark`, `scaling` |
| `cli.py` | CLI | подкоманды `list run stream verify optimize bench generate` |

## 3. Контракт теста

```python
@register_test(
    "group.my_test",              # уникальный id: <батарея>.<имя>
    "Человекочитаемое название",
    "Название батареи",           # группировка в интерфейсе
    min_bits=1000,                # короче — тест будет помечен «пропущен»
    description="Что проверяет и что обнаруживает.",
    complexity="O(n)",
    M=128,                        # параметры по умолчанию (редактируются в GUI / --param)
)
def my_test(bits: np.ndarray, M=128) -> dict:
    ...
    return {
        "statistic": float(stat),         # наблюдаемое значение статистики
        "p_values": [p1, p2],             # одно или несколько p-значений
        "details": {"N": N},              # необязательно: что угодно JSON-сериализуемое
        # "passed": bool,                 # необязательно: решение по фиксированным границам
    }
```

Правила:

* `bits` — `uint8` из 0/1, функция **не должна** его изменять;
* если тест неприменим к данным (мало бит, не выполнено условие), бросьте
  `InsufficientData("причина")` — результат будет `skipped` с сообщением;
* решение «пройден/не пройден» тест сам не принимает (кроме `passed` для стандартов
  с фиксированными границами) — это делает `run_test` по итоговому p и α;
* при нескольких p-значениях итоговое считается поправкой Шидака (`combined_p`).

## 4. Как добавить новый тест

1. Напишите функцию по контракту в подходящем `tests_*.py` (или новом модуле).
2. Если модуль новый, импортируйте его в `runner.py` рядом с остальными
   (`from . import tests_nist, ...`) — регистрация происходит при импорте.
3. При желании добавьте id в нужную батарею в `BATTERIES` или в `OPTIMAL`.
4. Проверьте:
   ```bash
   python -m rngtest run -g pcg64 -t group.my_test           # должен проходить
   python -m rngtest run -g biased -t group.my_test          # должен ловить свой дефект
   python -m rngtest verify --functional
   pytest -q
   ```
5. Тест сразу появляется в GUI (пользовательская батарея, справка, сравнение,
   эксперимент по оптимальному набору) и в CLI.

Как добавить генератор: функция `f(n, seed=None, **kw) -> ndarray` и запись
`Generator(...)` в `GENERATORS` (`sources.py`). Новый сценарий дефекта для подбора
набора — запись в `DEFAULT_SCENARIOS` (`optimize.py`).

## 5. C-ускоритель

* Исходник `rngtest/native/fastrng.c` (C99, без зависимостей). Собирается
  автоматически при первом вызове `native.available()` командой
  `cc -O3 -shared -fPIC -std=c99` (на x86-64 дополнительно `-march=native`) в
  `rngtest/native/libfastrng.so`. Пересобирается при изменении исходника.
* Кросс-компиляция для IoT (ARM Linux):
  `aarch64-linux-gnu-gcc -O3 -shared -fPIC -std=c99 fastrng.c -o libfastrng.so`.
* Новая функция: добавьте её в `fastrng.c`, объявите `argtypes/restype` в
  `get_lib()` и напишите обёртку в `native/__init__.py`. Обязательно оставьте NumPy-путь
  как запасной и сравните их результаты в автотесте (см. `test_native_packed_ops`).

## 6. Вердикт по батарее

`BatteryReport.verdict`: последовательность отвергается, если
(а) какой-либо тест дал p < 1 − (1 − α)^(1/T) или нарушил границы FIPS, или
(б) P(Bin(T, α) ≥ число отказов) < 0.001, или
(в) при `subsequences > 1` не пройден анализ второго уровня NIST (доля прошедших не
ниже (1−α) − 3√(α(1−α)/m), равномерность p-значений P ≥ 0.0001).

## 7. Тестирование

```bash
pytest -q
```

* `test_nist_reference_values` — 17 эталонных значений SP 800-22, Приложение B;
* `test_every_test_passes_random_and_rejects_zeros` — функциональная проверка всех тестов;
* `test_battery_verdicts` — вердикты оптимального и IoT-наборов на хороших и дефектных генераторах;
* `test_berlekamp_massey_numpy_matches_c`, `test_native_packed_ops` — эквивалентность C и NumPy;
* `test_gui.py` — каждая страница интерфейса открывается, основные кнопки работают
  (`streamlit.testing.AppTest`, без браузера).
