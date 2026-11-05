"""Интерфейс командной строки (ТЗ п. 6.3).

Примеры:
    python -m rngtest list
    python -m rngtest run data.bin --battery nist --report report.json
    python -m rngtest run --generator urandom -n 1000000 --battery optimal
    python -m rngtest run data.txt --format ascii --tests nist.frequency,nist.runs --alpha 0.05
    head -c 10000000 /dev/urandom | python -m rngtest stream --window 1000000 --battery iot
    python -m rngtest verify
    python -m rngtest optimize --reps 5
    python -m rngtest bench
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _progress(done, total, name):
    w = 30
    k = int(w * done / max(total, 1))
    sys.stderr.write(f"\r[{'#' * k}{'.' * (w - k)}] {done}/{total} {name[:40]:40s}")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _parse_params(items):
    """--param nist.serial.m=8  ->  {"nist.serial": {"m": 8}}"""
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        tid, p = key.rsplit(".", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        out.setdefault(tid, {})[p] = val
    return out


def cmd_list(a):
    from .core import REGISTRY
    from .runner import BATTERIES
    print("Батареи:")
    for k, b in BATTERIES.items():
        print(f"  {k:10s} {b['name']}  ({len(b['tests'])} тестов)")
    print("\nТесты:")
    cur = None
    for tid, s in REGISTRY.items():
        if s.battery != cur:
            cur = s.battery
            print(f"\n  [{cur}]")
        print(f"  {tid:34s} {s.name}  (мин. {s.min_bits:,} бит)".replace(",", " "))


def cmd_run(a):
    from .runner import run_battery
    from .sources import generate, load_file
    from . import report
    if a.generator:
        bits = generate(a.generator, a.n, a.seed)
        source = f"генератор {a.generator}"
    elif a.file:
        bits = load_file(a.file, a.format, a.n)
        source = a.file
    else:
        from .sources import parse_bits
        bits = parse_bits(sys.stdin.buffer.read(), a.format)[: a.n]
        source = "stdin"
    tests = a.tests or a.battery
    rep = run_battery(tests, bits, alpha=a.alpha, params=_parse_params(a.param),
                      progress=None if a.quiet else _progress, subsequences=a.subsequences,
                      source=source)
    print(report.to_txt(rep))
    if a.report:
        for path in a.report:
            report.save(rep, path)
            print(f"Отчёт сохранён: {path}")
    return 0 if rep.ok else 1


def cmd_stream(a):
    """Динамический режим: тестирование потока окнами фиксированной длины."""
    from .runner import run_battery
    from .sources import stream_chunks
    fh = open(a.file, "rb") if a.file and a.file != "-" else sys.stdin.buffer
    bad = 0
    for i, chunk in enumerate(stream_chunks(fh, a.window, a.format)):
        rep = run_battery(a.tests or a.battery, chunk, alpha=a.alpha, measure_memory=False)
        failed = [r.test_id for r in rep.results if r.status == "fail"]
        bad += not rep.ok
        print(f"окно {i:5d}: {'OK ' if rep.ok else 'ТРЕВОГА'}  {rep.n_pass}/{len(rep.counted)} "
              f"{rep.throughput_mbit_s:7.1f} Мбит/с  {','.join(failed)}", flush=True)
        if a.max_windows and i + 1 >= a.max_windows:
            break
    return 1 if bad else 0


def cmd_verify(a):
    from .verify import verify_nist, dieharder_available, compare_dieharder, functional_check
    print("Сверка с эталонными значениями NIST SP 800-22, Приложение B (двоичное разложение e):\n")
    ok_all = True
    for r in verify_nist():
        ok_all &= r.ok
        act = "—" if r.actual is None else f"{r.actual:.6f}"
        print(f"  {'OK ' if r.ok else 'ERR'} {r.test_id:34s} эталон {r.expected:.6f}  получено {act}  {r.note}")
    print(f"\nИтог: {'все значения совпали' if ok_all else 'есть расхождения'}")
    if a.functional:
        print("\nФункциональная проверка (случайная / неслучайные последовательности):")
        for row in functional_check(a.n):
            marks = "  ".join(f"{k}={row[k]}" for k in row if k not in ("test", "ok"))
            print(f"  {'OK ' if row['ok'] else '!! '} {row['test']:34s} {marks}")
    if dieharder_available():
        from .sources import generate
        print("\nСравнение с dieharder:")
        for row in compare_dieharder(generate("urandom", a.n)):
            print("  ", row)
    else:
        print("\n(dieharder не установлен — сравнение с DieHard пропущено)")
    return 0 if ok_all else 1


def cmd_optimize(a):
    from .optimize import optimize
    res = optimize(n_bits=a.n, reps=a.reps, cost_weight=a.cost_weight,
                   progress=lambda f, s: _progress(int(f * 100), 100, s))
    print("\nВыбранный набор:")
    for st in res.steps:
        print(f"  + {st['test']:34s} прирост {st['gain']:.2f}  стоимость {st['cost_s_per_mbit'] * 1000:.1f} мс/Мбит"
              f"  покрытие {st['coverage']:.1%}")
    cov = res.coverage(res.selected)
    print("\nВероятность обнаружения по сценариям:")
    for s, c in zip(res.scenarios, cov):
        print(f"  {s:28s} {c:.0%}")
    return 0


def cmd_bench(a):
    from .bench import benchmark
    from . import native
    print(f"C-ускоритель: {'доступен' if native.available() else 'недоступен'}\n")
    for r in benchmark(a.n, include_python=not a.no_python):
        thr = "" if r["throughput_mbit_s"] is None else f"{r['throughput_mbit_s']:10.1f} Мбит/с"
        print(f"  {r['operation']:40s} {r['implementation']:38s} {r['time_s'] * 1000:10.2f} мс {thr}"
              f"  {r['peak_mem_bytes'] / 1024:10.0f} КБ")
    return 0


def cmd_generate(a):
    from .sources import generate
    bits = generate(a.generator, a.n, a.seed)
    if a.format == "ascii":
        open(a.out, "w").write("".join(map(str, bits.tolist())))
    else:
        open(a.out, "wb").write(np.packbits(bits).tobytes())
    print(f"Записано {a.n} бит в {a.out}")
    return 0


def main(argv=None):
    from .sources import GENERATORS
    p = argparse.ArgumentParser(prog="rngtest", description="Оценка случайности двоичных последовательностей")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="список батарей и тестов").set_defaults(fn=cmd_list)

    def common(sp):
        sp.add_argument("--battery", "-b", default="optimal",
                        help="nist | diehard | testu01 | fips | hac | optimal | iot")
        sp.add_argument("--tests", "-t", help="свой набор: id тестов через запятую")
        sp.add_argument("--alpha", "-a", type=float, default=0.01, help="уровень значимости")
        sp.add_argument("--format", "-f", default="auto", choices=["auto", "bin", "ascii", "hex"])

    r = sub.add_parser("run", help="протестировать файл / генератор")
    r.add_argument("file", nargs="?", help="файл с последовательностью (без него — stdin)")
    common(r)
    r.add_argument("--generator", "-g", choices=list(GENERATORS), help="встроенный генератор")
    r.add_argument("-n", type=int, default=1_000_000, help="длина (бит)")
    r.add_argument("--seed", type=int)
    r.add_argument("--subsequences", "-m", type=int, default=1, help="число подпоследовательностей (NIST)")
    r.add_argument("--param", action="append", help="параметр теста: nist.serial.m=8")
    r.add_argument("--report", "-o", action="append", help="файл отчёта .json/.csv/.txt (можно несколько)")
    r.add_argument("--quiet", "-q", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("stream", help="динамический режим: тестирование потока окнами")
    s.add_argument("file", nargs="?", default="-")
    common(s)
    s.set_defaults(battery="iot")
    s.add_argument("--window", "-w", type=int, default=1_000_000, help="размер окна (бит)")
    s.add_argument("--max-windows", type=int, default=0)
    s.set_defaults(fn=cmd_stream, format="bin")

    v = sub.add_parser("verify", help="сверка с эталонами NIST / dieharder")
    v.add_argument("-n", type=int, default=1_000_000)
    v.add_argument("--functional", action="store_true", help="также функциональная проверка всех тестов")
    v.set_defaults(fn=cmd_verify)

    o = sub.add_parser("optimize", help="экспериментальный подбор оптимального набора")
    o.add_argument("-n", type=int, default=1_000_000)
    o.add_argument("--reps", type=int, default=5)
    o.add_argument("--cost-weight", type=float, default=0.5)
    o.set_defaults(fn=cmd_optimize)

    b = sub.add_parser("bench", help="сравнение производительности Python / NumPy / C")
    b.add_argument("-n", type=int, default=1_000_000)
    b.add_argument("--no-python", action="store_true", help="не запускать медленные наивные версии")
    b.set_defaults(fn=cmd_bench)

    g = sub.add_parser("generate", help="записать последовательность встроенного генератора в файл")
    g.add_argument("generator", choices=list(GENERATORS))
    g.add_argument("out")
    g.add_argument("-n", type=int, default=1_000_000)
    g.add_argument("--seed", type=int)
    g.add_argument("--format", default="bin", choices=["bin", "ascii"])
    g.set_defaults(fn=cmd_generate)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
