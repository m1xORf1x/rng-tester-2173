"""Сохранение отчётов о тестировании: JSON, CSV, TXT (ТЗ п. 6.3)."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from .runner import BatteryReport


def _fmt_p(p):
    return "—" if p is None else f"{p:.6f}"


def to_json(rep: BatteryReport) -> str:
    d = rep.to_dict()
    d["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return json.dumps(d, ensure_ascii=False, indent=2, default=float)


CSV_FIELDS = ["test_id", "name", "battery", "status", "p_value", "p_values", "statistic",
              "n_bits", "time_s", "peak_mem_bytes", "throughput_mbit_s", "message"]


def to_csv(rep: BatteryReport) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rep.results:
        d = r.to_dict()
        d["p_values"] = ";".join(f"{p:.6g}" for p in r.p_values)
        w.writerow(d)
    return buf.getvalue()


def to_txt(rep: BatteryReport) -> str:
    L = []
    L.append("=" * 96)
    L.append(f" ОТЧЁТ О ТЕСТИРОВАНИИ ПОСЛЕДОВАТЕЛЬНОСТИ   ({datetime.now():%Y-%m-%d %H:%M:%S})")
    L.append("=" * 96)
    L.append(f" Источник:            {rep.source or '—'}")
    L.append(f" Батарея:             {rep.battery}")
    L.append(f" Длина:               {rep.n_bits:,} бит".replace(",", " ")
             + (f" × {rep.subsequences} подпоследовательностей" if rep.subsequences > 1 else ""))
    L.append(f" Уровень значимости:  α = {rep.alpha}")
    L.append(f" Время:               {rep.total_time_s:.3f} с   "
             f"Скорость: {rep.throughput_mbit_s or 0:.2f} Мбит/с   "
             f"Пик памяти: {rep.peak_mem_bytes / 2**20:.2f} МБ")
    L.append(f" Итог:                пройдено {rep.n_pass}, не пройдено {rep.n_fail}, "
             f"пропущено {rep.n_skipped}")
    L.append(f" p-значение батареи:  {_fmt_p(rep.p_value)}  (Шидак: 1 − (1 − p_min)^T)")
    L.append(f" ВЕРДИКТ:             {rep.verdict}")
    L.append("-" * 96)
    L.append(f" {'Тест':44s} {'Статус':8s} {'p-значение':>11s} {'Статистика':>12s} "
             f"{'Время, мс':>10s} {'Память, КБ':>11s}")
    L.append("-" * 96)
    status_ru = {"pass": "ПРОЙДЕН", "fail": "ОТКАЗ", "skipped": "ПРОПУСК", "error": "ОШИБКА"}
    for r in rep.results:
        st = "—" if r.statistic is None else f"{r.statistic:.4g}"
        L.append(f" {r.name[:44]:44s} {status_ru[r.status]:8s} {_fmt_p(r.p_value):>11s} {st:>12s} "
                 f"{r.time_s * 1000:10.1f} {r.peak_mem_bytes / 1024:11.0f}")
        if r.message:
            L.append(f"     ↳ {r.message}")
        if len(r.p_values) > 1:
            L.append(f"     ↳ p-значений: {len(r.p_values)}, min = {min(r.p_values):.6f} "
                     f"(итоговое p с поправкой Шидака)")
    if rep.second_level:
        L.append("-" * 96)
        L.append(" Анализ второго уровня (NIST SP 800-22, разд. 4.2)")
        L.append(f" {'Тест':44s} {'Доля':>8s} {'Мин.доля':>9s} {'P-равном.':>10s}  Итог")
        for s in rep.second_level:
            L.append(f" {s.name[:44]:44s} {s.proportion:8.4f} {s.proportion_min:9.4f} "
                     f"{s.uniformity_p:10.6f}  {'OK' if s.passed else 'ОТКАЗ'}")
    L.append("=" * 96)
    return "\n".join(L)


def save(rep: BatteryReport, path: str) -> None:
    ext = path.rsplit(".", 1)[-1].lower()
    text = {"json": to_json, "csv": to_csv}.get(ext, to_txt)(rep)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
