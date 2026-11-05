"""Графический интерфейс (Streamlit).

Запуск:  streamlit run app.py
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from rngtest import report
from rngtest.core import REGISTRY
from rngtest.runner import BATTERIES, OPTIMAL, run_battery
from rngtest.sources import GENERATORS, generate, parse_bits

st.set_page_config(page_title="Оценка ДСЧ", layout="wide")

STANDARD = ["nist", "diehard", "testu01", "fips", "hac"]
CUSTOM_NAME = "Пользовательский набор"
GROUPS = [BATTERIES[k]["name"] for k in STANDARD]          # порядок колонок на вкладке 1

INK, MUTED, LINE = "#1f1f1f", "#7a7a78", "#dcdcda"
C_BAR, C_OURS = "#b0b0ad", "#2e2e2e"
STATUS_RU = {"pass": "пройден", "fail": "отказ", "skipped": "пропущен", "error": "ошибка"}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1 {{ font-weight: 600; font-size: 1.7rem; letter-spacing: -0.01em; }}
  h2, h3, h4 {{ font-weight: 600; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 500; }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.8rem; }}
  .verdict {{ border: 1px solid {LINE}; border-left: 4px solid {INK}; background: #fff;
             padding: 0.8rem 1rem; margin: 0.4rem 0 1rem; }}
  .verdict.ok {{ border-left-color: #a8a8a5; }}
  .verdict b {{ font-weight: 600; }}
  .muted {{ color: {MUTED}; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)


def fmt_int(v):
    return f"{v:,}".replace(",", " ")


def muted(text):
    st.markdown(f"<div class='muted'>{text}</div>", unsafe_allow_html=True)


def cb_key(tid):
    return f"sel_{tid}"


def selected_tests():
    return [t for t in REGISTRY if st.session_state.get(cb_key(t), False)]


def set_selection(tests):
    for t in REGISTRY:
        st.session_state[cb_key(t)] = t in tests


if "init" not in st.session_state:          # по умолчанию отмечен рекомендуемый набор
    set_selection(OPTIMAL)
    st.session_state["init"] = True

st.title("Оценка датчиков случайных чисел")
tab_set, tab_run = st.tabs(["Набор тестов", "Запуск"])

# ================================================================== вкладка 1
with tab_set:
    c1, c2, c3, _ = st.columns([1.3, 1, 1, 2])
    c1.button("Рекомендуемый набор", type="primary", on_click=set_selection, args=(OPTIMAL,),
              width="stretch")
    c2.button("Выбрать все", on_click=set_selection, args=(list(REGISTRY),), width="stretch")
    c3.button("Очистить", on_click=set_selection, args=([],), width="stretch")
    muted(f"Выбрано тестов: {len(selected_tests())} из {len(REGISTRY)}. "
          "Рекомендуемый набор — 9 тестов, каждый из которых обнаруживает свой класс дефектов: "
          "смещение, корреляции, залипание, периодичность, линейную структуру.")
    st.write("")
    for col, group in zip(st.columns(len(GROUPS)), GROUPS):
        with col:
            st.markdown(f"**{group}**")
            for tid, spec in REGISTRY.items():
                if spec.battery == group:
                    name = spec.name.split(": ", 1)[-1]       # "FIPS: покер-тест" -> "покер-тест"
                    label = name[:1].upper() + name[1:] + (" *" if tid in OPTIMAL else "")
                    st.checkbox(label, key=cb_key(tid))
    muted("* — входит в рекомендуемый набор")

# ================================================================== вкладка 2
LENGTHS = {100_000: "100 тыс.", 500_000: "500 тыс.", 1_000_000: "1 млн", 2_500_000: "2,5 млн",
           5_000_000: "5 млн"}
GOOD = [k for k, g in GENERATORS.items() if g.good]
BAD = [k for k, g in GENERATORS.items() if not g.good]


def fmt_p(p):
    if p is None or p != p:
        return "—"
    return "< 0.000001" if p < 1e-6 else f"{p:.6f}"


with tab_run:
    left, right = st.columns([3, 2], gap="medium")

    # ---------------------------------------------------------------- данные
    with left.container(border=True):
        st.markdown("**Данные**")
        kind = st.radio("Источник", ["Генератор", "Файл", "Ввод вручную"], horizontal=True,
                        label_visibility="collapsed")
        src, info = None, ""
        if kind == "Генератор":
            g = st.selectbox("Генератор", GOOD + BAD,
                             format_func=lambda k: GENERATORS[k].name if GENERATORS[k].good
                             else f"{GENERATORS[k].name} (с дефектом)")
            c1, c2 = st.columns([3, 1])
            n = c1.select_slider("Длина, бит", list(LENGTHS), value=1_000_000, format_func=LENGTHS.get)
            seed = c2.number_input("Seed", value=1, step=1)
            src = lambda: (generate(g, int(n), int(seed)), GENERATORS[g].name)
            info = "эталонный генератор" if GENERATORS[g].good else f"дефект: {GENERATORS[g].defect}"
        elif kind == "Файл":
            f = st.file_uploader("Файл с последовательностью", label_visibility="collapsed")
            fmt = st.radio("Формат", ["auto", "bin", "ascii", "hex"], horizontal=True,
                           format_func={"auto": "Авто", "bin": "Двоичный", "ascii": "Текст 0/1",
                                        "hex": "HEX"}.get)
            if f is not None:
                bits_preview = parse_bits(f.getvalue(), fmt)
                src = lambda: (bits_preview, f.name)
                info = f"{fmt_int(bits_preview.size)} бит"
        else:
            txt = st.text_area("Последовательность", height=110, label_visibility="collapsed",
                               placeholder="Вставьте последовательность из 0 и 1")
            if txt.strip():
                bits_preview = parse_bits(txt.encode(), "ascii")
                src = lambda: (bits_preview, "ручной ввод")
                info = f"{fmt_int(bits_preview.size)} бит"
        if info:
            muted(info)

    # ---------------------------------------------------------------- параметры
    custom = selected_tests()
    with right.container(border=True):
        st.markdown("**Параметры**")
        alpha = st.radio("Уровень значимости α", [0.001, 0.01, 0.05], index=1, horizontal=True)
        st.markdown("<div class='muted' style='margin-top:.6rem'>Будут запущены</div>", unsafe_allow_html=True)
        lines = [f"{BATTERIES[k]['name']}<span style='float:right'>{len(BATTERIES[k]['tests'])}</span>"
                 for k in STANDARD]
        lines.append(f"<b>Ваш набор</b><span style='float:right'><b>{len(custom)}</b></span>" if custom
                     else "<span class='muted'>Ваш набор не выбран</span>")
        st.markdown("<div style='line-height:1.9'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

    run = st.button("Запустить проверку", type="primary", disabled=src is None, width="stretch")
    if src is None:
        muted("Выберите данные для проверки.")

    if run:
        bits, sname = src()
        plan = [(BATTERIES[k]["tests"], BATTERIES[k]["name"]) for k in STANDARD]
        if custom:
            plan.append((custom, CUSTOM_NAME))
        total = sum(len(t) for t, _ in plan)
        bar = st.progress(0.0)
        reports, done = [], 0
        for tests, name in plan:
            reports.append(run_battery(
                tests, bits, alpha=alpha, battery_name=name, source=sname,
                progress=lambda d, t, s, base=done, name=name: bar.progress(
                    min((base + d) / total, 1.0), f"{name} — {s}")))
            done += len(tests)
        bar.empty()
        st.session_state["reports"] = reports

    # ---------------------------------------------------------------- результаты
    reports = st.session_state.get("reports")
    if reports:
        st.divider()
        ours = next((r for r in reports if r.battery == CUSTOM_NAME), None)
        main = ours or reports[0]
        st.markdown(f"<div class='verdict{' ok' if main.ok else ''}'><b>{main.verdict}</b><br>"
                    f"<span class='muted'>{main.source} · {fmt_int(main.n_bits)} бит · α = {main.alpha} · "
                    f"по набору «{main.battery}»</span></div>", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("p-значение набора", fmt_p(main.p_value))
        m[1].metric("Пройдено тестов", f"{main.n_pass} из {len(main.counted)}")
        m[2].metric("Время", f"{main.total_time_s:.3f} с")
        m[3].metric("Пик памяти", f"{main.peak_mem_bytes / 2**20:.1f} МБ")

        st.subheader("Сводка по батареям")
        summary = pd.DataFrame([{
            "Батарея": r.battery, "Тестов": len(r.results), "Пройдено": f"{r.n_pass}/{len(r.counted)}",
            "p-значение": r.p_value, "Вердикт": "случайна" if r.ok else "не случайна",
            "Время, с": r.total_time_s, "Мбит/с": r.throughput_mbit_s,
            "Память, МБ": r.peak_mem_bytes / 2**20} for r in reports])
        styled = summary.style.format({"p-значение": fmt_p, "Время, с": "{:.3f}", "Мбит/с": "{:.1f}",
                                       "Память, МБ": "{:.1f}"}).apply(
            lambda row: ["font-weight: 600; background-color: #f0f0ef" if row["Батарея"] == CUSTOM_NAME
                         else "" for _ in row], axis=1)
        st.dataframe(styled, hide_index=True, width="stretch")
        muted("p-значение батареи — 1 − (1 − p_min)^T, где T — число тестов в батарее (поправка Шидака). "
              "Батарея отвергает гипотезу о случайности, если p < α.")

        summary["Набор"] = summary["Батарея"].map(lambda b: "наш" if b == CUSTOM_NAME else "стандартный")
        colors = alt.Scale(domain=["стандартный", "наш"], range=[C_BAR, C_OURS])
        charts = (("p-значение", "p-значение", alt.Scale(domain=[0, 1])),
                  ("Время, с", "Время, с", alt.Undefined), ("Память, МБ", "Пик памяти, МБ", alt.Undefined))
        for col, (field, title, scale) in zip(st.columns(3), charts):
            ch = alt.Chart(summary).mark_bar(cornerRadiusEnd=2, height=14).encode(
                y=alt.Y("Батарея:N", sort=None, title=None, axis=alt.Axis(labelLimit=180)),
                x=alt.X(f"{field}:Q", title=title, scale=scale),
                color=alt.Color("Набор:N", scale=colors, legend=None),
                tooltip=["Батарея", alt.Tooltip(f"{field}:Q", format=".4f")])
            if field == "p-значение":
                ch = alt.layer(ch, alt.Chart(pd.DataFrame({"a": [main.alpha]})).mark_rule(
                    color=INK, strokeDash=[3, 3]).encode(x="a:Q"))
            col.altair_chart(ch.properties(height=alt.Step(28)), use_container_width=True)

        st.subheader("Результаты тестов")
        for r in reports:
            with st.expander(f"{r.battery} — {r.n_pass}/{len(r.counted)}, p = {fmt_p(r.p_value)}",
                             expanded=r is ours):
                st.dataframe(pd.DataFrame([{
                    "Тест": t.name, "Статус": STATUS_RU[t.status], "p-значение": t.p_value,
                    "Статистика": t.statistic, "Время, мс": t.time_s * 1000,
                    "Память, КБ": t.peak_mem_bytes / 1024, "Примечание": t.message} for t in r.results]),
                    hide_index=True, width="stretch",
                    column_config={"p-значение": st.column_config.NumberColumn(format="%.6f"),
                                   "Статистика": st.column_config.NumberColumn(format="%.4g"),
                                   "Время, мс": st.column_config.NumberColumn(format="%.2f"),
                                   "Память, КБ": st.column_config.NumberColumn(format="%.0f")})

        full = {"source": main.source, "n_bits": main.n_bits, "alpha": main.alpha,
                "batteries": [json.loads(report.to_json(r)) for r in reports]}
        st.markdown("<div class='muted' style='margin-top:.5rem'>Скачать отчёт</div>", unsafe_allow_html=True)
        c = st.columns([1, 1, 1, 3])
        c[0].download_button("JSON", json.dumps(full, ensure_ascii=False, indent=2), "report.json",
                             "application/json", width="stretch")
        c[1].download_button("CSV", "".join(report.to_csv(r) if i == 0 else report.to_csv(r).split("\n", 1)[1]
                                            for i, r in enumerate(reports)),
                             "report.csv", "text/csv", width="stretch")
        c[2].download_button("TXT", "\n\n".join(report.to_txt(r) for r in reports), "report.txt",
                             "text/plain", width="stretch")
