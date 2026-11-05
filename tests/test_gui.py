"""Дымовой тест GUI: вкладки открываются, кнопки работают без исключений."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def _selected(at):
    return sum(cb.value for cb in at.checkbox)


def test_selection_buttons():
    at = _app()
    assert _selected(at) == 9                      # по умолчанию — рекомендуемый набор
    assert _selected(_click(at, "Очистить")) == 0
    assert _selected(_click(at, "Выбрать все")) == len(at.checkbox)
    assert _selected(_click(at, "Рекомендуемый набор")) == 9


def test_run_all_batteries():
    at = _click(_app(), "Запустить проверку")
    assert any("Гипотеза" in m.value for m in at.markdown)
    rows = at.dataframe[0].value
    assert list(rows["Батарея"])[-1] == "Пользовательский набор" and len(rows) == 6


def test_run_without_custom_set():
    at = _click(_click(_app(), "Очистить"), "Запустить проверку")
    assert len(at.dataframe[0].value) == 5
