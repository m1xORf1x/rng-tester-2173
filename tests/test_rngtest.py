"""Автотесты комплекса (запуск: pytest -q)."""
import json

import numpy as np
import pytest

from rngtest import BATTERIES, REGISTRY, generate, parse_bits, run_battery, run_test
from rngtest import native, report
from rngtest.mathutil import gf2_rank_batch, rank_probabilities
from rngtest.tests_nist import aperiodic_templates, berlekamp_massey_batch
from rngtest.verify import functional_check, verify_nist

N = 1_000_000


@pytest.fixture(scope="module")
def good_bits():
    return generate("pcg64", N, 123)


# ---------------------------------------------------------------- эталон NIST
def test_nist_reference_values():
    rows = verify_nist()
    bad = [(r.test_id, r.expected, r.actual) for r in rows if not r.ok]
    assert not bad, bad


def test_aperiodic_templates_count():
    assert len(aperiodic_templates(9)) == 148  # как в sts-2.1.2


def test_rank_probabilities_32():
    p = rank_probabilities(32, 32)
    assert abs(p[32] - 0.2888) < 1e-4 and abs(p[31] - 0.5776) < 1e-4


def test_gf2_rank():
    eye = np.array([[1 << i for i in range(8)]], dtype=np.uint64)
    assert gf2_rank_batch(eye, 8)[0] == 8
    dup = np.array([[3, 3, 5, 6]], dtype=np.uint64)  # 3 ^ 5 = 6
    assert gf2_rank_batch(dup, 3)[0] == 2


def test_berlekamp_massey_numpy_matches_c():
    rng = np.random.default_rng(0)
    S = rng.integers(0, 2, (20, 200), dtype=np.uint8)
    py = berlekamp_massey_batch(S)
    if native.available():
        assert np.array_equal(py, native.lc_blocks(S.ravel(), 20, 200))
    # РСЛОС степени 31 имеет линейную сложность 31
    lfsr = generate("lfsr", 500, 1)
    assert berlekamp_massey_batch(lfsr.reshape(1, -1))[0] == 31


def test_native_packed_ops():
    if not native.available():
        pytest.skip("нет компилятора C")
    b = generate("pcg64", 100_003, 5)
    p = native.pack(b)
    assert native.packed_popcount(p, b.size) == int(b.sum())
    for d in (1, 3, 8, 13, 32):
        assert native.packed_autocorr(p, b.size, d) == int(np.count_nonzero(b[:-d] != b[d:]))


# ---------------------------------------------------------------- функциональные
def test_every_test_passes_random_and_rejects_zeros():
    rows = functional_check(N)
    bad = [r for r in rows if not r["ok"]]
    assert not bad, bad


@pytest.mark.parametrize("gen,battery,expect_ok", [
    ("pcg64", "optimal", True),
    ("biased", "optimal", False),
    ("markov", "optimal", False),
    ("lfsr", "optimal", False),
    ("xorshift32", "optimal", False),
    ("periodic", "optimal", False),
    ("counter", "iot", False),
])
def test_battery_verdicts(gen, battery, expect_ok):
    rep = run_battery(battery, generate(gen, N, 7), measure_memory=False)
    assert rep.ok == expect_ok, (gen, rep.verdict, [(r.test_id, r.p_value) for r in rep.results])


def test_all_batteries_run(good_bits):
    for key in BATTERIES:
        rep = run_battery(key, good_bits)
        assert not [r for r in rep.results if r.status == "error"]
        assert rep.total_time_s > 0 and rep.peak_mem_bytes > 0


def test_short_sequence_is_skipped_not_crashing():
    rep = run_battery("nist", generate("pcg64", 500, 1))
    assert all(r.status in ("pass", "fail", "skipped") for r in rep.results)
    assert rep.n_skipped > 0


def test_second_level_analysis(good_bits):
    rep = run_battery(["nist.frequency", "nist.runs"], good_bits, subsequences=20)
    assert len(rep.second_level) == 2 and rep.ok


def test_custom_params():
    r = run_test("nist.block_frequency", generate("pcg64", 10_000, 1), params={"M": 100})
    assert r.params["M"] == 100 and r.details["N"] == 100


# ---------------------------------------------------------------- ввод/вывод
def test_parse_formats():
    assert parse_bits(b"0101 11", "ascii").tolist() == [0, 1, 0, 1, 1, 1]
    assert parse_bits(b"\xf0", "bin").tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert parse_bits(b"0f", "hex").tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert parse_bits(b"0011", "auto").tolist() == [0, 0, 1, 1]


def test_reports(good_bits):
    rep = run_battery("fips", good_bits)
    d = json.loads(report.to_json(rep))
    assert d["passed"] == 4 and len(d["results"]) == 4
    assert report.to_csv(rep).count("\n") == 5
    assert "ВЕРДИКТ" in report.to_txt(rep)


def test_registry_metadata():
    for tid, s in REGISTRY.items():
        assert s.name and s.battery and s.min_bits > 0 and s.description, tid
