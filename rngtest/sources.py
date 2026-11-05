"""Источники бинарных последовательностей: файлы, потоки и встроенные генераторы.

Встроенные генераторы делятся на "хорошие" (эталон случайности) и "дефектные"
(с известным типом изъяна). Дефектные нужны для функциональной проверки тестов
(ТЗ п. 6.5.1) и для экспериментального подбора оптимального набора тестов.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterator, Optional

import numpy as np

from .constants import e_bits, pi_bits, sqrt_bits


# ------------------------------------------------------------------ файлы
def parse_bits(data: bytes, fmt: str = "auto") -> np.ndarray:
    """Разобрать содержимое файла.

    fmt: "bin"   — сырые байты, по 8 бит (старший бит первым);
         "ascii" — текст из символов '0'/'1' (прочие символы игнорируются);
         "hex"   — шестнадцатеричный текст;
         "auto"  — ascii, если файл состоит только из 0/1/пробелов, иначе bin.
    """
    if fmt == "auto":
        sample = data[:65536]
        fmt = "ascii" if sample and set(sample) <= set(b"01 \r\n\t") else "bin"
    if fmt == "bin":
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    if fmt == "ascii":
        a = np.frombuffer(data, dtype=np.uint8)
        a = a[(a == 48) | (a == 49)]
        return (a - 48).astype(np.uint8)
    if fmt == "hex":
        txt = "".join(ch for ch in data.decode("ascii", "ignore") if ch in "0123456789abcdefABCDEF")
        return np.unpackbits(np.frombuffer(bytes.fromhex(txt[: len(txt) // 2 * 2]), dtype=np.uint8))
    raise ValueError(f"Неизвестный формат: {fmt}")


def load_file(path, fmt: str = "auto", max_bits: Optional[int] = None) -> np.ndarray:
    data = Path(path).read_bytes()
    bits = parse_bits(data, fmt)
    return bits[:max_bits] if max_bits else bits


def stream_chunks(fh: BinaryIO, chunk_bits: int, fmt: str = "bin") -> Iterator[np.ndarray]:
    """Динамический режим: читать поток (stdin, устройство, сокет) окнами по chunk_bits."""
    buf = np.zeros(0, dtype=np.uint8)
    read_size = max(chunk_bits // 8, 4096) if fmt == "bin" else max(chunk_bits, 4096)
    while True:
        data = fh.read(read_size)
        if not data:
            break
        buf = np.concatenate([buf, parse_bits(data, fmt)])
        while buf.size >= chunk_bits:
            yield buf[:chunk_bits]
            buf = buf[chunk_bits:]


# ------------------------------------------------------------------ генераторы
def _bytes_to_bits(b: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:n]


def _words_to_bits(w: np.ndarray, n: int, width: int = 32) -> np.ndarray:
    w = w.astype(f">u{width // 8}")
    return np.unpackbits(w.view(np.uint8))[:n]


def gen_urandom(n, seed=None):
    return _bytes_to_bits(os.urandom((n + 7) // 8), n)


def gen_pcg64(n, seed=None):
    return _bytes_to_bits(np.random.Generator(np.random.PCG64(seed)).bytes((n + 7) // 8), n)


def gen_mt19937(n, seed=None):
    r = random.Random(seed)
    return _bytes_to_bits(r.getrandbits(((n + 7) // 8) * 8).to_bytes((n + 7) // 8, "big"), n)


def gen_sha256_ctr(n, seed=None):
    """Хеш-DRBG: SHA-256(seed || счётчик)."""
    key = str(seed).encode()
    out = bytearray()
    i = 0
    while len(out) * 8 < n:
        out += hashlib.sha256(key + i.to_bytes(8, "big")).digest()
        i += 1
    return _bytes_to_bits(bytes(out), n)


def _rng(seed):
    return np.random.default_rng(seed)


def gen_biased(n, seed=None, p=0.51):
    return (_rng(seed).random(n) < p).astype(np.uint8)


def gen_markov(n, seed=None, stay=0.52):
    """Марковская цепь: бит повторяет предыдущий с вероятностью stay (корреляция соседних бит)."""
    flips = (_rng(seed).random(n) >= stay).astype(np.uint8)
    flips[0] = 1
    return (np.cumsum(flips) & 1).astype(np.uint8)


def gen_periodic(n, seed=None, period=4096):
    """Случайный блок длиной period, повторённый много раз."""
    block = (_rng(seed).random(period) < 0.5).astype(np.uint8)
    return np.tile(block, n // period + 1)[:n]


def gen_lfsr(n, seed=None, taps=(31, 3)):
    """РСЛОС со степенью 31 (многочлен x^31 + x^3 + 1): идеальные частоты, низкая линейная сложность."""
    L, t = taps
    s = np.zeros(n + L, dtype=np.uint8)
    s[:L] = (_rng(seed).random(L) < 0.5)
    s[0] = 1
    step = L - t
    i = L
    while i < n + L:
        j = min(i + step, n + L)
        s[i:j] = s[i - L:j - L] ^ s[i - L + t:j - L + t]
        i = j
    return s[L:]


def _lcg_words(n_words, a, c, m, seed, shift=0, bits_out=32):
    x = (seed or 12345) % m
    out = np.empty(n_words, dtype=np.uint64)
    for i in range(n_words):
        x = (a * x + c) % m
        out[i] = (x >> shift) & ((1 << bits_out) - 1)
    return out


def gen_randu(n, seed=None):
    """RANDU (IBM, a=65539, m=2^31) — печально известный плохой ЛКГ."""
    nw = (n + 30) // 31
    w = _lcg_words(nw, 65539, 0, 2 ** 31, (seed or 1) | 1, 0, 31)
    return np.unpackbits((w << np.uint64(1)).astype(">u4").view(np.uint8)).reshape(-1, 32)[:, :31].ravel()[:n]


def gen_lcg_low(n, seed=None):
    """ЛКГ по модулю 2^32 (как rand() в старых libc), берутся младшие 8 бит — они короткопериодичны."""
    nw = (n + 7) // 8
    w = _lcg_words(nw, 1103515245, 12345, 2 ** 32, seed or 1, 0, 8)
    return np.unpackbits(w.astype(np.uint8))[:n]


def gen_xorshift32(n, seed=None):
    """Xorshift32 (Марсалья): хорошие частоты, но линеен над GF(2)."""
    x = (seed or 2463534242) & 0xFFFFFFFF or 1
    nw = (n + 31) // 32
    out = np.empty(nw, dtype=np.uint32)
    for i in range(nw):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x
    return _words_to_bits(out, n)


def gen_sticky_bit(n, seed=None, pos=7):
    """Каждый 8-й бит залип в 0 (дефект аппаратного шумового источника)."""
    b = gen_pcg64(n, seed).copy()
    b[pos::8] = 0
    return b


def gen_drift(n, seed=None, amp=0.03, period=100000):
    """Медленный дрейф смещения: P(1) = 0.5 + amp·sin(2πi/period)."""
    i = np.arange(n)
    p = 0.5 + amp * np.sin(2 * np.pi * i / period)
    return (_rng(seed).random(n) < p).astype(np.uint8)


def gen_counter(n, seed=None):
    """Двоичный счётчик (32-битные числа 0, 1, 2, ...) — заведомо неслучайная последовательность."""
    return _words_to_bits(np.arange((n + 31) // 32, dtype=np.uint32) + np.uint32(seed or 0), n)


def gen_const_e(n, seed=None):
    return e_bits(n)


def gen_const_pi(n, seed=None):
    return pi_bits(n)


@dataclass
class Generator:
    key: str
    name: str
    func: Callable
    good: bool
    defect: str = ""


GENERATORS: Dict[str, Generator] = {g.key: g for g in [
    Generator("urandom", "os.urandom (системный КСГПСЧ)", gen_urandom, True),
    Generator("pcg64", "NumPy PCG64", gen_pcg64, True),
    Generator("mt19937", "Mersenne Twister (random)", gen_mt19937, True),
    Generator("sha256", "SHA-256 в режиме счётчика", gen_sha256_ctr, True),
    Generator("e", "Двоичное разложение e (эталон NIST)", gen_const_e, True),
    Generator("pi", "Двоичное разложение π", gen_const_pi, True),
    Generator("biased", "Смещённый (P(1)=0.51)", gen_biased, False, "смещение"),
    Generator("markov", "Марковский (P(повтор)=0.52)", gen_markov, False, "корреляция соседних бит"),
    Generator("periodic", "Периодический (период 4096)", gen_periodic, False, "периодичность"),
    Generator("lfsr", "РСЛОС x^31+x^3+1", gen_lfsr, False, "линейная структура"),
    Generator("xorshift32", "Xorshift32", gen_xorshift32, False, "линейность над GF(2)"),
    Generator("randu", "RANDU (ЛКГ IBM)", gen_randu, False, "плохой ЛКГ"),
    Generator("lcg_low", "ЛКГ, младшие биты", gen_lcg_low, False, "короткий период младших бит"),
    Generator("sticky", "Залипший бит (каждый 8-й = 0)", gen_sticky_bit, False, "залипание бита"),
    Generator("drift", "Дрейф смещения (синус)", gen_drift, False, "нестационарность"),
    Generator("counter", "Двоичный счётчик", gen_counter, False, "полная детерминированность"),
]}


def generate(key: str, n: int, seed: Optional[int] = None) -> np.ndarray:
    g = GENERATORS[key]
    return np.ascontiguousarray(g.func(n, seed), dtype=np.uint8)
