"""Загрузка C-ускорителя через ctypes.

Библиотека собирается автоматически при первом обращении (нужен компилятор
cc/gcc/clang). Если сборка невозможна — все тесты работают на NumPy,
просто медленнее. Принудительно отключить: переменная окружения RNGTEST_NO_NATIVE=1.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "fastrng.c"
_LIB_NAME = "fastrng.dll" if sys.platform == "win32" else "libfastrng.so"
_LIB_PATH = _HERE / _LIB_NAME
_lib = None
_tried = False
build_log = ""


def build(force: bool = False) -> bool:
    """Скомпилировать fastrng.c. Возвращает True при успехе."""
    global build_log
    if _LIB_PATH.exists() and not force and _LIB_PATH.stat().st_mtime >= _SRC.stat().st_mtime:
        return True
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        build_log = "Компилятор C не найден"
        return False
    cmd = [cc, "-O3", "-shared", "-fPIC", "-std=c99", str(_SRC), "-o", str(_LIB_PATH)]
    if platform.machine().lower() in ("x86_64", "amd64"):
        cmd.insert(2, "-march=native")  # AVX2/AVX-512 при наличии; на ARM NEON включён по умолчанию
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:  # pragma: no cover
        build_log = str(e)
        return False
    build_log = " ".join(cmd) + "\n" + r.stdout + r.stderr
    return r.returncode == 0


def get_lib() -> Optional[ctypes.CDLL]:
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    if os.environ.get("RNGTEST_NO_NATIVE"):
        return None
    try:
        if not build():
            return None
        lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError:
        return None
    u8p = ctypes.POINTER(ctypes.c_uint8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.bm_linear_complexity.argtypes = [u8p, ctypes.c_int]
    lib.bm_linear_complexity.restype = ctypes.c_int
    lib.lc_blocks.argtypes = [u8p, ctypes.c_int, ctypes.c_int, i32p]
    lib.lc_blocks.restype = None
    lib.packed_popcount.argtypes = [u8p, ctypes.c_int64]
    lib.packed_popcount.restype = ctypes.c_int64
    lib.packed_autocorr.argtypes = [u8p, ctypes.c_int64, ctypes.c_int64]
    lib.packed_autocorr.restype = ctypes.c_int64
    lib.longest_runs.argtypes = [u8p, ctypes.c_int64, ctypes.c_int64, i32p]
    lib.longest_runs.restype = None
    lib.pack_bits.argtypes = [u8p, ctypes.c_int64, u8p]
    lib.pack_bits.restype = None
    _lib = lib
    return _lib


def available() -> bool:
    return get_lib() is not None


def _u8(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _i32(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def lc_blocks(bits: np.ndarray, N: int, M: int) -> np.ndarray:
    lib = get_lib()
    b = np.ascontiguousarray(bits[: N * M], dtype=np.uint8)
    out = np.zeros(N, dtype=np.int32)
    lib.lc_blocks(_u8(b), N, M, _i32(out))
    return out


def longest_runs(bits: np.ndarray, N: int, M: int) -> np.ndarray:
    lib = get_lib()
    b = np.ascontiguousarray(bits[: N * M], dtype=np.uint8)
    out = np.zeros(N, dtype=np.int32)
    lib.longest_runs(_u8(b), N, M, _i32(out))
    return out


def pack(bits: np.ndarray) -> np.ndarray:
    lib = get_lib()
    if lib is None:
        return np.packbits(bits)
    b = np.ascontiguousarray(bits, dtype=np.uint8)
    out = np.zeros((b.size + 7) // 8 + 8, dtype=np.uint8)
    lib.pack_bits(_u8(b), b.size, _u8(out))
    return out


def packed_popcount(packed: np.ndarray, nbits: int) -> int:
    return int(get_lib().packed_popcount(_u8(np.ascontiguousarray(packed)), nbits))


def packed_autocorr(packed: np.ndarray, nbits: int, d: int) -> int:
    p = np.ascontiguousarray(packed)
    if p.size < (nbits + 7) // 8 + 2:  # запас для чтения p[k+db+1]
        p = np.concatenate([p, np.zeros(2, np.uint8)])
    return int(get_lib().packed_autocorr(_u8(p), nbits, d))
