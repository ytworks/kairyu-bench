"""Local trusted BF16 merge statistics; C++ built with no fast-math, FP contraction off."""

from pathlib import Path
import ctypes
import os
import numpy as np

LIBRARY = Path(
    os.environ.get(
        "KAIRYU_BENCH_BF16_LORA_LIBRARY",
        str(Path(__file__).with_name("bf16_lora_kernel.so")),
    )
).resolve()
LIB = ctypes.CDLL(str(LIBRARY))
P16 = ctypes.POINTER(ctypes.c_uint16)
PF = ctypes.POINTER(ctypes.c_float)
PD = ctypes.POINTER(ctypes.c_double)
P64 = ctypes.POINTER(ctypes.c_uint64)
LIB.audit5.argtypes = [P16, PF, ctypes.c_uint64, PF, PD, P64]
LIB.audit5.restype = ctypes.c_int
LIB.merge1.argtypes = [P16, PF, ctypes.c_uint64, ctypes.c_float, P16, PD, P64]
LIB.merge1.restype = ctypes.c_int


def audit5(original, delta, coefficients):
    original = np.ascontiguousarray(original, dtype=np.uint16)
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float32)
    if (
        original.shape != delta.shape
        or coefficients.shape != (5,)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("kernel shapes")
    output = np.zeros(17, dtype=np.float64)
    counts = np.zeros(16, dtype=np.uint64)
    bad = LIB.audit5(
        original.ctypes.data_as(P16),
        delta.ctypes.data_as(PF),
        original.size,
        coefficients.ctypes.data_as(PF),
        output.ctypes.data_as(PD),
        counts.ctypes.data_as(P64),
    )
    if bad:
        raise ValueError("nonfinite BF16 merge inputs/outputs")
    return output, counts


def merge1(original, delta, coefficient):
    original = np.ascontiguousarray(original, dtype=np.uint16)
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    if (
        original.shape != delta.shape
        or not np.isfinite(coefficient)
        or not np.isfinite(np.float32(coefficient))
    ):
        raise ValueError("kernel shapes/coefficient")
    target = np.empty_like(original)
    stats = np.zeros(3, dtype=np.float64)
    counts = np.zeros(2, dtype=np.uint64)
    bad = LIB.merge1(
        original.ctypes.data_as(P16),
        delta.ctypes.data_as(PF),
        original.size,
        float(coefficient),
        target.ctypes.data_as(P16),
        stats.ctypes.data_as(PD),
        counts.ctypes.data_as(P64),
    )
    if bad:
        raise ValueError("nonfinite BF16 merge inputs/outputs")
    return target, stats, counts
