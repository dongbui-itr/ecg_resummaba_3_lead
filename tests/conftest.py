"""Shared fixtures. Nothing here reads the real dataset - the tests run anywhere."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecgr import xla                                   # noqa: E402

xla.ensure_libdevice(verbose=False)                    # before any tensorflow import


@pytest.fixture(scope='session')
def synthetic_record():
    """A 30 s, 3-lead, 250 Hz record with beats at a known 0.8 s R-R, plus those positions.

    Lead 0 carries the tallest QRS, leads 1 and 2 scaled copies with their own baseline
    wander, which is the shape of a real portal strip: one lead the reviewer worked on and
    two more of the same heart.
    """
    fs, seconds = 250, 30
    n = fs * seconds
    rng = np.random.default_rng(7)
    peaks = np.arange(int(0.5 * fs), n - fs, int(0.8 * fs))

    signal = np.zeros((n, 3))
    qrs = np.hanning(11) * 3.0
    for p in peaks:
        signal[p - 5:p + 6, 0] += qrs
    signal[:, 1] = signal[:, 0] * 0.6
    signal[:, 2] = signal[:, 0] * 0.3
    t = np.arange(n) / fs
    for c, freq in enumerate((0.15, 0.2, 0.25)):
        signal[:, c] += 0.3 * np.sin(2 * np.pi * freq * t)
    signal += rng.normal(0, 0.02, signal.shape)
    return signal.astype(np.float32), peaks
