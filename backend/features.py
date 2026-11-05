"""
features.py - shared by train.py and app.py.

WHY THIS FILE EXISTS
Training and live prediction must turn raw readings into model inputs in EXACTLY
the same way. If the two copies of this code ever differ, the model quietly gets
wrong inputs. So there is only one copy, and both files import it.

One "reading" = 11 numbers, always in this order:
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hall_0 .. hall_4
"""
import numpy as np

RAW_COLUMNS = [
    "mpu_acc_x", "mpu_acc_y", "mpu_acc_z",
    "mpu_gyro_x", "mpu_gyro_y", "mpu_gyro_z",
    "hall_0", "hall_1", "hall_2", "hall_3", "hall_4",
]

FEATURE_NAMES = [
    "hall_0", "hall_1", "hall_2", "hall_3", "hall_4",
    "acc_dir_x", "acc_dir_y", "acc_dir_z",
]

HALL_SCALE = 4095.0   # ESP32 analogRead is 0..4095
SMOOTH_MS = 60.0      # average the last 60 ms of readings (removes sensor noise)
MOTION_MS = 150.0     # how far back to look when measuring "is the hand moving"
FALLBACK_DT_MS = 50.0 # used only if a recording has no usable timestamps


def clean_ts(ts, fallback_dt_ms=FALLBACK_DT_MS):
    """Return a usable, non-decreasing time axis in milliseconds."""
    ts = np.asarray(ts, dtype=float)
    ok = ts.size > 0 and np.all(np.isfinite(ts)) and np.all(np.diff(ts) >= 0)
    if not ok:
        return np.arange(len(ts), dtype=float) * fallback_dt_ms
    return ts


def trailing_mean(ts, X, window_ms=SMOOTH_MS):
    """
    For every row i, the average of all rows in the time window (ts[i]-window, ts[i]].
    Uses TIME, not a row count, so it behaves the same at 20 Hz and at 100 Hz.
    """
    ts = clean_ts(ts)
    X = np.asarray(X, dtype=float)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    start = np.searchsorted(ts, ts - window_ms, side="right")
    idx = np.arange(len(ts))
    cnt = (idx + 1 - start).astype(float)
    return (cs[idx + 1] - cs[start]) / cnt[:, None]


def to_features(M):
    """
    M = smoothed readings, shape (n, 11)  ->  model inputs, shape (n, 8)
      5 finger values scaled to 0..1
      3 numbers for "which way is the wrist pointing" (gravity direction)
    The gyroscope is NOT a model input. It is used only to decide if the hand is moving.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    halls = M[:, 6:11] / HALL_SCALE
    acc = M[:, 0:3]
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    norm[norm < 1e-6] = 1.0
    return np.hstack([halls, acc / norm])


def gyro_magnitude(X):
    """Length of the gyro vector for each reading (how fast the hand is turning)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.linalg.norm(X[:, 3:6], axis=1)
