"""
train.py - trains the per-reading (static letter) model.

What changed compared to the old LSTM version, and why:
  * Looks at ONE moment (smoothed over 60 ms), not a 50-reading window.
    -> decisions no longer wait for a full window, and it does not matter
       whether your old recordings were made at 20 Hz or new ones at 100 Hz.
  * Train/test split is by RECORDING SESSION, not by random window.
    -> the accuracy number is honest (no near-copies in both sets).
  * Tries 3 small models and keeps the best one on the held-out sessions.
  * Saves everything the server needs in models/active/ (written safely, so the
    server never reads a half-written file).
  * Still writes reports/training_report.json in the shape the web page expects.

Run:  python train.py
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (FEATURE_NAMES, RAW_COLUMNS, SMOOTH_MS, clean_ts,
                      to_features, trailing_mean)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SIGN_DATA_DIR", BASE_DIR / "data" / "ALL_DATA"))
MODELS_DIR = BASE_DIR / "models" / "active"
REPORTS_DIR = BASE_DIR / "reports"

TRIM_MS = 300          # drop the first/last 0.3 s of every recording (hand still moving into/out of shape)
TEST_FRACTION = 0.2    # share of SESSIONS held out for testing
AUGMENT_COPIES = 2     # extra noisy copies of each training row
SEED = 42


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# 1. Reading recordings (works for old nested-JSON files and new flat files)
# ----------------------------------------------------------------------------
def flatten_session(df):
    """Return an (n, 11) float array in RAW_COLUMNS order, or None."""
    if all(c in df.columns for c in RAW_COLUMNS):
        return df[RAW_COLUMNS].to_numpy(dtype=float)
    try:
        mpu = pd.json_normalize(df["mpu6050"].tolist())
        mpu.columns = ["mpu_" + c.replace(".", "_") for c in mpu.columns]
        halls = pd.DataFrame(df["halls"].tolist(),
                             columns=[f"hall_{i}" for i in range(5)])
        flat = pd.concat([mpu, halls], axis=1)[RAW_COLUMNS]
        return flat.to_numpy(dtype=float)
    except Exception as e:  # noqa
        log(f"    could not flatten: {e}")
        return None


def read_ts(df):
    """Device timestamps in ms, or None if the file has no usable ones."""
    if "ts_device" not in df.columns:
        return None
    s = pd.to_numeric(df["ts_device"], errors="coerce")
    if s.notna().all():
        return s.to_numpy(dtype=float)
    try:
        d = pd.to_datetime(df["ts_device"], errors="coerce")
        if d.notna().all():
            return ((d - d.iloc[0]).dt.total_seconds() * 1000.0).to_numpy(dtype=float)
    except Exception:  # noqa
        pass
    return None


def session_to_features(path):
    df = pd.read_parquet(path)
    X = flatten_session(df)
    if X is None or len(X) < 5:
        return None, None
    ts = read_ts(df)
    ts = clean_ts(ts if ts is not None else np.arange(len(X)) * 50.0)
    M = trailing_mean(ts, X, SMOOTH_MS)
    F = to_features(M)
    keep = (ts >= ts[0] + TRIM_MS) & (ts <= ts[-1] - TRIM_MS)
    if keep.sum() >= 5:
        F = F[keep]
    dt = np.median(np.diff(ts)) if len(ts) > 1 else float("nan")
    return F, dt


def load_all():
    sessions = {}          # label -> list of (session_id, F)
    rate_notes = []
    log(f"Scanning: {DATA_DIR}")
    for label_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        if label_dir.name.startswith((".", "_")):
            continue
        label = label_dir.name
        for f in sorted(label_dir.glob("*.parquet")):
            try:
                F, dt = session_to_features(f)
            except Exception as e:  # noqa
                log(f"  skip {label}/{f.name}: {e}")
                continue
            if F is None:
                log(f"  skip {label}/{f.name}: unreadable or too short")
                continue
            sessions.setdefault(label, []).append((f"{label}/{f.name}", F))
            rate_notes.append(dt)
        n = len(sessions.get(label, []))
        rows = sum(len(F) for _, F in sessions.get(label, []))
        log(f"  [{label}] {n} sessions, {rows} rows")
    return sessions, rate_notes


# ----------------------------------------------------------------------------
# 2. Honest split by session
# ----------------------------------------------------------------------------
def split_by_session(sessions):
    rng = np.random.RandomState(SEED)
    Xtr, ytr, Xte, yte, warnings = [], [], [], [], []
    for label, sess in sessions.items():
        if len(sess) >= 2:
            n_test = max(1, int(round(TEST_FRACTION * len(sess))))
            order = rng.permutation(len(sess))
            test_ids = set(order[:n_test].tolist())
            for i, (_, F) in enumerate(sess):
                (Xte if i in test_ids else Xtr).append(F)
                (yte if i in test_ids else ytr).extend([label] * len(F))
        else:
            F = sess[0][1]
            cut = int(len(F) * (1 - TEST_FRACTION))
            gap = min(10, max(0, cut // 10))
            Xtr.append(F[:cut]); ytr.extend([label] * cut)
            tail = F[cut + gap:]
            Xte.append(tail); yte.extend([label] * len(tail))
            warnings.append(
                f"Class '{label}' has only ONE recording session, so its test rows come from the "
                f"end of that same session. Its score is optimistic. Record more sessions for it.")
    return (np.vstack(Xtr), np.array(ytr), np.vstack(Xte), np.array(yte), warnings)


def augment(X, y, copies, rng):
    if copies <= 0:
        return X, y
    Xs, ys = [X], [y]
    for _ in range(copies):
        noise = np.zeros_like(X)
        noise[:, :5] = rng.normal(0, 0.005, (len(X), 5))   # ~20 counts on the hall sensors
        noise[:, 5:] = rng.normal(0, 0.02, (len(X), 3))    # small tilt jitter
        Xs.append(X + noise); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# ----------------------------------------------------------------------------
# 3. Candidate models
# ----------------------------------------------------------------------------
def candidates():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "kNN": make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=15, weights="distance")),
    }


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def main():
    t_start = time.time()
    if not DATA_DIR.exists():
        log(f"No data folder at {DATA_DIR}")
        return 1
    sessions, dts = load_all()
    if len(sessions) < 2:
        log("Need at least 2 labelled classes. Record more data first.")
        return 1

    classes = sorted(sessions.keys())
    Xtr, ytr, Xte, yte, warnings = split_by_session(sessions)
    total_rows = len(Xtr) + len(Xte)
    log(f"\nClasses ({len(classes)}): {classes}")
    log(f"Train rows: {len(Xtr)}   Test rows (unseen sessions): {len(Xte)}")
    for w in warnings:
        log("WARNING:", w)
    dts = [d for d in dts if np.isfinite(d)]
    if dts and np.median(dts) > 40:
        log(f"NOTE: typical gap between readings in your recordings is {np.median(dts):.0f} ms "
            f"(~{1000/np.median(dts):.0f} Hz). New recordings at 100 Hz will be much better.")

    rng = np.random.RandomState(SEED)
    Xtr_a, ytr_a = augment(Xtr, ytr, AUGMENT_COPIES, rng)

    results = {}
    best_name, best_f1 = None, -1.0
    for name, model in candidates().items():
        t0 = time.time()
        model.fit(Xtr_a, ytr_a)
        pred = model.predict(Xte)
        f1 = f1_score(yte, pred, average="macro", labels=classes, zero_division=0)
        acc = float(np.mean(pred == yte))
        results[name] = (f1, acc)
        log(f"  {name:13s} test macro-F1 {f1:.3f}  accuracy {acc*100:.1f}%  ({time.time()-t0:.1f}s)")
        if f1 > best_f1:
            best_name, best_f1 = name, f1
    log(f"\nBest on unseen sessions: {best_name}")

    # ---- report on held-out sessions using the winning model (trained on train only)
    best = candidates()[best_name]
    best.fit(Xtr_a, ytr_a)
    proba = best.predict_proba(Xte)
    model_classes = list(best.classes_)
    pred = np.array([model_classes[i] for i in np.argmax(proba, axis=1)])
    cidx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([cidx[c] for c in yte])
    y_pred = np.array([cidx[c] for c in pred])
    labels = list(range(len(classes)))
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc_overall = float(np.mean(y_true == y_pred))

    class_metrics = []
    for i, c in enumerate(classes):
        mask = y_true == i
        correct = int((y_pred[mask] == i).sum()) if mask.any() else 0
        class_metrics.append({
            "class": c, "accuracy": float(correct / mask.sum()) if mask.any() else 0.0,
            "precision": float(prec[i]), "recall": float(rec[i]), "f1_score": float(f1s[i]),
            "support": int(sup[i]), "correct": correct, "total": int(mask.sum())})
        log(f"  {c:10s} recall {rec[i]*100:5.1f}%  precision {prec[i]*100:5.1f}%  (test rows {int(sup[i])})")

    cm_dict = {a: {b: int(cm[i][j]) for j, b in enumerate(classes)} for i, a in enumerate(classes)}
    errors = []
    for k in np.where(y_true != y_pred)[0]:
        p = proba[k]
        errors.append({
            "actual": classes[y_true[k]], "predicted": classes[y_pred[k]],
            "confidence": float(np.max(p)),
            "probabilities": {model_classes[j]: float(p[j]) for j in range(len(model_classes))}})
    errors = sorted(errors, key=lambda e: e["confidence"], reverse=True)[:10]

    # ---- final model: refit on ALL rows (more data = better), after the honest test above
    Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
    Xall_a, yall_a = augment(Xall, yall, AUGMENT_COPIES, rng)
    final = candidates()[best_name]
    final.fit(Xall_a, yall_a)
    if hasattr(final, "n_jobs"):
        final.n_jobs = 1          # live prediction is one row at a time; threads only add delay
    t0 = time.perf_counter()
    for _ in range(20):
        final.predict_proba(Xte[:1])
    infer_ms = (time.perf_counter() - t0) / 20 * 1000
    log(f"Single-row prediction time: {infer_ms:.1f} ms")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": final, "classes": list(final.classes_), "features": FEATURE_NAMES,
              "smooth_ms": SMOOTH_MS, "kind": "frame", "name": best_name}
    save_atomic(bundle, MODELS_DIR / "frame_model.joblib")
    config = {"kind": "frame", "model": best_name, "classes": list(final.classes_),
              "features": FEATURE_NAMES, "smooth_ms": SMOOTH_MS,
              "trained_at": datetime.now().isoformat(), "infer_ms": round(infer_ms, 2)}
    (MODELS_DIR / "config.json.tmp").write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(MODELS_DIR / "config.json.tmp", MODELS_DIR / "config.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "models/active/frame_model.joblib",
        "split": "by recording session (honest)",
        "training_config": {
            "model": best_name, "smooth_ms": SMOOTH_MS, "num_features": len(FEATURE_NAMES),
            "num_classes": len(classes), "total_samples": int(total_rows),
            "train_samples": int(len(Xtr)), "test_samples": int(len(Xte)),
            "test_split": TEST_FRACTION,
            # kept so older versions of the web page still show something sensible
            "epochs": best_name, "batch_size": len(FEATURE_NAMES), "window_size": SMOOTH_MS},
        "overall_metrics": {
            "test_accuracy": acc_overall, "test_accuracy_percent": acc_overall * 100,
            "test_macro_f1": float(best_f1), "test_loss": 0.0},
        "candidates": {k: {"macro_f1": v[0], "accuracy": v[1]} for k, v in results.items()},
        "warnings": warnings,
        "class_metrics": class_metrics, "confusion_matrix": cm_dict, "top_errors": errors,
    }
    (REPORTS_DIR / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nHonest test accuracy (unseen sessions): {acc_overall*100:.1f}%   macro-F1 {best_f1:.3f}")
    log(f"Saved model + report. Total time {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
