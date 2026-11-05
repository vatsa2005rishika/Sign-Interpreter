"""
app.py - Sign-Interpreter server.

What changed compared to the old version, and why:
  * /data-batch only STORES readings and returns at once. No model, no broadcast
    inside the request, so the glove is never kept waiting.
  * ONE background loop (every ~30 ms) looks at the newest readings, runs the
    model, and applies the decision rule. One prediction at a time, no pile-ups.
  * Decision rule: smooth probabilities, need high confidence + a clear lead,
    only while the hand is STILL, and the same answer for a few ticks in a row.
    A letter is announced once, and not again until you change or move away.
  * The browser gets updates ~15 times a second (not on every reading), with
    real confidence, motion state, decision time and measured data rate.
  * No TensorFlow needed any more (the model is a small scikit-learn model).
  * The old /data endpoint is kept so old firmware still works.

Run:  python app.py      (then open http://localhost:8000)
"""
import asyncio
import csv
import json
import os
import subprocess  # noqa: F401  (kept for clarity; asyncio subprocess is used below)
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from features import (MOTION_MS, SMOOTH_MS, gyro_magnitude, to_features)

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "index.html"
VENDOR_DIR = FRONTEND_DIR / "vendor"
DATA_FOLDER = Path(os.environ.get("SIGN_DATA_DIR", BASE_DIR / "data" / "ALL_DATA"))
MODELS_FOLDER = BASE_DIR / "models" / "active"
REPORTS_FOLDER = BASE_DIR / "reports"
LOGS_FOLDER = BASE_DIR / "logs"
TRAIN_SCRIPT = BASE_DIR / "train.py"
MODEL_PATH = MODELS_FOLDER / "frame_model.joblib"

for p in (VENDOR_DIR, LOGS_FOLDER, REPORTS_FOLDER, DATA_FOLDER):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- settings
# You can change these while the server runs:  POST /settings  {"accept_prob": 0.8}
CFG = {
    "tick_ms": 30,              # how often the model looks at the newest data
    "broadcast_every": 2,       # send to browser every N ticks (~15 per second)
    "ema_alpha": 0.5,           # smoothing of probabilities (1.0 = no smoothing)
    "accept_prob": 0.75,        # top guess must be at least this confident...
    "accept_margin": 0.25,      # ...and this far ahead of the runner-up
    "stable_ms": 150,           # ...and keep winning for this long (milliseconds)...
    "stable_min_ticks": 2,      # ...across at least this many separate predictions
    "use_motion_gate": True,    # only accept letters while the hand is still
    "motion_enter": 1.2,        # gyro level (rad/s) that means "MOVING"
    "motion_exit": 0.6,         # gyro level below which it is "STILL" again (hysteresis)
    "hold_ms": 700,             # keep showing the last letter this long after confidence drops
    "rearm_move_ms": 600,       # moving this long lets the same letter be announced again
    "signal_timeout_ms": 600,   # no data for this long = glove considered disconnected
    "rest_label": "Neutral",    # class that means "not signing"; never announced as a letter
}


# ---------------------------------------------------------------- state
class State:
    def __init__(self):
        self.buf = deque(maxlen=600)       # (ts_device_ms, [11 numbers])
        self.bundle = None                 # loaded model
        self.mode = "prediction"           # or "recording"
        self.rec = []
        self.rec_label = None
        self.rec_person = "p1"
        self.last_rx = None
        self.last_seq = None
        self.seq_gaps = 0
        self.dropped_reported = 0
        self.total_rows = 0
        self.rate_win = deque()            # (monotonic time, rows)
        self.new_rows = 0
        self.last_row = None
        self.infer_ms = 0.0
        # decision state
        self.ema = None
        self.top_guess = None
        self.top_conf = None
        self.margin = None
        self.motion = 0.0
        self.motion_state = "STILL"
        self.moving_since = None
        self.still_since = time.monotonic()
        self.cand = None
        self.cand_ticks = 0
        self.cand_since = 0.0
        self.last_good = 0.0
        self.committed = None
        self.last_emitted = None
        self.decision_ms = None
        self.since_still_ms = None


S = State()


# ---------------------------------------------------------------- browser connections
class ConnectionManager:
    def __init__(self):
        self.active = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data):
        if not self.active:
            return
        msg = json.dumps(data)

        async def _send(ws):
            try:
                await asyncio.wait_for(ws.send_text(msg), 0.25)
            except Exception:
                self.disconnect(ws)

        await asyncio.gather(*(_send(w) for w in list(self.active)))


manager = ConnectionManager()


# ---------------------------------------------------------------- model
def load_model():
    try:
        bundle = joblib.load(MODEL_PATH)
        bundle["classes"] = [str(c) for c in bundle["classes"]]
        m = bundle["model"]
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
        S.bundle = bundle                  # one atomic swap; the loop reads a local copy
        S.ema = None
        print(f"Model loaded: {bundle.get('name')}  classes: {bundle['classes']}")
        return True
    except Exception as e:
        S.bundle = None
        print(f"No model loaded ({e}). Record data and press Train.")
        return False


# ---------------------------------------------------------------- ingest
ROW_LEN = 13  # seq, ts, acc xyz, gyro xyz, hall x5


def ingest_rows(rows, dropped=None):
    """Store readings. Must stay FAST: it runs inside the glove's HTTP request."""
    now = time.monotonic()
    n = 0
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) != ROW_LEN:
            continue
        seq, ts = r[0], float(r[1])
        if S.last_seq is not None and seq > S.last_seq + 1:
            S.seq_gaps += int(seq - S.last_seq - 1)
        if S.buf and ts < S.buf[-1][0]:          # glove rebooted: timestamps restarted
            S.buf.clear()
        S.last_seq = seq
        feats = [float(v) for v in r[2:]]
        S.buf.append((ts, feats))
        S.last_row = (ts, feats)
        if S.mode == "recording":
            S.rec.append({
                "seq": int(seq), "ts_device": ts,
                "mpu_acc_x": feats[0], "mpu_acc_y": feats[1], "mpu_acc_z": feats[2],
                "mpu_gyro_x": feats[3], "mpu_gyro_y": feats[4], "mpu_gyro_z": feats[5],
                "hall_0": feats[6], "hall_1": feats[7], "hall_2": feats[8],
                "hall_3": feats[9], "hall_4": feats[10],
                "person_id": S.rec_person,
            })
        n += 1
    if n:
        S.total_rows += n
        S.new_rows += n
        S.last_rx = now
        S.rate_win.append((now, n))
    if dropped is not None:
        S.dropped_reported = int(dropped)
    return n


def telemetry():
    now = time.monotonic()
    while S.rate_win and now - S.rate_win[0][0] > 3.0:
        S.rate_win.popleft()
    rate = sum(n for _, n in S.rate_win) / 3.0
    connected = S.last_rx is not None and (now - S.last_rx) * 1000 < CFG["signal_timeout_ms"]
    return {
        "rate_hz": round(rate, 1), "seq_gaps": S.seq_gaps,
        "dropped_on_glove": S.dropped_reported, "glove_connected": connected,
        "total_messages": S.total_rows, "messages_last_min": int(rate * 60),
        "recording_count": len(S.rec), "infer_ms": round(S.infer_ms, 1),
        "last_received": datetime.now().isoformat() if S.last_rx else None,
    }


# ---------------------------------------------------------------- decision loop
def tail_rows(window_ms):
    """Readings from the last window_ms (by the glove's own clock), oldest first."""
    if not S.buf:
        return None
    last_ts = S.buf[-1][0]
    out = []
    for ts, f in reversed(S.buf):
        if ts <= last_ts - window_ms:
            break
        if ts <= last_ts:
            out.append(f)
    out.reverse()
    return np.array(out, dtype=float) if out else None


def log_decision(label, conf, decision_ms, since_still_ms):
    path = LOGS_FOLDER / "decisions.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "label", "confidence", "decision_ms", "since_still_ms", "rate_hz"])
        w.writerow([datetime.now().isoformat(), label, round(conf, 3),
                    round(decision_ms or 0), round(since_still_ms or 0), telemetry()["rate_hz"]])


def tick_once(now):
    """One look at the newest data. Returns a list of events to announce."""
    events = []
    bundle = S.bundle
    smooth = tail_rows(SMOOTH_MS)
    motion_rows = tail_rows(MOTION_MS)
    if smooth is None or motion_rows is None:
        return events

    # --- motion state (with two thresholds so it does not flicker)
    S.motion = float(gyro_magnitude(motion_rows).mean())
    if S.motion_state == "STILL" and S.motion > CFG["motion_enter"]:
        S.motion_state, S.moving_since = "MOVING", now
    elif S.motion_state == "MOVING" and S.motion < CFG["motion_exit"]:
        S.motion_state, S.moving_since, S.still_since = "STILL", None, now
    if S.motion_state == "MOVING" and S.moving_since and \
            (now - S.moving_since) * 1000 > CFG["rearm_move_ms"]:
        S.last_emitted = None              # a real movement away: same letter may be announced again

    if bundle is None or S.new_rows == 0:
        return events
    S.new_rows = 0

    # --- model
    t0 = time.perf_counter()
    X = to_features(smooth.mean(axis=0, keepdims=True))
    proba = bundle["model"].predict_proba(X)[0]
    ms = (time.perf_counter() - t0) * 1000
    S.infer_ms = ms if S.infer_ms == 0 else 0.9 * S.infer_ms + 0.1 * ms
    classes = bundle["classes"]

    a = CFG["ema_alpha"]
    S.ema = proba if (S.ema is None or len(S.ema) != len(proba)) else a * proba + (1 - a) * S.ema
    order = np.argsort(S.ema)[::-1]
    top = float(S.ema[order[0]])
    second = float(S.ema[order[1]]) if len(order) > 1 else 0.0
    cand = classes[order[0]]
    S.top_guess, S.top_conf, S.margin = cand, top, top - second

    # --- decision rule
    gate_ok = (not CFG["use_motion_gate"]) or S.motion_state == "STILL"
    good = top >= CFG["accept_prob"] and (top - second) >= CFG["accept_margin"] and gate_ok
    if good:
        S.last_good = now
        if cand == S.cand:
            S.cand_ticks += 1
        else:
            S.cand, S.cand_ticks, S.cand_since = cand, 1, now
        if S.cand_ticks >= CFG["stable_min_ticks"] and (now - S.cand_since) * 1000 >= CFG["stable_ms"]:
            S.committed = cand
            if cand == CFG["rest_label"]:
                S.last_emitted = None
            elif cand != S.last_emitted:
                S.last_emitted = cand
                S.decision_ms = (now - S.cand_since) * 1000
                S.since_still_ms = (now - S.still_since) * 1000
                events.append({"label": cand, "confidence": top,
                               "decision_ms": round(S.decision_ms),
                               "since_still_ms": round(S.since_still_ms)})
    else:
        S.cand, S.cand_ticks = None, 0
        if S.committed is not None and (now - S.last_good) * 1000 > CFG["hold_ms"]:
            S.committed = None
    return events


def live_payload():
    ts, f = S.last_row if S.last_row else (0, [0.0] * 11)
    return {
        "type": "live_update",
        "data": {
            "prediction": S.committed if S.committed is not None else "...",
            "top_guess": S.top_guess, "confidence": S.top_conf, "margin": S.margin,
            "motion": {"value": round(S.motion, 2), "state": S.motion_state},
            "decision_ms": None if S.decision_ms is None else round(S.decision_ms),
            "recording_count": len(S.rec),
            "raw_data": {
                "mpu6050": {"acc": {"x": f[0], "y": f[1], "z": f[2]},
                            "gyro": {"x": f[3], "y": f[4], "z": f[5]}},
                "halls": f[6:11], "ts_device": ts,
                "ts_server": datetime.now().isoformat(),
            },
            "telemetry": telemetry(),
        },
    }


async def inference_loop():
    n = 0
    errors = 0
    while True:
        await asyncio.sleep(CFG["tick_ms"] / 1000.0)
        n += 1
        try:
            now = time.monotonic()
            connected = S.last_rx is not None and (now - S.last_rx) * 1000 < CFG["signal_timeout_ms"]
            events = tick_once(now) if connected else []
            for ev in events:
                log_decision(ev["label"], ev["confidence"], ev["decision_ms"], ev["since_still_ms"])
                await manager.broadcast({"type": "letter", "data": {**ev, "t": datetime.now().isoformat()}})
            if n % max(1, CFG["broadcast_every"]) == 0:
                await manager.broadcast(live_payload())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep running, but do not hide the problem
            errors += 1
            if errors <= 5:
                print(f"[inference_loop] error: {e!r}")


@asynccontextmanager
async def lifespan(app):
    load_model()
    task = asyncio.create_task(inference_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/vendor", StaticFiles(directory=str(VENDOR_DIR)), name="vendor")


# ---------------------------------------------------------------- glove endpoints
@app.post("/data-batch")
async def data_batch(request: Request):
    """Main endpoint for the glove: a batch of readings. Store and return - nothing else."""
    body = await request.json()
    rows = body.get("rows", []) if isinstance(body, dict) else body
    n = ingest_rows(rows, body.get("drop") if isinstance(body, dict) else None)
    return {"ok": 1, "n": n}


@app.post("/data")
async def legacy_data(request: Request):
    """Old firmware: one reading per request in the old nested JSON format."""
    s = await request.json()
    try:
        ts = s.get("ts_device", None)
        try:
            ts = float(ts)
        except Exception:
            ts = time.monotonic() * 1000.0
        a, g, h = s["mpu6050"]["acc"], s["mpu6050"]["gyro"], s["halls"]
        row = [s.get("seq", 0), ts, a["x"], a["y"], a["z"], g["x"], g["y"], g["z"],
               h[0], h[1], h[2], h[3], h[4]]
    except Exception:
        return JSONResponse({"prediction": "Error"}, status_code=400)
    ingest_rows([row])
    return {"prediction": S.committed if S.committed is not None else "..."}


# ---------------------------------------------------------------- browser endpoints
@app.get("/")
async def get_index():
    return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))


@app.get("/dataset-stats")
async def get_dataset_stats():
    stats = []
    if not DATA_FOLDER.exists():
        return stats
    for d in DATA_FOLDER.iterdir():
        if d.is_dir():
            stats.append({"word": d.name, "count": len(list(d.glob("*.parquet")))})
    return sorted(stats, key=lambda x: x["word"])


@app.get("/training-report")
async def get_training_report():
    path = REPORTS_FOLDER / "training_report.json"
    if not path.exists():
        return JSONResponse({"error": "No training report yet. Train the model first."}, status_code=404)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return JSONResponse({"error": f"Failed to load report: {e}"}, status_code=500)


@app.get("/status")
async def status():
    return {**telemetry(), "app_mode": S.mode, "model_loaded": S.bundle is not None,
            "motion": {"value": round(S.motion, 2), "state": S.motion_state}, "settings": CFG}


@app.get("/settings")
async def get_settings():
    return CFG


@app.post("/settings")
async def set_settings(request: Request):
    body = await request.json()
    changed = {}
    for k, v in body.items():
        if k in CFG:
            try:
                CFG[k] = type(CFG[k])(v)
                changed[k] = CFG[k]
            except Exception:
                pass
    return {"changed": changed}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.post("/set-mode")
async def set_mode(request: Request):
    body = await request.json()
    new_mode = body.get("mode")
    if new_mode not in ("prediction", "recording"):
        return JSONResponse({"status": "error", "message": "Invalid mode"}, status_code=400)
    S.mode = new_mode
    if new_mode != "recording":
        S.rec, S.rec_label = [], None
    return {"status": "success", "mode": S.mode}


@app.post("/start-recording")
async def start_recording(request: Request):
    body = await request.json()
    label = (body.get("label") or "").strip()
    if not label:
        return JSONResponse({"status": "error", "message": "No label provided"}, status_code=400)
    if any(c in label for c in '/\\:*?"<>|'):
        return JSONResponse({"status": "error", "message": "Label has characters not allowed in a folder name"},
                            status_code=400)
    S.mode, S.rec, S.rec_label = "recording", [], label
    S.rec_person = (body.get("person_id") or "p1").strip() or "p1"
    print(f"--- Recording started: [{label}] person={S.rec_person}")
    return {"status": "recording_started", "label": label}


@app.post("/stop-recording")
async def stop_recording():
    if S.mode != "recording" or not S.rec_label:
        return JSONResponse({"status": "error", "message": "Not recording"}, status_code=400)
    try:
        if not S.rec:
            return {"status": "stopped", "message": "No data collected."}
        df = pd.DataFrame(S.rec)
        folder = DATA_FOLDER / S.rec_label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(path, index=False)
        rate = ""
        if len(df) > 2:
            dt = np.median(np.diff(df["ts_device"].to_numpy(dtype=float)))
            if dt > 0:
                rate = f" (about {1000/dt:.0f} Hz)"
        return {"status": "stopped_and_saved", "message": f"Saved {len(df)} rows{rate} to {path.name}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {e}"}
    finally:
        S.rec, S.rec_label, S.mode = [], None, "prediction"


@app.post("/start-training")
async def start_training(background_tasks: BackgroundTasks):
    async def run():
        await manager.broadcast({"type": "train_log", "data": "--- Starting training ---"})
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(TRAIN_SCRIPT), cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        while proc.stdout:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"[TRAIN] {text}")
            await manager.broadcast({"type": "train_log", "data": text})
        await proc.wait()
        if proc.returncode != 0:
            await manager.broadcast({"type": "train_log", "data": "--- Training failed ---"})
            await manager.broadcast({"type": "train_complete", "success": False})
            return
        ok = load_model()
        await manager.broadcast({"type": "train_log",
                                 "data": "--- New model loaded ---" if ok else "ERROR: could not load new model"})
        await manager.broadcast({"type": "train_complete", "success": bool(ok)})

    background_tasks.add_task(run)
    return {"status": "training_started"}


if __name__ == "__main__":
    print("Sign-Interpreter server: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
