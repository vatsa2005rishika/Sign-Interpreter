# Sign-Interpreter update: what to install, in what order

## Files and where they go
| File in this zip | Put it here (your repo) | What it is |
|---|---|---|
| `firmware/SignGlove/SignGlove.ino` | new Arduino sketch folder `SignGlove/` | New glove firmware (steady 100 Hz, batched, no waiting) |
| `firmware/SignGlove/secrets.h` | same folder as the .ino | Your Wi-Fi name, password, laptop IP (already in `.gitignore`) |
| `backend/app.py` | `backend/app.py` (replace) | New server |
| `backend/train.py` | `backend/train.py` (replace) | New training (honest split, per-reading model) |
| `backend/features.py` | `backend/features.py` (new) | Shared feature code, used by BOTH files above |
| `backend/get_vendor.py` | `backend/` (new) | Downloads web libraries once so the page works offline |
| `backend/patch_index.py` | `backend/` (new) | Applies the small web-page fixes to your `frontend/index.html` |
| `backend/requirements.txt` | `backend/` | Python packages (TensorFlow is no longer needed) |

## Steps
1. `pip install -r backend/requirements.txt`
2. Copy the files in. Keep your `data/` folder. The old `models/active/*.keras`, `scaler.joblib`, `label_encoder.joblib` are simply ignored.
3. While the laptop has internet: `python get_vendor.py`, then `python patch_index.py`. (patch_index.py makes `index.html.bak` first and is safe to run twice.)
4. `python train.py` (or press Train in the page). Your OLD recordings work for this model, because it looks at one moment, not a window. Read the printed accuracy: it is now measured on recordings the model has never seen, so it will probably be LOWER than the old 96.3%. That lower number is the honest one.
5. Edit `secrets.h` (Wi-Fi name/password, laptop IP). Flash `SignGlove.ino`. Libraries: Adafruit MPU6050 and Adafruit Unified Sensor. ArduinoJson is no longer needed.
6. `python app.py`, open `http://localhost:8000`. Allow port 8000 in the laptop firewall (as before).

## How to check it worked
- Dashboard line under the big letter shows **Hz** (should read about 100) and **gaps** (should stay at 0 or very low).
- Serial monitor prints every 5 s: `queue=... dropped=... postFail=...`. `queue` near 0 and `dropped` not rising = healthy.
- Record a new session: the Record tab message now shows the real rate (for example "about 100 Hz"). Your Step 0 script should now give gaps near 10 ms.
- `logs/decisions.csv` gets one row per announced letter, with `decision_ms` (how long it took to become sure) and the data rate.

## What changed and why (short)
- **Firmware:** sensors have their own timer task; sending is a separate task with a queue; no waiting for the server; Wi-Fi power-saving off (it causes random 100+ ms delays); 4x averaged finger reads; 400 kHz I2C.
- **Server:** `/data-batch` only stores and returns. One loop every 30 ms predicts and applies the rule: confident (>= 0.75), clear lead (>= 0.25), hand STILL, same answer for 150 ms. A letter is announced once, then not again until you change letter, pass through Neutral, or move your hand away. UI gets ~15 updates/s with real confidence.
- **Training:** split by recording session; classes with only one session get a warning (their score is optimistic); three small models compared, best kept; report JSON keeps the shape the page expects.
- **Page:** real confidence, hand state, decision time, data rate; letters strip with Backspace / Clear / Speak; works offline.

## Tuning without restarting
`POST http://localhost:8000/settings` with JSON, for example `{"accept_prob": 0.8, "motion_enter": 2.0}`. `GET /settings` lists all knobs. If letters never appear, lower `accept_prob` or raise `motion_enter` (the hand may be counted as MOVING all the time; the dashboard shows the live value).

## Please read: what was and was not tested
- Tested here: training and the live server, end to end, with SIMULATED glove data (both old nested and new flat recordings; letters announced once; no repeats while held or moving; recording, settings and legacy `/data` endpoint). The index patcher was tested on copies of the exact lines it changes.
- Syntax-checked only: the firmware (against stand-in Arduino headers). It has NOT been compiled for a real ESP32 or flashed. Expect to fix a small compile detail if your library versions differ.
- Not tested: `get_vendor.py` (no internet in the test environment), the patched page in a browser, and accuracy on your real data. I cannot promise a real accuracy number.
- Not included yet: calibration (Step 4), dynamic gestures, BLE, on-device model. This update fixes timing, the decision logic, honest evaluation and the UI bugs.

## If something is off
| Symptom | Likely cause | Try |
|---|---|---|
| Hz on dashboard far below 100 | Weak Wi-Fi / hotspot | Move closer; check `rssi` in serial; check `postFail` |
| `gaps` rising | Glove dropping readings when network stalls | Check `dropped` in serial; reduce other Wi-Fi use |
| Letter never appears | Hand counted as MOVING, or confidence too low | Watch "hand STILL/MOVING"; adjust `motion_enter`, `accept_prob` |
| A letter appears twice | You moved away and back, or passed through another letter | Expected; raise `rearm_move_ms` if too eager |
| Training says only ONE session for a class | Not enough recordings for that letter | Record 4+ separate sessions per letter |
| Page is blank offline | Vendor files missing | Run `get_vendor.py` while online, then `patch_index.py` |
