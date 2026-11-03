# import uvicorn
# import asyncio
# import json
# import pandas as pd
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
# from fastapi.responses import HTMLResponse
# from datetime import datetime
# import os

# # --- 1. CONFIGURATION ---
# DATA_FOLDER = "data" # Root folder for all labeled data

# # --- 2. GLOBAL STATE ---
# app = FastAPI()
# is_recording = False
# current_label = None
# data_buffer = [] # Holds data temporarily during recording

# # --- 3. WEBSOCKET MANAGER ---
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: list[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)

#     async def broadcast(self, message: str):
#         for connection in self.active_connections:
#             await connection.send_text(message)

# manager = ConnectionManager()

# # --- 4. API ENDPOINTS ---

# @app.get("/")
# async def get():
#     """Serves the index.html file."""
#     with open("index.html") as f:
#         return HTMLResponse(f.read())

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             await websocket.receive_text() # Keep connection alive
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         print("Client disconnected")

# # --- RECORDING LOGIC ---

# @app.post("/start")
# async def start_recording(request: Request):
#     global is_recording, current_label, data_buffer

#     body = await request.json()
#     label = body.get("label")

#     if not label:
#         return {"status": "error", "message": "No label provided."}
#     if is_recording:
#         return {"status": "error", "message": f"Already recording. Stop first."}

#     data_buffer = []
#     current_label = label
#     is_recording = True

#     print(f"--- Started recording for label: [{label}] ---")
#     return {"status": "recording_started", "label": label}

# @app.post("/stop")
# async def stop_recording():
#     global is_recording, current_label, data_buffer

#     if not is_recording:
#         return {"status": "error", "message": "Not recording."}

#     print("--- Stopping recording... ---")
#     is_recording = False

#     if not data_buffer:
#         print("No data collected, no file saved.")
#         return {"status": "stopped", "message": "No data collected."}

#     try:
#         df = pd.DataFrame(data_buffer)
#         label_folder = os.path.join(DATA_FOLDER, current_label)
#         os.makedirs(label_folder, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"session_{timestamp}.parquet"
#         filepath = os.path.join(label_folder, filename)

#         df.to_parquet(filepath, index=False)

#         print(f"Saved {len(data_buffer)} rows to {filepath}")

#         return {
#             "status": "stopped_and_saved",
#             "label": current_label,
#             "rows": len(data_buffer),
#             "filepath": filepath
#         }

#     except Exception as e:
#         print(f"Error saving file: {e}")
#         return {"status": "error", "message": f"Failed to save file: {e}"}
#     finally:
#         data_buffer = []
#         current_label = None

# # --- ESP32 ENDPOINT ---

# @app.post("/data")
# async def receive_data(request: Request):
#     global data_buffer

#     try:
#         data = await request.json()
#         data["ts_server"] = datetime.now().isoformat()

#         if is_recording:
#             data_buffer.append(data)

#         # Broadcast raw data to the UI
#         await manager.broadcast(json.dumps(data))

#         return {"status": "success"}

#     except Exception as e:
#         print(f"Error processing request: {e}")
#         return {"status": "error", "message": str(e)}

# # --- 5. RUN THE APP ---
# if __name__ == "__main__":
#     print(f"Creating data folder at: {DATA_FOLDER}")
#     os.makedirs(DATA_FOLDER, exist_ok=True)
#     print("Starting FastAPI DATA COLLECTION server at http://localhost:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import uvicorn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.models import load_model
from collections import deque
import json
import os
import asyncio
import sys
from datetime import datetime, timedelta

# Suppress TensorFlow info and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error only

# --- 1. Configuration ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
DATA_FOLDER = BASE_DIR / "data" / "ALL_DATA"
MODELS_FOLDER = BASE_DIR / "models" / "active"
REPORTS_FOLDER = BASE_DIR / "reports"
TRAIN_SCRIPT = BASE_DIR / "train.py"

MODEL_PATH = MODELS_FOLDER / "sign_language_model.keras"
SCALER_PATH = MODELS_FOLDER / "scaler.joblib"
ENCODER_PATH = MODELS_FOLDER / "label_encoder.joblib"

# Prediction settings
PREDICTION_THRESHOLD = 0.60 # Lowered to 60%
WINDOW_SIZE = 50
NUM_FEATURES = 11

# --- 2. Load Models and Objects ---
# We load these on startup
try:
    print("Loading model, scaler, and encoder...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    CLASS_NAMES = label_encoder.classes_
    print(f"Models loaded. Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model/scaler/encoder. {e}")
    print("Please run train.py to create these files.")
    model, scaler, label_encoder, CLASS_NAMES = None, None, None, []

# --- 3. FastAPI App & Global State ---
app = FastAPI()

# This is our live "sliding window" for prediction
prediction_buffer = deque(maxlen=WINDOW_SIZE)
last_prediction = "..."

# Telemetry: track last-received sample and recent message rate
last_received_time = None
message_timestamps = deque(maxlen=1000)  # store recent receive times
total_messages = 0

# This is our buffer for data collection
recording_buffer = []
current_label = None
# This is our buffer for data collection
recording_buffer = []
current_label = None

# Global state for the app's mode
# Modes: "prediction", "recording"
app_mode = "prediction" # Default mode

# --- 4. WebSocket Manager (for Browser UI) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        """Broadcasts a JSON message to all connected UI clients."""
        message_str = json.dumps(data)
        # Send messages asynchronously to avoid blocking the request handler
        for connection in list(self.active_connections):
            try:
                # Schedule send without awaiting to prevent slow clients blocking
                asyncio.create_task(connection.send_text(message_str))
            except Exception:
                # If scheduling fails, remove the connection to keep list clean
                try:
                    self.active_connections.remove(connection)
                except Exception:
                    pass

manager = ConnectionManager()

# --- 5. Helper Function for Flattening ---
def flatten_json_sample(sample_json):
    """Flattens a single incoming JSON sample into a 1D list."""
    try:
        features = [
            sample_json['mpu6050']['acc']['x'],
            sample_json['mpu6050']['acc']['y'],
            sample_json['mpu6050']['acc']['z'],
            sample_json['mpu6050']['gyro']['x'],
            sample_json['mpu6050']['gyro']['y'],
            sample_json['mpu6050']['gyro']['z'],
            sample_json['halls'][0],
            sample_json['halls'][1],
            sample_json['halls'][2],
            sample_json['halls'][3],
            sample_json['halls'][4],
        ]
        return features
    except Exception:
        return None

# --- 6. Main ESP32 Endpoint ---
@app.post("/data")
async def receive_data(request: Request):
    """
    Main endpoint for the ESP32.
    It ALWAYS runs prediction and returns it.
    It conditionally saves data if in "recording" mode.
    """
    global prediction_buffer, last_prediction, recording_buffer, app_mode

    sample_json = await request.json()
    flat_features = flatten_json_sample(sample_json)

    if not flat_features:
        return JSONResponse({"prediction": "Error"}, status_code=400)

    # Record telemetry
    try:
        now = datetime.now()
        global last_received_time, total_messages
        last_received_time = now
        message_timestamps.append(now)
        total_messages += 1
    except Exception:
        pass

    # 1. Prediction Logic (Always runs)
    prediction_buffer.append(flat_features)
    current_prediction = "..."

    if len(prediction_buffer) == WINDOW_SIZE:
        # Run prediction off the event loop to avoid blocking high-rate POSTs
        window_data = np.array(prediction_buffer)

        def _predict_window(window_arr):
            try:
                scaled_data = scaler.transform(window_arr)
                scaled_data_reshaped = np.reshape(scaled_data, (1, WINDOW_SIZE, NUM_FEATURES))
                prediction_probs = model.predict(scaled_data_reshaped, verbose=0)
                confidence_val = float(np.max(prediction_probs))
                pred_idx = int(np.argmax(prediction_probs))
                probs = prediction_probs[0].tolist()
                return confidence_val, pred_idx, probs
            except Exception:
                return 0.0, None, None

        try:
            confidence, pred_idx, probs = await asyncio.to_thread(_predict_window, window_data)
            if probs is not None and confidence is not None:
                if confidence > PREDICTION_THRESHOLD and pred_idx is not None:
                    current_prediction = CLASS_NAMES[pred_idx]
                last_prediction = current_prediction
        except Exception:
            # Prediction failed; keep last_prediction unchanged
            pass
        # print(f"Pred: {last_prediction} (Conf: {confidence:.2f})") # For debugging

    # 2. Recording Logic (Runs conditionally)
    if app_mode == "recording":
        # Add server timestamp for recording
        sample_json["ts_server"] = datetime.now().isoformat()
        recording_buffer.append(sample_json)

    # 3. Broadcast to UI (WebSocket)
    # Send both raw data and prediction to the browser, plus telemetry
    # Compute messages per minute
    try:
        cutoff = datetime.now() - timedelta(seconds=60)
        msgs_last_min = sum(1 for t in message_timestamps if t >= cutoff)
    except Exception:
        msgs_last_min = len(message_timestamps)

    broadcast_data = {
        "type": "live_update",
        "data": {
            "prediction": last_prediction,
            "raw_data": {
                "mpu6050": sample_json.get("mpu6050", {"acc": {"x": 0, "y": 0, "z": 0}, "gyro": {"x": 0, "y": 0, "z": 0}}),
                "halls": sample_json.get("halls", [0, 0, 0, 0, 0]),
                "ts_device": sample_json.get("ts_device", datetime.now().isoformat()),
                "ts_server": sample_json.get("ts_server", datetime.now().isoformat())
            },
            # Telemetry fields the frontend can display
            "telemetry": {
                "last_received": last_received_time.isoformat() if last_received_time else None,
                "messages_last_min": int(msgs_last_min),
                "total_messages": int(total_messages),
                "recording_count": len(recording_buffer)
            }
        }
    }

    await manager.broadcast_json(broadcast_data)

    # 4. Return prediction to ESP32
    # This is the "answer" your glove is waiting for
    return {"prediction": last_prediction}

# --- 7. Browser UI Endpoints ---
@app.get("/")
async def get_index():
    """Serves the main HTML file."""
    return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))

@app.get("/dataset-stats")
async def get_dataset_stats():
    """Returns the count of files per label in the dataset."""
    stats = []
    data_path = DATA_FOLDER

    if not data_path.exists():
        return stats

    for label_dir in data_path.iterdir():
        if label_dir.is_dir():
            parquet_files = list(label_dir.glob("*.parquet"))
            stats.append({
                "word": label_dir.name,
                "count": len(parquet_files)
            })

    return sorted(stats, key=lambda x: x["word"])

@app.get("/training-report")
async def get_training_report():
    """Returns the latest training report."""
    report_path = REPORTS_FOLDER / "training_report.json"

    if not report_path.exists():
        return JSONResponse(
            {"error": "No training report available yet. Train the model first."},
            status_code=404
        )

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to load report: {str(e)}"},
            status_code=500
        )


@app.get("/status")
async def status():
    """Return simple server telemetry so frontend or user can diagnose stalls."""
    try:
        last = last_received_time.isoformat() if last_received_time else None
        # messages in last minute
        try:
            cutoff = datetime.now() - pd.Timedelta(seconds=60)
            msgs_last_min = sum(1 for t in message_timestamps if t >= cutoff)
        except Exception:
            msgs_last_min = len(message_timestamps)

        return {
            "last_received": last,
            "messages_last_min": int(msgs_last_min),
            "total_messages": int(total_messages),
            "app_mode": app_mode
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the browser's WebSocket connection."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("UI Client disconnected")

@app.post("/set-mode")
async def set_mode(request: Request):
    """Sets the server's mode (prediction or recording)."""
    global app_mode, recording_buffer, current_label
    body = await request.json()
    new_mode = body.get("mode")

    if new_mode not in ["prediction", "recording"]:
        return JSONResponse({"status": "error", "message": "Invalid mode"}, status_code=400)

    app_mode = new_mode
    print(f"--- Mode changed to: {app_mode} ---")

    # If switching away from recording, clear any buffer
    if app_mode != "recording":
        recording_buffer = []
        current_label = None

    return {"status": "success", "mode": app_mode}

@app.post("/start-recording")
async def start_recording(request: Request):
    """Starts a new recording session."""
    global recording_buffer, current_label, app_mode
    body = await request.json()
    label = body.get("label")

    if not label:
        return JSONResponse({"status": "error", "message": "No label provided"}, status_code=400)

    # Set mode to recording and clear buffer
    app_mode = "recording"
    recording_buffer = []
    current_label = label

    print(f"--- Started recording for label: [{label}] ---")
    return {"status": "recording_started", "label": label}

@app.post("/stop-recording")
async def stop_recording():
    """Stops recording and saves data to a Parquet file."""
    global app_mode, recording_buffer, current_label

    if app_mode != "recording" or not current_label:
        return JSONResponse({"status": "error", "message": "Not recording"}, status_code=400)

    print("--- Stopping recording... ---")

    if not recording_buffer:
        print("No data collected, no file saved.")
        app_mode = "prediction" # Switch back to prediction mode
        return {"status": "stopped", "message": "No data collected."}

    try:
        df = pd.DataFrame(recording_buffer)
        label_folder = DATA_FOLDER / current_label
        os.makedirs(label_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.parquet"
        filepath = label_folder / filename

        df.to_parquet(filepath, index=False)

        message = f"Saved {len(recording_buffer)} rows to {filepath}"
        print(message)

        return {"status": "stopped_and_saved", "message": message}

    except Exception as e:
        print(f"Error saving file: {e}")
        return {"status": "error", "message": f"Failed to save file: {e}"}
    finally:
        # Reset state
        recording_buffer = []
        current_label = None
        app_mode = "prediction" # Always default back to prediction

@app.post("/start-training")
async def start_training(background_tasks: BackgroundTasks):
    """Kicks off the training script as a background process."""

    async def run_training_in_background():
        """Runs train.py and streams stdout to the UI."""
        print("--- Starting Training Process ---")
        await manager.broadcast_json({"type": "train_log", "data": "--- Starting Training Process ---"})

        # We use asyncio.create_subprocess_exec to run the command
        # This allows us to capture stdout line by line
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(TRAIN_SCRIPT),
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream stdout
        if process.stdout:
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8').strip()
                print(f"[TRAIN]: {line}")
                await manager.broadcast_json({"type": "train_log", "data": line})

        # Stream stderr (for errors)
        if process.stderr:
            while True:
                line_bytes = await process.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8').strip()
                print(f"[TRAIN ERROR]: {line}")
                await manager.broadcast_json({"type": "train_log", "data": f"ERROR: {line}"})

        await process.wait()
        print("--- Training Process Finished ---")
        await manager.broadcast_json({"type": "train_log", "data": "--- Training Process Finished ---"})

        if process.returncode != 0:
            await manager.broadcast_json({"type": "train_complete", "success": False})
            return

        # Reload the model after training
        try:
            global model, scaler, label_encoder, CLASS_NAMES
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            CLASS_NAMES = label_encoder.classes_
            print("--- Hot-reloaded new model, scaler, and encoder ---")
            await manager.broadcast_json({"type": "train_log", "data": "--- New model has been loaded! ---"})
            await manager.broadcast_json({"type": "train_complete", "success": True})
        except Exception as e:
            print(f"Error hot-reloading model: {e}")
            await manager.broadcast_json({"type": "train_log", "data": f"ERROR: Could not reload new model: {e}"})
            await manager.broadcast_json({"type": "train_complete", "success": False})

    # Add the training function as a background task
    # This lets the server return a response immediately
    background_tasks.add_task(run_training_in_background)

    return {"status": "training_started"}

# --- 8. Run the App ---
if __name__ == "__main__":
    print("Starting all-in-one FastAPI server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)