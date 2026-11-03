import uvicorn
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from collections import deque
import json

# --- 1. Configuration ---
MODEL_PATH = "sign_language_model.h5"
SCALER_PATH = "scaler.joblib"
ENCODER_PATH = "label_encoder.joblib"

# --- NEW: Confidence Threshold ---
# Only show a prediction if the model is > 90% confident
PREDICTION_THRESHOLD = 0.75

# These must match your training script
WINDOW_SIZE = 100 # 2.0 seconds * 50 Hz
NUM_FEATURES = 11
FEATURE_COLUMNS = [ 
    'mpu_acc_x', 'mpu_acc_y', 'mpu_acc_z',
    'mpu_gyro_x', 'mpu_gyro_y', 'mpu_gyro_z',
    'hall_0', 'hall_1', 'hall_2', 'hall_3', 'hall_4'
]

# --- 2. Load Models and Objects ---
try:
    print("Loading model and utilities...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model, scaler, and encoder loaded successfully.")
    
    CLASS_NAMES = label_encoder.classes_
    print(f"Classes: {CLASS_NAMES}")
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model/scaler. {e}")
    print("Please run train.py to create the files.")
    exit()

# --- 3. FastAPI App & Global State ---
app = FastAPI()
data_buffer = deque(maxlen=WINDOW_SIZE)
last_prediction = "..."

# --- 4. WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- 5. Helper Function for Flattening ---
def flatten_json_sample(sample_json):
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
    except Exception as e:
        print(f"Error flattening sample: {e}")
        return None

# --- 6. API Endpoints ---
@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("UI Client disconnected")

@app.post("/data")
async def receive_data(request: Request):
    global data_buffer, last_prediction
    
    try:
        sample_json = await request.json()
        flat_features = flatten_json_sample(sample_json)
        
        if not flat_features:
            raise ValueError("Invalid data format from ESP32")
            
        data_buffer.append(flat_features)
        
        current_prediction = "..." # Default to "..."
        
        if len(data_buffer) == WINDOW_SIZE:
            window_data = np.array(data_buffer)
            scaled_data = scaler.transform(window_data)
            scaled_data_reshaped = np.reshape(scaled_data, (1, WINDOW_SIZE, NUM_FEATURES))
            
            prediction_probs = model.predict(scaled_data_reshaped)
            
            # --- NEW Threshold Logic ---
            confidence = np.max(prediction_probs)
            if confidence > PREDICTION_THRESHOLD:
                prediction_index = np.argmax(prediction_probs)
                current_prediction = CLASS_NAMES[prediction_index]
                print(f"Prediction: {current_prediction} (Confidence: {confidence*100:.2f}%)")
            else:
                # Not confident, so stay "neutral"
                current_prediction = "..."
                # print(f"Prediction: ... (Low Confidence: {confidence*100:.2f}%)")
            # --- End New Logic ---
            
            last_prediction = current_prediction
            
        # --- UPDATED BROADCAST ---
        # Send a JSON object with BOTH prediction and raw data
        broadcast_message = {
            "prediction": last_prediction,
            "raw_data": sample_json 
        }
        await manager.broadcast(json.dumps(broadcast_message))
        # --- END UPDATED BROADCAST ---
        
        # Return the prediction to the ESP32
        return {"prediction": last_prediction}
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"prediction": "Error"}

# --- 7. Run the App ---
if __name__ == "__main__":
    print("Starting FastAPI prediction server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)