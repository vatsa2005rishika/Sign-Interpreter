# Sign-Interpreter

A wearable ESP32 glove that streams motion and finger sensor readings to a
FastAPI server for live sign classification.

## Run the application

Requires Python 3.12. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\app.py
```

Open `http://localhost:8000` on the laptop. Before flashing the ESP32, replace
`YOUR_WIFI_SSID`, `YOUR_WIFI_PASSWORD`, and `YOUR_LAPTOP_IP` in the selected
firmware sketch with your local network settings. Keep real network credentials
out of source control. The ESP32 and laptop must share a Wi-Fi network; allow
inbound TCP port 8000 through Windows Firewall if the glove cannot reach the
server.

The backend resolves its UI, data, model, and report paths relative to
`backend\`, so it can be launched from the project root or another working
directory. The trained inference artifacts are included under
`backend\models\active`.

## Training

Labeled training sessions are intentionally not committed. To train a new
model, collect sessions into `backend\data\ALL_DATA\<label>\` using the web UI,
then select the training action in the UI. Each session should be a Parquet
file. Training replaces the active model artifacts and updates the report.

## Layout

```text
backend\
  app.py                    FastAPI server and live prediction endpoints
  train.py                  Model training
  requirements.txt          Python dependencies
  frontend\index.html       Browser UI served at /
  data\ALL_DATA\            Add labeled Parquet sessions here (not in Git)
  models\active\             Model, scaler, and label encoder used by the server
  reports\                   Latest training report and evaluation exports
firmware\                    ESP32 sketches (configure local Wi-Fi before use)
docs\                        Project documentation
.venv\                       Local Python virtual environment (not in Git)
```

The active model files are `backend\models\active\sign_language_model.keras`,
`scaler.joblib`, and `label_encoder.joblib`. Training reads labeled session
folders beneath `backend\data\ALL_DATA`, writes the active model files there,
and updates `backend\reports\training_report.json`.
