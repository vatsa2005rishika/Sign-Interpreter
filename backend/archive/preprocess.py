import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# STEP 1: LOAD + MERGE DATA
# ==============================

base_path = "ALL_DATA"   # Folder containing A, B, C...
all_data = []

for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".parquet"):
                file_path = os.path.join(folder_path, file)

                try:
                    df = pd.read_parquet(file_path)

                    # ✅ Fix ArrowTypeError (convert timestamps safely)
                    if "ts_device" in df.columns:
                        df["ts_device"] = df["ts_device"].astype(str)
                    if "ts_server" in df.columns:
                        df["ts_server"] = df["ts_server"].astype(str)

                    # ✅ Add label
                    df["label"] = label

                    all_data.append(df)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Combine all alphabets
final_df = pd.concat(all_data, ignore_index=True)
print("Total rows:", final_df.shape)


# ==============================
# STEP 2: DROP USELESS COLUMNS
# ==============================

final_df = final_df.drop(columns=["ts_device", "ts_server"], errors="ignore")


# ==============================
# STEP 3: FLATTEN DATA
# ==============================

def flatten_row(row):
    try:
        return {
            "acc_x": row["mpu6050"]["acc"]["x"],
            "acc_y": row["mpu6050"]["acc"]["y"],
            "acc_z": row["mpu6050"]["acc"]["z"],
            "gyro_x": row["mpu6050"]["gyro"]["x"],
            "gyro_y": row["mpu6050"]["gyro"]["y"],
            "gyro_z": row["mpu6050"]["gyro"]["z"],
            "hall_1": row["halls"][0],
            "hall_2": row["halls"][1],
            "hall_3": row["halls"][2],
            "hall_4": row["halls"][3],
            "hall_5": row["halls"][4],
            "label": row["label"]
        }
    except:
        return None  # skip bad rows

flat_data = final_df.apply(flatten_row, axis=1)
flat_data = flat_data.dropna()

flat_df = pd.DataFrame(list(flat_data))

print("Flattened shape:", flat_df.shape)


# ==============================
# STEP 4: FEATURES + LABELS
# ==============================

X = flat_df.drop("label", axis=1)
y = flat_df["label"]


# ==============================
# STEP 5: NORMALIZATION
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# STEP 6: TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# STEP 7: TRAIN MODEL
# ==============================

model = RandomForestClassifier(n_estimators=150, random_state=42)

model.fit(X_train, y_train)


# ==============================
# STEP 8: PREDICTION
# ==============================

y_pred = model.predict(X_test)


# ==============================
# STEP 9: EVALUATION
# ==============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ==============================
# STEP 10: SAVE MODEL
# ==============================
