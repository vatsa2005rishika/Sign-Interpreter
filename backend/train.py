import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import joblib
import json
from datetime import datetime

# Suppress TensorFlow info and debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error only


# --- 1. Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ALL_DATA"
MODELS_DIR = BASE_DIR / "models" / "active"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLE_RATE_HZ = 50

# CHANGE THIS FROM 2.0 TO 1.0
WINDOW_SECONDS = 1.0  # Now 1 second window
STEP_SECONDS = 0.25   # Slide faster (overlap more)

# This will automatically calculate WINDOW_SIZE to be 50
WINDOW_SIZE = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)
STEP_SIZE = int(SAMPLE_RATE_HZ * STEP_SECONDS)

# Define the features (columns) we want to use for training
FEATURE_COLUMNS = [
    'mpu_acc_x', 'mpu_acc_y', 'mpu_acc_z',
    'mpu_gyro_x', 'mpu_gyro_y', 'mpu_gyro_z',
    'hall_0', 'hall_1', 'hall_2', 'hall_3', 'hall_4'
]
NUM_FEATURES = len(FEATURE_COLUMNS)

# --- 2. Data Loading and Processing Functions ---

def flatten_session_data(df):
    """
    Takes a raw DataFrame (with nested JSON) and flattens it.
    Returns a new DataFrame with simple columns like 'mpu_acc_x', 'hall_0', etc.
    Handles both nested JSON format and already-flattened format.
    """
    try:
        # Check if data is already in the new flattened format
        if all(col in df.columns for col in FEATURE_COLUMNS):
            print("  Data is already flattened, using as-is")
            return df[FEATURE_COLUMNS]
    except:
        pass

    try:
        # 1. Flatten mpu6050 (which is a dict column)
        if 'mpu6050' in df.columns:
            mpu_df = pd.json_normalize(df['mpu6050'])
            mpu_df.columns = [f'mpu_{c.replace(".", "_")}' for c in mpu_df.columns]
        else:
            # Alternative: maybe it's stored differently
            print(f"  Warning: 'mpu6050' column not found. Available columns: {df.columns.tolist()}")
            return None

        # 2. Flatten halls (which is a list column)
        if 'halls' in df.columns:
            halls_df = pd.DataFrame(
                df['halls'].tolist(),
                columns=[f'hall_{i}' for i in range(5)],
                index=df.index
            )
        else:
            print(f"  Warning: 'halls' column not found. Available columns: {df.columns.tolist()}")
            return None

        # 3. Combine all flattened features
        features_df = pd.concat([mpu_df, halls_df], axis=1)

        # 4. Select and reorder just the features we want
        return features_df[FEATURE_COLUMNS]
    except Exception as e:
        print(f"  Error flattening session data: {e}")
        print(f"  Available columns in DataFrame: {df.columns.tolist()}")
        return None


def create_windows(session_df, label):
    """
    Takes a single session's DataFrame and slices it into
    overlapping windows.
    """
    windows = []
    labels = []

    # Slide the window across the data
    for start in range(0, len(session_df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_data = session_df.iloc[start:end]
        windows.append(window_data.values) # as numpy array
        labels.append(label)

    return windows, labels


def load_and_process_data(data_dir):
    """
    Main function to load all data from the `data` directory,
    flatten it, and create labeled windows.
    """
    all_windows = []
    all_labels = []

    print(f"Scanning for data in: {data_dir}")
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        print(f"\nProcessing label: [{label}]")

        label_file_count = 0
        for session_file in label_dir.glob("*.parquet"):
            try:
                raw_df = pd.read_parquet(session_file)
                features_df = flatten_session_data(raw_df)

                if features_df is None or features_df.empty:
                    print(f"  Skipping {session_file.name}: Could not flatten data")
                    continue

                windows, labels = create_windows(features_df, label)
                all_windows.extend(windows)
                all_labels.extend(labels)
                label_file_count += 1
            except Exception as e:
                print(f"  Error processing {session_file.name}: {e}")
                continue

        print(f"  Loaded {label_file_count} files for '{label}'")

    print("\nData loading complete.")
    if len(all_windows) == 0:
        print("WARNING: No valid data was loaded!")
        return np.array([]), np.array([])

    X = np.array(all_windows)
    y = np.array(all_labels)

    return X, y

# --- 3. Main Training Script ---
if __name__ == "__main__":

    # 1. LOAD AND PROCESS DATA
    X, y = load_and_process_data(DATA_DIR)

    if X.shape[0] == 0:
        print("No data found! Check your DATA_DIR and file structure.")
        exit()

    print(f"\nCreated {X.shape[0]} training examples (windows).")
    print(f"Data shape (X): {X.shape}")
    print(f"Labels shape (y): {y.shape}")

    # 2. ENCODE LABELS
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes: {class_names}")

    # We must save this encoder to decode predictions later!
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print(f"Label encoder saved to {MODELS_DIR / 'label_encoder.joblib'}")

    y_categorical = to_categorical(y_encoded, num_classes)

    # 3. SPLIT DATA (Train & Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_categorical
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # 4. SCALE (NORMALIZE) DATA
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, NUM_FEATURES)
    scaler.fit(X_train_reshaped)

    # --- 2. THIS IS THE NEW PART ---
    # Save the scaler to a file
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"Scaler saved to {MODELS_DIR / 'scaler.joblib'}")
    # --- END OF NEW PART ---

    # Transform both train and test sets
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, NUM_FEATURES)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    print("\nData scaling complete.")

    # 5. BUILD THE LSTM MODEL
    model = Sequential()
    model.add(Input(shape=(WINDOW_SIZE, NUM_FEATURES)))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # 6. COMPILE THE MODEL
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. TRAIN THE MODEL
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test_scaled, y_test)
    )

    print("--- Training Complete ---")

    # 8. EVALUATE THE MODEL
    print("\n--- Evaluating Model on Test Data ---")
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 9. SHOW CONFUSION MATRIX
    y_pred_probs = model.predict(X_test_scaled)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    # 9b. GENERATE DETAILED JSON REPORT FOR FRONTEND
    print("\n--- Generating Training Report ---")

    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=range(len(class_names))
    )

    # Build per-class statistics
    class_metrics = []
    for i, class_name in enumerate(class_names):
        class_mask = y_true_labels == i
        class_accuracy = np.mean(y_pred_labels[class_mask] == y_true_labels[class_mask]) if class_mask.sum() > 0 else 0

        class_metrics.append({
            "class": class_name,
            "accuracy": float(class_accuracy),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
            "correct": int((y_pred_labels[class_mask] == y_true_labels[class_mask]).sum()) if class_mask.sum() > 0 else 0,
            "total": int(class_mask.sum())
        })

    # Calculate confusion matrix as nested dict for easier visualization
    cm_dict = {}
    for i, actual_class in enumerate(class_names):
        cm_dict[actual_class] = {}
        for j, pred_class in enumerate(class_names):
            cm_dict[actual_class][pred_class] = int(cm[i][j])

    # Get top misclassifications
    top_errors = []
    for i in range(len(y_true_labels)):
        if y_pred_labels[i] != y_true_labels[i]:
            top_errors.append({
                "actual": class_names[y_true_labels[i]],
                "predicted": class_names[y_pred_labels[i]],
                "confidence": float(np.max(y_pred_probs[i])),
                "probabilities": {class_names[j]: float(y_pred_probs[i][j]) for j in range(len(class_names))}
            })

    # Sort by confidence (highest first - most confident errors)
    top_errors = sorted(top_errors, key=lambda x: x['confidence'], reverse=True)[:10]

    # Build comprehensive report
    training_report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "models/active/sign_language_model.keras",
        "training_config": {
            "epochs": 20,
            "batch_size": 32,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "num_features": NUM_FEATURES,
            "num_classes": num_classes,
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_split": 0.2
        },
        "overall_metrics": {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "test_accuracy_percent": float(accuracy * 100)
        },
        "class_metrics": class_metrics,
        "confusion_matrix": cm_dict,
        "top_errors": top_errors,
        "training_history": {
            "epochs": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "loss_history": [float(x) for x in history.history['loss']],
            "val_loss_history": [float(x) for x in history.history['val_loss']],
            "accuracy_history": [float(x) for x in history.history['accuracy']],
            "val_accuracy_history": [float(x) for x in history.history['val_accuracy']]
        }
    }

    # Save report to JSON
    report_path = REPORTS_DIR / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2)
    print(f"Training report saved to {report_path}")

    # 10. SAVE THE MODEL (using modern Keras format)
    model_path = MODELS_DIR / "sign_language_model.keras"
    model.save(model_path)
    print(f"\nModel saved as '{model_path}'")