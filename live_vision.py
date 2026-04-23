import os
import cv2
import numpy as np
import tensorflow as tf
import serial
import re
from datetime import datetime

CAMERA_SOURCE = 1
ARDUINO_PORT = 'COM5'
BAUD_RATE = 9600
IMG_SIZE = (192, 192)

CLASS_NAMES = [
    'chicken_curry', 'grilled_salmon', 'pizza', 'samosa', 'garlic_bread',
    'steak', 'french_fries', 'hamburger', 'spaghetti_bolognese', 'falafel'
]

LOG_FILE = 'food_waste_log.txt'
MIN_CONFIDENCE = 0.50
MIN_WEIGHT_GRAMS = 1.0
WEIGHT_SETTLE_SECONDS = 2.0
REMOVAL_THRESHOLD_GRAMS = 0.5

try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.01)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    ser = None
    print(f"Serial Error: {e}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using {gpus[0].name}")
    except Exception as e:
        print(f"GPU Error: {e}")

def load_inference_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def parse_weight(weight_text):
    if weight_text is None:
        return None

    text = str(weight_text).strip()
    match = re.match(r"^Weight:\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def append_log(food_name, confidence, weight_grams):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    needs_separator = False

    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        with open(LOG_FILE, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            needs_separator = f.read(1) not in (b'\n', b'\r')

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        if needs_separator:
            f.write('\n')
        f.write(
            f"{timestamp} | food={food_name} | confidence={confidence:.2%} | weight={weight_grams:.2f}g\n"
        )

ensemble = []

for i in range(1, 6):
    path = f"model_fold_{i}.h5"
    if os.path.exists(path):
        try:
            m = load_inference_model(path)
            ensemble.append(m)
            print(f"[OK] Fold {i} Loaded.")
        except Exception as e:
            print(f"[!] Failed Fold {i}: {e}")
    else:
        print(f"[MISSING] {path}")

if not ensemble:
    raise SystemExit("CRITICAL: No models found.")

cap = cv2.VideoCapture(CAMERA_SOURCE)
blob = np.zeros((1, 192, 192, 3), dtype=np.float32)

current_weight = "Weight: 0.00 g"

weight_present = False
weight_detected_time = None
item_logged = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read Arduino
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                if line.lower().startswith("weight:"):
                    current_weight = line
                else:
                    print(f"[ARDUINO] {line}")
        except Exception:
            pass

    # Prepare image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob[0] = cv2.resize(rgb, IMG_SIZE) / 255.0

    # Predict
    preds = [model(blob, training=False).numpy()[0] for model in ensemble]
    avg_pred = np.mean(preds, axis=0)

    idx = np.argmax(avg_pred)
    conf = float(avg_pred[idx])
    detected_food = CLASS_NAMES[idx]

    weight_value = parse_weight(current_weight)
    now = datetime.now()

    valid_weight = (
        weight_value is not None and
        weight_value >= MIN_WEIGHT_GRAMS
    )

    removed_weight = (
        weight_value is None or
        weight_value <= REMOVAL_THRESHOLD_GRAMS
    )

    valid_detection = conf >= MIN_CONFIDENCE

    if valid_weight and not weight_present and not item_logged:
        weight_present = True
        weight_detected_time = now
        print(f"[WAIT] Weight detected. Waiting {WEIGHT_SETTLE_SECONDS:.1f}s for stabilization...")

    if weight_present and not item_logged and weight_detected_time is not None:
        elapsed = (now - weight_detected_time).total_seconds()

        if elapsed >= WEIGHT_SETTLE_SECONDS:
            if valid_weight and valid_detection:
                append_log(detected_food, conf, weight_value)
                item_logged = True
                print(f"[LOGGED] {detected_food} | {conf:.1%} | {weight_value:.2f}g")
            elif not valid_weight:
                weight_present = False
                weight_detected_time = None
                print("[RESET] Weight disappeared before stabilization.")

    if item_logged and removed_weight:
        print("[RESET] Weight removed. System re-armed for next item.")
        weight_present = False
        weight_detected_time = None
        item_logged = False

    label = f"{detected_food.upper()} ({conf:.1%})"

    if weight_value is not None:
        weight_label = f"WEIGHT: {weight_value:.1f} g"
    else:
        weight_label = f"WEIGHT: {current_weight}"

    if item_logged:
        state_label = "STATE: LOGGED - REMOVE ITEM TO RE-ARM"
        state_color = (0, 255, 255)
    elif weight_present and weight_detected_time is not None:
        elapsed = (now - weight_detected_time).total_seconds()
        remaining = max(0.0, WEIGHT_SETTLE_SECONDS - elapsed)
        state_label = f"STATE: STABILIZING ({remaining:.1f}s)"
        state_color = (0, 165, 255)
    else:
        state_label = "STATE: READY"
        state_color = (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (1100, 160), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, weight_label, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, state_label, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
    cv2.putText(
        frame,
        f"LOG RULE: wait {WEIGHT_SETTLE_SECONDS:.1f}s, log once, reset only after weight removal",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2
    )

    cv2.imshow("Wide-VGG Live Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
