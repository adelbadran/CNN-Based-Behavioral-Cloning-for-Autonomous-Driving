import argparse
import base64
from datetime import datetime
from io import BytesIO
import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

# ============================================================
# Device & SocketIO
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sio = socketio.Server()
app = Flask(__name__)

model = None
prev_steering = 0.0
prev_throttle = 0.0


# ============================================================
# NVIDIA Model
# ============================================================
class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3, 1), nn.ELU(),
            nn.Conv2d(64, 64, 3, 1), nn.ELU(),
            nn.Dropout(0.5)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


def build_nvidia_model():
    return NvidiaModel().to(device)


# ============================================================
# Image Preprocessing
# ============================================================
def preprocess(image):
    image = image[60:135, 0:320]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device)


# ============================================================
# Save Camera Images
# ============================================================
def save_image_folder(img_b64, folder):
    try:
        img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img.save(os.path.join(folder, f"{ts}.jpg"))
    except:
        pass


# ============================================================
# SocketIO Telemetry Event (Main Driving Loop)
# ============================================================
@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, prev_throttle, model, ARGS

    if not data:
        send_control(0, 0)
        return

    try:
        speed = float(data.get("speed", 0))
        img_b64 = data["image"]

        if ARGS.image_folder:
            save_image_folder(img_b64, ARGS.image_folder)

        img = np.asarray(Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB"))
        tensor = preprocess(img)

        with torch.no_grad():
            raw_steering = float(model(tensor).cpu().item())

        # Camera correction
        if ARGS.camera == 'left':
            raw_steering += ARGS.steer_correction
        elif ARGS.camera == 'right':
            raw_steering -= ARGS.steer_correction

        curve = abs(raw_steering)

        # ====================================================
        # Adaptive response & angle boost by curve
        # ====================================================
        if curve > 0.8:
            category = "EXTREME"
            response = 0.95
            boost = 1.5 + (curve - 0.8) * 4.0

        elif curve > 0.65:
            category = "VERY_SHARP"
            response = 0.90
            boost = 1.4 + (curve - 0.65) * 3.0

        elif curve > 0.5:
            category = "SHARP"
            response = 0.85
            boost = 1.3 + (curve - 0.5) * 2.5

        elif curve > 0.35:
            category = "MEDIUM"
            response = 0.70
            boost = 1.2 + (curve - 0.35) * 1.8

        elif curve > 0.2:
            category = "GENTLE"
            response = 0.55
            boost = 1.1 + (curve - 0.2) * 1.2

        else:
            category = "STRAIGHT"
            response = 0.40
            boost = 1.0

        # Apply angle smoothing
        steering = raw_steering * boost
        steering = response * steering + (1 - response) * prev_steering
        prev_steering = steering
        steering = np.clip(steering, -1.0, 1.0)
        curve = abs(steering)
        
        # ====================================================
        # Adaptive throttle control by curve category
        # ====================================================
        if category == "EXTREME":
            if speed > 7:
                throttle = -0.5
            elif speed > 5:
                throttle = -0.1
            elif speed > 3:
                throttle = 0.05
            else:
                throttle = 0.15
            throttle_smooth = 0.85

        elif category == "VERY_SHARP":
            if speed > 10:
                throttle = -0.35
            elif speed > 7:
                throttle = -0.05
            elif speed > 5:
                throttle = 0.08
            else:
                throttle = 0.18
            throttle_smooth = 0.8

        elif category == "SHARP":
            if speed > 13:
                throttle = -0.2
            elif speed > 9:
                throttle = 0.0
            elif speed > 7:
                throttle = 0.12
            else:
                throttle = 0.23
            throttle_smooth = 0.75

        elif category == "MEDIUM":
            if speed > 17:
                throttle = -0.05
            elif speed > 13:
                throttle = 0.12
            elif speed > 9:
                throttle = 0.22
            else:
                throttle = 0.32
            throttle_smooth = 0.6

        elif category == "GENTLE":
            if speed > 20:
                throttle = 0.18
            elif speed > 15:
                throttle = 0.3
            else:
                throttle = 0.42
            throttle_smooth = 0.45

        else:  # STRAIGHT
            if speed < 18:
                throttle = 0.52
            elif speed < 23:
                throttle = 0.38
            else:
                throttle = 0.25
            throttle_smooth = 0.35

        # Smooth throttle
        throttle = throttle_smooth * throttle + (1 - throttle_smooth) * prev_throttle
        prev_throttle = throttle
        throttle = np.clip(throttle, -1.0, ARGS.max_throttle)

        send_control(steering, throttle)

        # Logging
        emoji = {
            "EXTREME": "🔴",
            "VERY_SHARP": "🟠",
            "SHARP": "🟡",
            "MEDIUM": "🟢",
            "GENTLE": "🔵",
            "STRAIGHT": "⚪"
        }

        print(f"{datetime.now().strftime('%H:%M:%S')} {emoji[category]} {category:12s} | "
              f"Rsp:{response:.0%} Bst:{boost:.2f}x | "
              f"S:{steering:+.3f} | T:{throttle:+.3f} | "
              f"Spd:{speed:4.1f} | C:{curve:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        send_control(0, 0)


# ============================================================
# On Connect
# ============================================================
@sio.on('connect')
def connect(sid, environ):
    global prev_steering, prev_throttle

    prev_steering = 0.0
    prev_throttle = 0.0

    print("\n🎯 ENHANCED CORNERING SYSTEM")
    print("=" * 70)
    print("Category      | Response | Angle Boost | Speed Range")
    print("-" * 70)
    print("🔴 EXTREME    |   95%    |  1.5x-2.7x  | 3-7 km/h   (Max turn)")
    print("🟠 VERY_SHARP |   90%    |  1.4x-1.9x  | 5-10 km/h  (High turn)")
    print("🟡 SHARP      |   85%    |  1.3x-1.7x  | 7-13 km/h  (Strong turn)")
    print("🟢 MEDIUM     |   70%    |  1.2x-1.5x  | 9-17 km/h  (Moderate turn)")
    print("🔵 GENTLE     |   55%    |  1.1x-1.3x  | 15-20 km/h (Light turn)")
    print("⚪ STRAIGHT   |   40%    |  1.0x       | 18-23 km/h (No turn)")
    print("=" * 70 + "\n")

    send_control(0, 0)


# ============================================================
# Send Commands to Simulator
# ============================================================
def send_control(steering, throttle):
    sio.emit("steer", {
        'steering_angle': f"{steering:.6f}",
        'throttle': f"{throttle:.6f}"
    }, skip_sid=True)


# ============================================================
# Main
# ============================================================
def main():
    global model, ARGS

    parser = argparse.ArgumentParser(description="Adaptive Response Autonomous Driving")
    parser.add_argument('model', type=str, help='Path to model (.pth)')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Folder to save images')
    parser.add_argument('--camera', choices=['center', 'left', 'right'], default='center')
    parser.add_argument('--steer_correction', type=float, default=0.25)
    parser.add_argument('--max_throttle', type=float, default=0.65)
    parser.add_argument('--port', type=int, default=4567)

    ARGS = parser.parse_args()

    print(f"\n📁 Loading model: {ARGS.model}")
    model = build_nvidia_model()

    checkpoint = torch.load(ARGS.model, map_location=device)
    if isinstance(checkpoint, dict):
        key = (
            'model_state_dict'
            if 'model_state_dict' in checkpoint
            else 'state_dict'
            if 'state_dict' in checkpoint
            else None
        )
        model.load_state_dict(checkpoint[key] if key else checkpoint)
    else:
        model = checkpoint.to(device)

    model.eval()
    print("✓ Model loaded\n")

    if ARGS.image_folder:
        os.makedirs(ARGS.image_folder, exist_ok=True)
        print(f"📸 Images: {ARGS.image_folder}\n")

    print(f"🌐 Server starting on port {ARGS.port}...")
    print(f"🎯 Enhanced angle boost in curves")
    print(f"🐢 Lower speeds for safety")
    print(f"⚡ Max throttle: {ARGS.max_throttle}\n")

    eventlet.wsgi.server(
        eventlet.listen(('', ARGS.port)),
        socketio.Middleware(sio, app)
    )


if __name__ == "__main__":
    main()
