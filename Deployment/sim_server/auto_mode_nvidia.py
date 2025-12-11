#!/usr/bin/env python3
"""
drive.py
Full driving script for simulator (SocketIO).
- Preprocessing matches your training pipeline (crop, RGB->YUV, blur, resize 66x200, normalize).
- Smart throttle: dynamic speed limit based on curve (max_limit default 15 km/h, min 5).
- Supports camera selection (center / left / right) with steering correction.
- Saves run images if image_folder is provided.

Usage:
    python drive.py path/to/model.pth image_folder --camera center --steer_correction 0.2
"""

import argparse
import base64
from datetime import datetime
from io import BytesIO
import os
import shutil

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

# =============================
# Device & SocketIO
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sio = socketio.Server()
app = Flask(__name__)

model = None
prev_steering = 0.0

# =============================
# NVIDIA Model (same arch you used)
# =============================
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def build_nvidia_model():
    return NvidiaModel().to(device)

# =============================
# Preprocess: must match training pipeline
# =============================
def preprocess(image):
    """
    image: numpy array in RGB (H x W x C)
    Steps (exact to training):
      - Crop y:60:135, x:0:320
      - Convert RGB -> YUV
      - GaussianBlur (3x3)
      - Resize to (200, 66)  (width, height)
      - Normalize [0,255] -> [0,1]
      - CHW and to tensor
    Returns: torch tensor (1,3,66,200) on device
    """
    if image is None:
        raise ValueError("preprocess: input image is None")

    # 1) Crop (y_min=60, y_max=135, full width)
    h, w = image.shape[:2]
    y1 = 60
    y2 = 135
    x1 = 0
    x2 = min(320, w)
    
    # Fallback for smaller images
    if h < y2 or w < x2:
        cy, cx = h // 2, w // 2
        half_h = 75 // 2
        half_w = 320 // 2
        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)

    image = image[y1:y2, x1:x2]

    # 2) RGB -> YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 3) Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 4) Resize to (width=200, height=66)
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

    # 5) Normalize to [0,1]
    image = image.astype(np.float32) / 255.0

    # 6) HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # 7) Convert to tensor and move to device
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor

# =============================
# Utility: save image (RGB) with timestamp
# =============================
def save_image_folder(img_b64, folder):
    try:
        img = Image.open(BytesIO(base64.b64decode(img_b64)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(folder, f"{ts}.jpg")
        img.save(path)
    except Exception as e:
        print("⚠️ Error saving image:", e)

# =============================
# SocketIO handlers
# =============================
def clamp(v, a, b):
    return max(a, min(b, v))

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, model, ARGS

    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return

    try:
        # Read telemetry
        speed = float(data.get("speed", 0.0))
        img_b64 = data["image"]

        # Optionally save image
        if ARGS.image_folder:
            save_image_folder(img_b64, ARGS.image_folder)

        # Load image and ensure RGB
        image = Image.open(BytesIO(base64.b64decode(img_b64)))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.asarray(image)

        # Preprocess exactly like training
        img_tensor = preprocess(image_array)

        # Predict steering
        with torch.no_grad():
            out = model(img_tensor)
            steering = float(out.item())

        # Apply camera correction if needed
        if ARGS.camera in ("left", "right") and ARGS.steer_correction != 0.0:
            if ARGS.camera == "left":
                steering += ARGS.steer_correction
            else:
                steering -= ARGS.steer_correction

        # Smooth steering to reduce jitter
        alpha = ARGS.alpha
        steering = alpha * steering + (1.0 - alpha) * prev_steering
        prev_steering = steering
        steering = clamp(steering, -1.0, 1.0)

        # ================================
        # Smart dynamic throttle (curve-based)
        # ================================
        curve_strength = abs(steering)
        dynamic_limit = max(5.0, ARGS.max_limit - curve_strength * 10.0)

        # Compute throttle
        if dynamic_limit <= 0.0:
            throttle = 0.0
        else:
            throttle = 1.0 - (speed / dynamic_limit) ** 2

        # Clamp throttle
        throttle = clamp(throttle, 0.0, ARGS.max_throttle)

        # Emergency slow
        if curve_strength > 0.9 or speed > (dynamic_limit + 5.0):
            throttle = 0.0

        # Send control
        send_control(steering, throttle)

        # Debug print
        print(
            f"[{datetime.utcnow().isoformat(timespec='seconds')}] "
            f"Steer: {steering:.3f} | Throttle: {throttle:.3f} | Speed: {speed:.2f} | "
            f"Limit: {dynamic_limit:.1f} | Camera: {ARGS.camera}"
        )

    except Exception as e:
        print("❌ Error during telemetry handling:", e)
        send_control(0.0, 0.0)

@sio.on('connect')
def connect(sid, environ):
    print("🔗 Simulator connected:", sid)
    send_control(0.0, 0.0)

def send_control(steering, throttle):
    """Emit steering and throttle to simulator."""
    sio.emit("steer", data={
        'steering_angle': str(float(steering)),
        'throttle': str(float(throttle))
    }, skip_sid=True)

# =============================
# Main
# =============================
def main():
    global model, ARGS
    
    parser = argparse.ArgumentParser(description='Remote Driving - smart drive.py')
    parser.add_argument('model', type=str, help='Path to model checkpoint file (.pth/.pt)')
    parser.add_argument('image_folder', type=str, nargs='?', default='', 
                        help='Folder to save run images (optional)')
    parser.add_argument('--camera', type=str, choices=['center','left','right'], 
                        default='center', help='Which camera stream (default: center)')
    parser.add_argument('--steer_correction', type=float, default=0.2,
                        help='Steering correction for left/right camera (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help='Steering smoothing alpha (default: 0.2)')
    parser.add_argument('--max_limit', type=float, default=15.0, 
                        help='Max speed limit for straight road in km/h (default: 15)')
    parser.add_argument('--max_throttle', type=float, default=0.8, 
                        help='Max throttle allowed 0-1 (default: 0.8)')
    parser.add_argument('--port', type=int, default=4567, 
                        help='Port to listen on (default: 4567)')
    
    args = parser.parse_args()
    ARGS = args

    # Load model
    print("📦 Loading model from:", args.model)
    model = build_nvidia_model()

    try:
        checkpoint = torch.load(args.model, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
            model.to(device)
            
    except Exception as e:
        print("⚠️ Warning: standard load failed, trying alternative method:", e)
        try:
            model = torch.jit.load(args.model, map_location=device)
            model.to(device)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model: {e2}")

    model.eval()
    print("✅ Model loaded and set to eval on", device)

    # Setup image folder
    if args.image_folder:
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        print("📸 Recording run images in:", args.image_folder)

    # Start server
    print(f"🚦 Starting server on port {args.port}...")
    app_wrapped = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', args.port)), app_wrapped)

if __name__ == "__main__":
    main()