#!/usr/bin/env python3
"""
drive.py - Driving script for simulator (SocketIO)
- Uses ViTRegression model
- Hardcoded model path: "vit.pth"
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
import timm

# =============================
# Device & SocketIO
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sio = socketio.Server()
app = Flask(__name__)

model = None
prev_steering = 0.0
MODEL_FILENAME = "vit.pth"  # Hardcoded ViTRegression model file

# =============================
# Vision Transformer (ViT) Regression Model
# =============================
class ViTRegression(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True,
                 drop_rate=0.3, img_size=(64, 192)):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            global_pool='avg', img_size=img_size, dynamic_img_size=True
        )
        features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(features, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out.squeeze(1)

# =============================
# Model builder
# =============================
def build_model():
    return ViTRegression().to(device)

# =============================
# Preprocess
# =============================
def preprocess(image):
    h, w = image.shape[:2]
    y1, y2 = 60, 135
    x1, x2 = 0, min(320, w)
    if h < y2 or w < x2:
        cy, cx = h // 2, w // 2
        half_h, half_w = 75 // 2, 320 // 2
        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)
    image = image[y1:y2, x1:x2]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (192, 64), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor

# =============================
# Utility
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

def clamp(v, a, b):
    return max(a, min(b, v))

def send_control(steering, throttle):
    sio.emit("steer", data={
        'steering_angle': str(float(steering)),
        'throttle': str(float(throttle))
    }, skip_sid=True)

# =============================
# SocketIO events
# =============================
@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, model, ARGS
    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return
    try:
        speed = float(data.get("speed", 0.0))
        img_b64 = data["image"]
        if ARGS.image_folder:
            save_image_folder(img_b64, ARGS.image_folder)
        image = np.asarray(Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB"))
        img_tensor = preprocess(image)
        with torch.no_grad():
            steering = float(model(img_tensor).item())
        # Steering correction for left/right camera
        if ARGS.camera in ("left", "right") and ARGS.steer_correction != 0.0:
            steering += ARGS.steer_correction if ARGS.camera == "left" else -ARGS.steer_correction
        steering = ARGS.alpha * steering + (1.0 - ARGS.alpha) * prev_steering
        prev_steering = steering
        steering = clamp(steering, -1.0, 1.0)
        # Throttle control
        curve_strength = abs(steering)
        dynamic_limit = max(5.0, ARGS.max_limit - curve_strength * 10.0)
        throttle = 0.0 if dynamic_limit <= 0.0 else 1.0 - (speed / dynamic_limit) ** 2
        throttle = clamp(throttle, 0.0, ARGS.max_throttle)
        if curve_strength > 0.9 or speed > (dynamic_limit + 5.0):
            throttle = 0.0
        send_control(steering, throttle)
        print(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Steer: {steering:.3f} | Throttle: {throttle:.3f} | Speed: {speed:.2f}")
    except Exception as e:
        print("❌ Error during telemetry handling:", e)
        send_control(0.0, 0.0)

@sio.on('connect')
def connect(sid, environ):
    print("🔗 Simulator connected:", sid)
    send_control(0.0, 0.0)

# =============================
# Main
# =============================
def main():
    global model, ARGS
    parser = argparse.ArgumentParser(description='Remote Driving - ViTRegression')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Folder to save run images (optional)')
    parser.add_argument('--camera', type=str, choices=['center','left','right'], default='center')
    parser.add_argument('--steer_correction', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--max_limit', type=float, default=15.0)
    parser.add_argument('--max_throttle', type=float, default=0.8)
    parser.add_argument('--port', type=int, default=4567)
    ARGS = parser.parse_args()

    # Check model file
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Error: Could not find '{MODEL_FILENAME}'. Place it in the same folder as this script.")
        return

    # Load ViTRegression model
    print("📦 Loading ViTRegression model:", MODEL_FILENAME)
    model = build_model()
    try:
        checkpoint = torch.load(MODEL_FILENAME, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model = checkpoint.to(device)
    except Exception as e:
        print(f"❌ Failed to load model checkpoint: {e}")
        print("   Make sure to save the model using:")
        print("       torch.save({'model_state_dict': model.state_dict()}, 'vit.pth')")
        return

    model.eval()
    print("✅ Model loaded and set to eval on", device)

    # Prepare image folder
    if ARGS.image_folder:
        if os.path.exists(ARGS.image_folder):
            shutil.rmtree(ARGS.image_folder)
        os.makedirs(ARGS.image_folder)
        print("Recording run images in:", ARGS.image_folder)

    # Start server
    print(f"🚦 Starting server on port {ARGS.port} ...")
    app_wrapped = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', ARGS.port)), app_wrapped)

if __name__ == "__main__":
    main()
