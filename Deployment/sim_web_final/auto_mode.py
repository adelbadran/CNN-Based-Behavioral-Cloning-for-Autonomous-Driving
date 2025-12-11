#!/usr/bin/env python3

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
from flask import Flask, send_from_directory, jsonify

# =============================
# Device & SocketIO Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sio = socketio.Server(cors_allowed_origins="*", async_mode='eventlet')
app = Flask(__name__, static_folder='.', static_url_path='')

model = None
prev_steering = 0.0
history = []  # لتخزين كل الإطارات للـ Dashboard


# =============================
# NVIDIA Model Architecture
# =============================
class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(64*1*18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def build_nvidia_model():
    return NvidiaModel().to(device)


# =============================
# Image Preprocessing
# =============================
def preprocess(image):
    if image is None:
        raise ValueError("Image is None")
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
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2,0,1))
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)


# =============================
# Save image (optional)
# =============================
def save_image_to_folder(img_b64, folder):
    try:
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(folder, f"{timestamp}.jpg")
        img.save(path)
    except Exception as e:
        print("Save image error:", e)


# =============================
# Control helpers
# =============================
def clamp(v, mn, mx):
    return max(mn, min(mx, v))


# =============================
# SocketIO Events
# =============================
@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, model, history

    if not data:
        return

    try:
        speed = float(data.get("speed", 0.0))
        img_b64 = data["image"]

        # حفظ الصور (اختياري)
        if ARGS.image_folder:
            save_image_to_folder(img_b64, ARGS.image_folder)

        # فك تشفير الصورة
        img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        img_array = np.asarray(img)

        # التنبؤ
        tensor = preprocess(img_array)
        with torch.no_grad():
            steering = float(model(tensor).item())

        # تصحيح الكاميرا الجانبية
        if ARGS.camera == "left":
            steering += ARGS.steer_correction
        elif ARGS.camera == "right":
            steering -= ARGS.steer_correction

        # تنعيم التوجيه
        steering = ARGS.alpha * steering + (1 - ARGS.alpha) * prev_steering
        prev_steering = steering
        steering = clamp(steering, -1.0, 1.0)

        # حساب الثروتل الذكي
        curve = abs(steering)
        speed_limit = max(5.0, ARGS.max_speed - curve * 10)
        throttle = 1.0 - (speed / speed_limit)**2 if speed_limit > 0 else 0.0
        throttle = clamp(throttle, 0.0, 1.0)

        # حفظ في التاريخ للـ Dashboard
        history.append({
            "timestamp": len(history),
            "steering_angle": steering,
            "speed": speed,
            "throttle": throttle
        })

        # إرسال الأوامر للسيميولاتر
        sio.emit('steer', {
            'steering_angle': str(steering),
            'throttle': str(throttle)
        }, skip_sid=True)

        # إرسال البيانات للواجهة (الـ View + Dashboard)
        sio.emit('web_telemetry', {
            'image_b64': img_b64,
            'steering': round(steering, 4),
            'speed': round(speed, 2),
            'throttle': round(throttle, 3)
        }, skip_sid=True)

    except Exception as e:
        print("Telemetry error:", e)


@sio.on('connect')
def connect(sid, environ):
    print(f"Simulator connected: {sid}")


# =============================
# API للـ Dashboard
# =============================
@app.route('/health')
def health():
    return "OK", 200

@app.route('/stats')
def stats():
    if not history:
        return jsonify({"total_predictions":0,"avg_steering":0,"max_steering":0,"std_steering":0})
    angles = [h['steering_angle'] for h in history]
    return jsonify({
        "total_predictions": len(history),
        "avg_steering": round(np.mean(angles), 4),
        "max_steering": round(np.max(np.abs(angles)), 4),
        "std_steering": round(np.std(angles), 4)
    })

@app.route('/history')
def get_history():
    return jsonify({"history": history})


# =============================
# Serve static files (HTML, JS, assets)
# =============================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


# =============================
# Main
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to .pth model')
    parser.add_argument('--image_folder', type=str, default='', help='Folder to save images')
    parser.add_argument('--camera', type=str, default='center', choices=['center','left','right'])
    parser.add_argument('--steer_correction', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--max_speed', type=float, default=15.0)
    parser.add_argument('--port', type=int, default=4567)
    global ARGS
    ARGS = parser.parse_args()

    # تحميل الموديل
    model = build_nvidia_model()
    checkpoint = torch.load(ARGS.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded")

    if ARGS.image_folder:
        os.makedirs(ARGS.image_folder, exist_ok=True)
        print(f"Images will be saved to: {ARGS.image_folder}")

    # ربط SocketIO مع Flask
    wrapped_app = socketio.Middleware(sio, app)

    print("\n" + "="*70)
    print("سيرفر القيادة الذاتية + Dashboard شغال بنجاح!")
    print(f"افتحي المتصفح وراحي على:")
    print(f"        http://127.0.0.1:{ARGS.port}")
    print("="*70 + "\n")

    eventlet.wsgi.server(eventlet.listen(('', ARGS.port)), wrapped_app)