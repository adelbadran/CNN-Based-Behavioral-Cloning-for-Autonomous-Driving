#!/usr/bin/env python3

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
from flask import Flask, send_from_directory, jsonify

# =============================
# Device & SocketIO Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sio = socketio.Server(cors_allowed_origins="*", async_mode='eventlet')
app = Flask(__name__, static_folder='.', static_url_path='')

model = None
prev_steering = 0.0
prev_throttle = 0.0
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
# SocketIO Events
# =============================
@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, prev_throttle, model, history, ARGS

    if not data:
        sio.emit('steer', {
            'steering_angle': str(0.0),
            'throttle': str(0.0)
        }, skip_sid=True)
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
            raw_steering = float(model(tensor).item())

        # تصحيح الكاميرا الجانبية
        if ARGS.camera == "left":
            raw_steering += ARGS.steer_correction
        elif ARGS.camera == "right":
            raw_steering -= ARGS.steer_correction

        curve = abs(raw_steering)

        # ===== نظام الاستجابة التكيفي حسب التصنيف =====
        
        # تصنيف المنعطف وتحديد معامل الاستجابة + تكبير الزاوية
        if curve > 0.8:
            # خطير جداً - استجابة 95% + تكبير قوي للزاوية
            category = "EXTREME"
            response = 0.95
            boost = 1.5 + (curve - 0.8) * 4.0  # تكبير قوي جداً
            
        elif curve > 0.65:
            # حاد جداً - استجابة 90% + تكبير عالي
            category = "VERY_SHARP"
            response = 0.90
            boost = 1.4 + (curve - 0.65) * 3.0  # تكبير عالي
            
        elif curve > 0.5:
            # حاد - استجابة 85% + تكبير متوسط/عالي
            category = "SHARP"
            response = 0.85
            boost = 1.3 + (curve - 0.5) * 2.5  # تكبير جيد
            
        elif curve > 0.35:
            # متوسط - استجابة 70% + تكبير متوسط
            category = "MEDIUM"
            response = 0.70
            boost = 1.2 + (curve - 0.35) * 1.8  # تكبير معتدل
            
        elif curve > 0.2:
            # خفيف - استجابة 55% + تكبير خفيف
            category = "GENTLE"
            response = 0.55
            boost = 1.1 + (curve - 0.2) * 1.2  # تكبير خفيف
            
        else:
            # مستقيم - استجابة 40% + بدون تكبير
            category = "STRAIGHT"
            response = 0.40
            boost = 1.0

        # تطبيق التكبير والاستجابة
        steering = raw_steering * boost
        steering = response * steering + (1 - response) * prev_steering
        
        prev_steering = steering
        steering = np.clip(steering, -1.0, 1.0)
        curve = abs(steering)

        # ===== سرعات منخفضة جداً حسب التصنيف =====
        
        if category == "EXTREME":
            # منعطف خطير - سرعة بطيئة جداً جداً
            if speed > 7:
                throttle = -0.5
            elif speed > 5:
                throttle = -0.1
            elif speed > 3:
                throttle = 0.05
            else:
                throttle = 0.15
            throttle_smooth = 0.85  # استجابة سريعة للفرملة
                
        elif category == "VERY_SHARP":
            # منعطف حاد جداً - سرعة منخفضة
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
            # منعطف حاد - سرعة محدودة
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
            # منعطف متوسط - سرعة معتدلة
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
            # منعطف خفيف - سرعة جيدة
            if speed > 20:
                throttle = 0.18
            elif speed > 15:
                throttle = 0.3
            else:
                throttle = 0.42
            throttle_smooth = 0.45
                
        else:  # STRAIGHT
            # مستقيم - سرعة هادئة
            if speed < 18:
                throttle = 0.52
            elif speed < 23:
                throttle = 0.38
            else:
                throttle = 0.25
            throttle_smooth = 0.35

        # تنعيم الثروتل حسب الفئة
        throttle = throttle_smooth * throttle + (1 - throttle_smooth) * prev_throttle
        prev_throttle = throttle
        throttle = np.clip(throttle, -1.0, ARGS.max_throttle)

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

        # رموز تعبيرية حسب الفئة
        emoji = {
            "EXTREME": "🔴",
            "VERY_SHARP": "🟠", 
            "SHARP": "🟡",
            "MEDIUM": "🟢",
            "GENTLE": "🔵",
            "STRAIGHT": "⚪"
        }
        
        print(f"{datetime.now().strftime('%H:%M:%S')} {emoji[category]} {category:12s} | "
              f"Rsp:{response:.0%} Bst:{boost:.2f}x | S:{steering:+.3f} | T:{throttle:+.3f} | "
              f"Spd:{speed:4.1f} | C:{curve:.3f}")

    except Exception as e:
        print("Telemetry error:", e)


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
    parser.add_argument('--steer_correction', type=float, default=0.25)
    parser.add_argument('--max_throttle', type=float, default=0.65)
    parser.add_argument('--port', type=int, default=4567)
    global ARGS
    ARGS = parser.parse_args()

    # تحميل الموديل
    model = build_nvidia_model()
    checkpoint = torch.load(ARGS.model, map_location=device)
    if isinstance(checkpoint, dict):
        key = 'model_state_dict' if 'model_state_dict' in checkpoint else \
              'state_dict' if 'state_dict' in checkpoint else None
        model.load_state_dict(checkpoint[key] if key else checkpoint)
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