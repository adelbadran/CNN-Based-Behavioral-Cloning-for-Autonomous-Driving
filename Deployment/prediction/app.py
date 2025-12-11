import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# تحديد اسم ملف الأوزان
MODEL_PATH = "nvidia_model_T1_test.pth"
# افتراض أن حجم الصورة المُدخلة بعد التجهيز هو (C, H, W)
INPUT_SHAPE = (3, 66, 200)

# ============================
# 1) Model Architecture (مع الحساب الديناميكي لحجم fc1)
# ============================

class AutonomousCarModel(nn.Module):
    """
    نموذج القيادة الذاتية (PilotNet-style) مع تحديد حجم طبقة FC الأولى ديناميكياً.
    """
    def __init__(self, input_shape=INPUT_SHAPE, fc_out=100):
        super(AutonomousCarModel, self).__init__()

        # طبقات الالتفاف (Conv Layers)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)

        # حساب حجم الـ Flatten ديناميكياً (سيكون 1152 لـ (3, 66, 200))
        try:
            num_flatten = self._get_flatten_size(input_shape)
        except Exception as e:
            # في حال فشل الحساب الديناميكي لسبب ما (قد يحدث إذا لم يتم تعريف الـ Convs بعد)
            print(f"⚠️ Dynamic size calculation failed, using default (1152): {e}")
            num_flatten = 1152 # الحجم الصحيح لـ 66x200
        
        print(f"[Model Init] Calculated Flatten features: {num_flatten}")

        # طبقات الـ FC (باستخدام الحجم المحسوب)
        self.fc1 = nn.Linear(num_flatten, fc_out)
        self.fc2 = nn.Linear(fc_out, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def _get_flatten_size(self, input_shape):
        """يحسب عدد الميزات بعد آخر طبقة Conv."""
        # يجب تعريف الـ Convs أولاً لاستخدامها
        conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        with torch.no_grad():
            # إنشاء تنسور وهمي بحجم دفعة (batch) واحد
            x = torch.zeros(1, *input_shape)
            
            for conv_layer in conv_layers:
                x = F.elu(conv_layer(x))
            
            x = x.view(1, -1)
            return x.size(1)

    def forward(self, x):
        # تمرير عبر طبقات Conv
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        # Flattening (يتحول من (Batch, C, H, W) إلى (Batch, C*H*W))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # تمرير عبر طبقات FC
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

# ============================
# 2) Load Model and Setup
# ============================

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تهيئة النموذج بالحجم الصحيح
# لاحظ: حجم fc1 سيتم حسابه الآن (وسيكون 1152)
model = AutonomousCarModel(input_shape=INPUT_SHAPE).to(device)


def load_model_safely(model, path, device):
    """
    تحميل أوزان النموذج مع تجاهل الأوزان التي لا تتطابق أحجامها (مثل طبقة fc1 بعد تغيير حجمها).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    # تحميل القاموس
    checkpoint = torch.load(path, map_location=device)
    
    # استخراج state_dict إذا كان الملف يحتوي على قاموس أكبر
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # تصفية الأوزان (Filtering): 
    # يتم تجاهل أوزان طبقات FC (fc1, fc2, fc3, fc4) لأننا غيرنا حجم fc1، 
    # وبذلك نضمن تحميل أوزان طبقات Conv فقط.
    new_state_dict = {}
    
    # مفاتيح طبقات FC التي سنقوم بتخطيها
    fc_keys_to_skip = ['fc1.weight', 'fc1.bias', 
                       'fc2.weight', 'fc2.bias', 
                       'fc3.weight', 'fc3.bias', 
                       'fc4.weight', 'fc4.bias']
                       
    for k, v in state_dict.items():
        # معالجة prefix 'module.' إذا كان الموديل قد تم تدريبه بـ DataParallel
        key = k.replace('module.', '')
        
        # تخطي مفاتيح FC غير المتوافقة
        if key not in fc_keys_to_skip:
            new_state_dict[key] = v
        else:
            print(f"Skipping incompatible layer weight: {key}")


    # محاولة التحميل بـ strict=False لتجاهل مفاتيح FC التي لم نقم بتحميلها
    load_res = model.load_state_dict(new_state_dict, strict=False)
    
    # يتم تجاهل الـ 'Missing keys' هنا لأنها ستكون طبقات FC
    if load_res.unexpected_keys:
        print(f"⚠️ Warning: Unexpected keys in checkpoint: {load_res.unexpected_keys}")
    
    print(f"Loaded {len(new_state_dict)} compatible layers successfully. Missing layers (expected for FCs): {load_res.missing_keys}")

    return model

# تحميل وتجهيز النموذج
try:
    model = load_model_safely(model, MODEL_PATH, device)
    model.eval()
    print("🔥 Model Loaded and Ready for Inference!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Exiting application. Please ensure your model file is present.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    sys.exit(1)


# ============================
# 3) FastAPI Setup
# ============================

app = FastAPI(title="Autonomous Car Steering Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 4) Image Preprocessing
# ============================

def preprocess_image(image):
    """
    تجهيز الصورة المدخلة لتتوافق مع نموذج PilotNet.
    """
    # 1) Crop (إزالة السماء وغطاء السيارة)
    image_np = np.array(image)
    # عادةً يتم اقتصاص الجزء العلوي (السماء) والجزء السفلي (غطاء السيارة)
    # لـ 66x200، قد يكون الاقتصاص من 60 إلى 135 (بارتفاع 75 بكسل)
    image_cropped = image_np[60:135, :, :]

    # 2) Resize للـ (66, 200) (النموذج يعمل مع هذا الحجم HxW)
    image_resized = Image.fromarray(image_cropped).resize((200, 66)) # W, H

    # 3) Normalize (تحويل البكسلات من [0, 255] إلى [0.0, 1.0])
    image_normalized = np.array(image_resized) / 255.0

    # 4) HWC → CHW (من Height, Width, Channel إلى Channel, Height, Width)
    image_chw = np.transpose(image_normalized, (2, 0, 1))

    # 5) إلى Tensor وإضافة بُعد الدفعة (Batch Dimension)
    tensor = torch.tensor(image_chw, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor

# ============================
# 5) Prediction Endpoint
# ============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    نقطة نهاية للتنبؤ بزاوية التوجيه من صورة كاميرا أمامية.
    """
    try:
        image_bytes = await file.read()
        # فتح وتحويل الصورة إلى RGB (لتجنب مشاكل الشفافية)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # تجهيز الصورة
        tensor_img = preprocess_image(image)

    except Exception as e:
        # رسالة خطأ واضحة في حال فشل قراءة أو تجهيز الصورة
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    # يجب أن يكون حجم التنسور الآن (1, 3, 66, 200)
    print(f"Tensor shape prepared for model: {tensor_img.shape}")
    
    with torch.no_grad():
        try:
            # التنبؤ بزاوية التوجيه
            steering = model(tensor_img).item()
            
        except Exception as e:
            # رسالة الخطأ القديمة (mat1 and mat2) لن تظهر الآن بسبب التعديل الديناميكي
            print(f"Error in model forward pass: {e}")
            raise HTTPException(status_code=500, detail=f"Model forward pass failed (Internal): {e}")

    # النتيجة هي زاوية التوجيه كقيمة عائمة
    return {"steering_angle": float(steering)}

# ============================
# 6) Run Server
# ============================

# عند التشغيل المباشر لملف البايثون، سيقوم uvicorn بتشغيل التطبيق
if __name__ == "__main__":
    # ملاحظة: تم تعديل اسم ملف التطبيق في uvicorn من "app:app" إلى اسم ملفك الحالي
    uvicorn.run("autonomous_car_api:app", host="0.0.0.0", port=8000, reload=True)