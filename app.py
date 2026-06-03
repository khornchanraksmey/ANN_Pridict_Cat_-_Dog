import io
import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog vs Cat Predictor",
    page_icon="🐾",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:ital,wght@0,400;0,500;0,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* hide default streamlit header gap */
.block-container { padding-top: 2rem !important; }

.title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -1.5px;
    margin-bottom: 0;
    line-height: 1.1;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    line-height: 1.5;
}

/* ── Result cards ── */
.result-wrap {
    border-radius: 20px;
    padding: 1.8rem 1.2rem 1.4rem;
    text-align: center;
    margin: 1.2rem 0 0.6rem;
}
.result-emoji { font-size: 3.5rem; line-height: 1; margin-bottom: 0.3rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}
.result-conf {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.15rem;
    opacity: 0.75;
}
.dog-card  { background: linear-gradient(135deg,#fff7ed,#ffedd5); color: #9a3412; border: 2px solid #fb923c; }
.cat-card  { background: linear-gradient(135deg,#fdf4ff,#fae8ff); color: #7e22ce; border: 2px solid #d946ef; }
.unk-card  { background: #f8fafc; color: #475569; border: 2px dashed #cbd5e1; }

/* ── Confidence meter ── */
.meter-wrap { margin: 1rem 0 0.2rem; }
.meter-row  { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem; }
.meter-icon { font-size:1.1rem; width:1.4rem; text-align:center; }
.meter-bar-bg {
    flex:1; height:14px; border-radius:99px;
    background:#f1f5f9; overflow:hidden;
}
.meter-fill {
    height:100%; border-radius:99px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.fill-dog { background: linear-gradient(90deg,#fb923c,#f97316); }
.fill-cat { background: linear-gradient(90deg,#d946ef,#a855f7); }
.meter-pct { font-size:0.82rem; font-weight:600; color:#64748b; min-width:3.2rem; text-align:right; }

.note { font-size:0.78rem; color:#94a3b8; text-align:center; margin-top:0.3rem; }
.divider { border:none; border-top:1px solid #f1f5f9; margin:1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
IMG_SIZE   = (64, 64)
INPUT_SIZE = 64 * 64 * 3
MODEL_PATH = "best_ann_model.pth"
THRESHOLD  = 50
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────
class my_ANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1  = nn.Linear(input_size, 512); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(0.3)
        self.fc2  = nn.Linear(512, 256);        self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(0.3)
        self.fc3  = nn.Linear(256, 128);        self.relu3 = nn.ReLU()
        self.fc4  = nn.Linear(128, 2)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)

@st.cache_resource
def load_model():
    m = my_ANN(INPUT_SIZE).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m.eval()
    return m

def preprocess(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv.resize(img, IMG_SIZE) / 255.0
    return torch.tensor(img.reshape(1, -1), dtype=torch.float32).to(device)

def show_result(dog_prob, cat_prob):
    top_prob = max(dog_prob, cat_prob)
    pred     = 0 if dog_prob >= cat_prob else 1

    # ── Result card ──────────────────────────────────────────────
    if top_prob < THRESHOLD:
        st.markdown(f"""
        <div class="result-wrap unk-card">
            <div class="result-emoji">❓</div>
            <div class="result-label">Not Sure</div>
            <div class="result-conf">{top_prob:.1f}% — not confident enough</div>
        </div>
        <p class="note">The model does not recognise this as a dog or cat.</p>
        """, unsafe_allow_html=True)
    elif pred == 0:
        st.markdown(f"""
        <div class="result-wrap dog-card">
            <div class="result-emoji">🐶</div>
            <div class="result-label">Dog</div>
            <div class="result-conf">{dog_prob:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-wrap cat-card">
            <div class="result-emoji">🐱</div>
            <div class="result-label">Cat</div>
            <div class="result-conf">{cat_prob:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence meter bars ─────────────────────────────────────
    st.markdown(f"""
    <div class="meter-wrap">
        <div class="meter-row">
            <span class="meter-icon">🐶</span>
            <div class="meter-bar-bg">
                <div class="meter-fill fill-dog" style="width:{dog_prob:.1f}%"></div>
            </div>
            <span class="meter-pct">{dog_prob:.1f}%</span>
        </div>
        <div class="meter-row">
            <span class="meter-icon">🐱</span>
            <div class="meter-bar-bg">
                <div class="meter-fill fill-cat" style="width:{cat_prob:.1f}%"></div>
            </div>
            <span class="meter-pct">{cat_prob:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def run_prediction(pil_img):
    col_img, col_res = st.columns([1, 1], gap="medium")

    with col_img:
        st.image(pil_img, use_container_width=True)

    with col_res:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found on server.")
            return
        with st.spinner("Analyzing…"):
            model    = load_model()
            tensor   = preprocess(pil_img)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            dog_prob = probs[0].item() * 100
            cat_prob = probs[1].item() * 100
        show_result(dog_prob, cat_prob)

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="title">🐾 Dog vs Cat</div>', unsafe_allow_html=True)
st.markdown("""
<div class="subtitle">
    Animal Classification · Artificial Neural Network (ANN)<br>
    ITC Group 02 &nbsp;·&nbsp; Lecturer: Mr. TOUCH Sopheak
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────
if "upload_img" not in st.session_state:
    st.session_state.upload_img = None
if "camera_img" not in st.session_state:
    st.session_state.camera_img = None

# ── Tabs ──────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📷 Camera"])

# ── Tab 1 : Upload ────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop or choose an image (JPG, PNG, WEBP…)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="file_uploader_main",
    )
    # Save to session_state as soon as a new file arrives
    if uploaded is not None:
        st.session_state.upload_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    # Show result from session_state (survives tab switches / reruns)
    if st.session_state.upload_img is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        run_prediction(st.session_state.upload_img)

# ── Tab 2 : Camera ────────────────────────────────────────────────
with tab_camera:
    facing = st.radio(
        "Select camera",
        ["🔭 Back camera", "🤳 Front camera"],
        horizontal=True,
        key="cam_facing",
    )
    facing_mode = "environment" if facing == "🔭 Back camera" else "user"
    cam_key     = f"cam_{facing_mode}"

    # Force correct facingMode on mobile via JS
    st.components.v1.html(f"""
    <script>
    (function() {{
        function trySetCamera() {{
            const videos = window.parent.document.querySelectorAll('video');
            if (!videos.length) {{ setTimeout(trySetCamera, 300); return; }}
            navigator.mediaDevices.getUserMedia({{
                video: {{ facingMode: {{ ideal: '{facing_mode}' }} }}
            }}).then(stream => {{
                videos.forEach(v => {{
                    if (v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
                    v.srcObject = stream;
                }});
            }}).catch(e => console.warn('Camera switch:', e));
        }}
        trySetCamera();
    }})();
    </script>
    """, height=0)

    photo = st.camera_input(
        "Take a photo of a dog or cat",
        key=cam_key,
    )
    if photo is not None:
        st.session_state.camera_img = Image.open(io.BytesIO(photo.read())).convert("RGB")

    if st.session_state.camera_img is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        run_prediction(st.session_state.camera_img)