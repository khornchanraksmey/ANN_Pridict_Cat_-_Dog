import io
import os
import base64
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Dog vs Cat Predict",
    page_icon="🐾",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── DARK ─── */
:root,
[data-theme="dark"],
.stApp[data-theme="dark"] {
    --page-bg:      #111318;
    --card-bg:      #1c1f26;
    --card-bg2:     #22262f;
    --border:       #2c303a;
    --border-hi:    #3d4250;
    --txt-h:        #f0f2f7;
    --txt-b:        #a8adbf;
    --txt-m:        #6b7080;
    --txt-lo:       #3d4250;
    --orange:       #f97316;
    --orange-hi:    #fb923c;
    --orange-dim:   rgba(249,115,22,0.15);
    --orange-glow:  rgba(249,115,22,0.25);
    --purple:       #a855f7;
    --purple-hi:    #c084fc;
    --purple-dim:   rgba(168,85,247,0.15);
    --purple-glow:  rgba(168,85,247,0.25);
    --blue:         #3b82f6;
    --blue-dim:     rgba(59,130,246,0.12);
    --green:        #22c55e;
    --dog-card-bg:  #1a1410;
    --cat-card-bg:  #160f1c;
    --unk-card-bg:  #161820;
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    18px;
    --shadow:       0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
    --shadow-sm:    0 1px 2px rgba(0,0,0,0.3);
}

/* ─── LIGHT ─── */
[data-theme="light"],
.stApp[data-theme="light"],
html[data-theme="light"],
body[data-theme="light"] {
    --page-bg:      #f4f5f7;
    --card-bg:      #ffffff;
    --card-bg2:     #f9fafb;
    --border:       #e5e7eb;
    --border-hi:    #d1d5db;
    --txt-h:        #111318;
    --txt-b:        #374151;
    --txt-m:        #6b7280;
    --txt-lo:       #d1d5db;
    --orange:       #ea6c08;
    --orange-hi:    #f97316;
    --orange-dim:   rgba(234,108,8,0.10);
    --orange-glow:  rgba(234,108,8,0.18);
    --purple:       #9333ea;
    --purple-hi:    #a855f7;
    --purple-dim:   rgba(147,51,234,0.10);
    --purple-glow:  rgba(147,51,234,0.18);
    --blue:         #2563eb;
    --blue-dim:     rgba(37,99,235,0.08);
    --green:        #16a34a;
    --dog-card-bg:  #fff8f3;
    --cat-card-bg:  #fdf5ff;
    --unk-card-bg:  #f9fafb;
    --shadow:       0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.06);
    --shadow-sm:    0 1px 2px rgba(0,0,0,0.06);
}

/* ─── GLOBAL ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--page-bg) !important; color: var(--txt-b) !important; }

[data-theme="light"] .stApp,
[data-theme="light"] .main,
[data-theme="light"] .block-container { background-color: var(--page-bg) !important; color: var(--txt-b) !important; }
[data-theme="dark"] .stApp,
[data-theme="dark"] .main,
[data-theme="dark"] .block-container { background-color: var(--page-bg) !important; color: var(--txt-b) !important; }

/* ─── BLOCK CONTAINER ─── */
.block-container {
    padding: 1rem 0.75rem !important;
    max-width: 1360px !important;
}
@media (min-width: 600px) {
    .block-container { padding: 1.5rem 1.25rem !important; }
}
@media (min-width: 1024px) {
    .block-container { padding: 1.8rem 2rem !important; }
}

/* ─── HERO ─── */
.hero {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}
.hero-icon { font-size: clamp(1.8rem, 5vw, 3rem); line-height: 1; }
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(1.7rem, 7vw, 3.2rem);
    letter-spacing: clamp(1px, 1vw, 4px);
    line-height: 1;
    color: var(--txt-h);
}
.hero-title span { color: var(--orange); }
.hero-meta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.35rem;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--card-bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(0.52rem, 1.5vw, 0.64rem);
    color: var(--txt-m);
    letter-spacing: 0.5px;
}
.badge-dot { width: 6px; height: 6px; border-radius: 99px; background: var(--green); flex-shrink: 0; }

/* ─── INFO / WARN BOXES ─── */
.info-box {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    background: var(--blue-dim);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: var(--radius-md);
    padding: 0.7rem 0.9rem;
    margin-bottom: 1rem;
}
.info-box-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.info-box-text { font-size: clamp(0.7rem, 2vw, 0.8rem); color: var(--txt-b); line-height: 1.6; }
.info-box-text strong { color: var(--txt-h); }

.warn-box {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.28);
    border-radius: var(--radius-md);
    padding: 0.7rem 0.9rem;
    margin-top: 0.9rem;
}
.warn-box-icon { font-size: 1rem; flex-shrink: 0; }
.warn-box-text { font-size: clamp(0.68rem, 2vw, 0.78rem); color: var(--txt-b); line-height: 1.6; }
.warn-box-text strong { color: var(--txt-h); }

/* ─── FIELD LABEL ─── */
.field-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--txt-m);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

/* ─── RESULT CARD ─── */
.result-wrap {
    border-radius: var(--radius-lg);
    padding: clamp(0.9rem, 3vw, 1.6rem) clamp(0.7rem, 2.5vw, 1.2rem) clamp(0.8rem, 2.5vw, 1.4rem);
    text-align: center;
    box-shadow: var(--shadow);
}
.dog-wrap { background: var(--dog-card-bg); border: 1.5px solid var(--orange); box-shadow: 0 0 0 4px var(--orange-dim), var(--shadow); }
.cat-wrap { background: var(--cat-card-bg); border: 1.5px solid var(--purple); box-shadow: 0 0 0 4px var(--purple-dim), var(--shadow); }
.unk-wrap { background: var(--unk-card-bg); border: 1.5px dashed var(--border-hi); box-shadow: var(--shadow); }

.result-emoji { font-size: clamp(2rem, 6vw, 3.2rem); display: block; margin-bottom: 0.5rem; line-height: 1; }
.result-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(1.5rem, 5vw, 2.5rem);
    letter-spacing: clamp(1px, 0.8vw, 3px);
    line-height: 1; margin-bottom: 0.3rem;
}
.dog-wrap .result-label { color: var(--orange-hi); }
.cat-wrap .result-label { color: var(--purple-hi); }
.unk-wrap .result-label { color: var(--txt-m); }
.result-conf { font-family: 'JetBrains Mono', monospace; font-size: clamp(0.62rem, 2vw, 0.75rem); color: var(--txt-m); letter-spacing: 0.5px; }

.meter-block { margin-top: 1rem; text-align: left; }
.meter-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
.meter-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--txt-m); width: 2.4rem; text-align: right; }
.meter-track { flex: 1; height: 7px; border-radius: 99px; background: var(--border); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 99px; }
.fill-dog { background: linear-gradient(90deg, #c2410c, #f97316); }
.fill-cat { background: linear-gradient(90deg, #7e22ce, #a855f7); }
.meter-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--txt-m); min-width: 3rem; text-align: right; }

/* ─── SECTION HEADER ─── */
.sect-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.9rem; }
.sect-title { font-size: 0.72rem; font-weight: 600; color: var(--txt-m); letter-spacing: 1.2px; text-transform: uppercase; }
.sect-line { flex: 1; height: 1px; background: var(--border); }

/* ─── STATS ROW ─── */
.stats-row { display: flex; gap: 0.4rem; margin-bottom: 0.9rem; flex-wrap: wrap; }
.stat-chip {
    display: flex; align-items: center; gap: 0.4rem;
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.3rem 0.55rem;
    box-shadow: var(--shadow-sm); flex: 1 1 auto; min-width: 55px;
}
.stat-chip-num { font-family: 'Bebas Neue', sans-serif; font-size: clamp(1rem, 3vw, 1.2rem); line-height: 1; }
.stat-chip-lbl { font-size: clamp(0.52rem, 1.5vw, 0.62rem); font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--txt-m); }
.chip-dog .stat-chip-num { color: var(--orange); }
.chip-cat .stat-chip-num { color: var(--purple); }
.chip-unk .stat-chip-num { color: var(--txt-m); }
.chip-tot .stat-chip-num { color: var(--txt-h); }

/* ─── HISTORY ROWS ─── */
.hist-item {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 0.5rem 0.6rem;
    margin-bottom: 0.4rem; box-shadow: var(--shadow-sm);
    transition: border-color 0.18s, box-shadow 0.18s;
}
.hist-item:hover { border-color: var(--border-hi); box-shadow: var(--shadow); }
.hist-item.is-dog:hover { border-color: var(--orange); box-shadow: 0 0 0 3px var(--orange-dim); }
.hist-item.is-cat:hover { border-color: var(--purple); box-shadow: 0 0 0 3px var(--purple-dim); }
.hist-thumb { width: 42px; height: 42px; border-radius: 8px; object-fit: cover; flex-shrink: 0; border: 1px solid var(--border); }
.hist-body { flex: 1; min-width: 0; }
.hist-pred { font-family: 'Bebas Neue', sans-serif; font-size: clamp(0.85rem, 2.5vw, 1rem); letter-spacing: 1px; line-height: 1.2; }
.hist-pred.dog { color: var(--orange-hi); }
.hist-pred.cat { color: var(--purple-hi); }
.hist-pred.unk { color: var(--txt-m); }
.hist-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--txt-m); margin-top: 0.15rem; }
.hist-minibars { width: 56px; flex-shrink: 0; }
.mini-row { display: flex; align-items: center; gap: 0.25rem; margin-bottom: 3px; }
.mini-track { flex: 1; height: 3px; border-radius: 99px; background: var(--border); overflow: hidden; }
.mini-fill { height: 100%; border-radius: 99px; }

/* ─── EMPTY STATE ─── */
.empty-state {
    background: var(--card-bg); border: 1px dashed var(--border);
    border-radius: var(--radius-lg); padding: 2rem 1.2rem;
    text-align: center; box-shadow: var(--shadow-sm);
}
.empty-state-icon { font-size: 2rem; opacity: 0.3; display: block; margin-bottom: 0.6rem; }
.empty-state-title { font-size: 0.78rem; font-weight: 600; color: var(--txt-m); letter-spacing: 0.5px; margin-bottom: 0.3rem; }
.empty-state-sub { font-size: 0.7rem; color: var(--txt-lo); line-height: 1.5; }

/* ─── RESPONSIVE OVERRIDES ─── */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: none !important;
    }
    .hist-minibars { display: none; }
    .result-wrap { margin-top: 0.75rem; }
}

/* ─── STREAMLIT OVERRIDES ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: var(--card-bg2);
    border-radius: var(--radius-md); padding: 3px;
    border: 1px solid var(--border); width: fit-content; max-width: 100%;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 0.35rem clamp(0.5rem, 2vw, 1rem);
    font-family: 'Inter', sans-serif;
    font-size: clamp(0.68rem, 2vw, 0.78rem); font-weight: 500;
    color: var(--txt-m); background: transparent; border: none;
}
.stTabs [aria-selected="true"] {
    background: var(--card-bg) !important;
    color: var(--txt-h) !important;
    box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

div[data-testid="stFileUploader"] {
    background: var(--card-bg);
    border: 1.5px dashed var(--border-hi);
    border-radius: var(--radius-md); padding: 0.6rem;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--orange); }
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small { color: var(--txt-m) !important; }

div[data-testid="stImage"] img { border-radius: var(--radius-md); border: 1px solid var(--border); width: 100%; }
.stSpinner > div { border-top-color: var(--orange) !important; }
div[data-testid="stRadio"] label { font-size: 0.8rem; color: var(--txt-b) !important; }
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--txt-b) !important; }

button[data-testid="baseButton-primary"] {
    background: var(--orange) !important;
    border: none !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 4px 14px var(--orange-glow) !important;
    transition: all 0.18s !important;
}
button[data-testid="baseButton-primary"]:hover {
    background: var(--orange-hi) !important;
    box-shadow: 0 6px 20px var(--orange-glow) !important;
    transform: translateY(-1px) !important;
}

button[data-testid="baseButton-secondary"] {
    background: var(--card-bg2) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--txt-b) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
}
button[data-testid="baseButton-secondary"]:hover { border-color: var(--orange) !important; color: var(--orange) !important; }

hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────
IMG_SIZE   = (64, 64)
INPUT_SIZE = 64 * 64 * 3
MODEL_PATH = "best_ann_model.pth"
THRESHOLD  = 65.0
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Correct Architecture Matching Your best_ann_model.pth ───────────
class SavedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1  = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2  = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3  = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        
        self.fc4  = nn.Linear(128, 2)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)

@st.cache_resource
def load_model():
    m = SavedNeuralNetwork(INPUT_SIZE).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m.eval()
    return m

def preprocess(cv2_img):
    # Process matrix through OpenCV functions directly
    img_rgb = cv.cvtColor(cv2_img, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img_rgb, IMG_SIZE)
    img_array = img_resized / 255.0
    return torch.tensor(img_array.reshape(1, -1), dtype=torch.float32).to(device)

# ── Session state ──────────────────────────────────────────────────
for k, v in {
    "upload_img": None, "camera_img": None, "history": [],
    "upload_result": None, "last_uploaded_name": None, "_last_saved": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────
def add_to_history(cv2_img, label, dog_prob, cat_prob):
    # Encode OpenCV image to jpeg format bytes for base64 distribution
    _, buf = cv.imencode(".jpg", cv2_img, [int(cv.IMWRITE_JPEG_QUALITY), 80])
    st.session_state.history.insert(0, {
        "bytes": buf.tobytes(), "label": label,
        "dog": dog_prob, "cat": cat_prob,
        "time": datetime.now().strftime("%H:%M"),
    })
    if len(st.session_state.history) > 30:
        st.session_state.history = st.session_state.history[:30]

def run_inference(cv2_img):
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: `{MODEL_PATH}`")
        return None, None
    with st.spinner("Classifying…"):
        model  = load_model()
        tensor = preprocess(cv2_img)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            
    dog_pct = probs[0].item() * 100.0
    cat_pct = probs[1].item() * 100.0
    return dog_pct, cat_pct

def show_result(cv2_img, dog_prob, cat_prob, save_history=True):
    top = max(dog_prob, cat_prob)
    if top < THRESHOLD:
        label, wrap, emoji, title = "unknown", "unk-wrap", "🚫", "Not Detected"
        conf_txt = f"{top:.1f}% — not a dog or cat"
    elif dog_prob >= cat_prob:
        label, wrap, emoji, title = "dog", "dog-wrap", "🐶", "Dog"
        conf_txt = f"{dog_prob:.1f}% confidence"
    else:
        label, wrap, emoji, title = "cat", "cat-wrap", "🐱", "Cat"
        conf_txt = f"{cat_prob:.1f}% confidence"

    if save_history:
        add_to_history(cv2_img, label, dog_prob, cat_prob)

    st.markdown(f"""
    <div class="result-wrap {wrap}">
        <span class="result-emoji">{emoji}</span>
        <div class="result-label">{title}</div>
        <div class="result-conf">{conf_txt}</div>
        <div class="meter-block">
            <div class="meter-row">
                <span class="meter-lbl">DOG</span>
                <div class="meter-track">
                    <div class="meter-fill fill-dog" style="width:{dog_prob:.1f}%"></div>
                </div>
                <span class="meter-pct">{dog_prob:.1f}%</span>
            </div>
            <div class="meter-row">
                <span class="meter-lbl">CAT</span>
                <div class="meter-track">
                    <div class="meter-fill fill-cat" style="width:{cat_prob:.1f}%"></div>
                </div>
                <span class="meter-pct">{cat_prob:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if label == "unknown":
        st.markdown("""
        <div class="warn-box">
            <span class="warn-box-icon">⚠️</span>
            <div class="warn-box-text">
                <strong>Why "Not Detected"?</strong><br>
                This model was trained only on <strong>dogs</strong> and <strong>cats</strong>.
                Photos of people, scenery, or other animals cannot be classified.
                Please try a clear, close-up photo of a dog 🐶 or cat 🐱.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-icon">🐾</div>
    <div class="hero-text">
        <div class="hero-title">DOG <span>VS</span> CAT</div>
        <div class="hero-meta">
            <span class="badge"><span class="badge-dot"></span> ANN Model</span>
            <span class="badge">ITC · Group 02</span>
            <span class="badge">Mr. Touch Sopheak</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

# ════════════════════════
# LEFT — classifier
# ════════════════════════
with left:
    tab_up, tab_cam = st.tabs(["📁  Upload Photo", "📷  Camera"])

    with tab_up:
        st.markdown("""
        <div class="info-box">
            <span class="info-box-icon">ℹ️</span>
            <div class="info-box-text">
                Upload a photo of a <strong>dog</strong> or <strong>cat</strong>,
                then click <strong>Predict</strong> to classify it.<br>
                Photos of people or other objects will return <strong>Not Detected</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="field-label">Image file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "", type=["jpg","jpeg","png","bmp","webp"],
            key="uploader", label_visibility="collapsed",
        )
        if uploaded is not None:
            # Safely transform upload byte sequences straight to OpenCV array matrices
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            
            if st.session_state.last_uploaded_name != uploaded.name:
                st.session_state.upload_img         = img
                st.session_state.last_uploaded_name = uploaded.name
                st.session_state.upload_result      = None
                st.session_state._last_saved        = None

        if st.session_state.upload_img is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            col_img, col_res = st.columns([1, 1], gap="medium")
            with col_img:
                # Convert to RGB color channel display for streamlit template execution
                disp_rgb = cv.cvtColor(st.session_state.upload_img, cv.COLOR_BGR2RGB)
                st.image(disp_rgb, use_container_width=True)
            with col_res:
                clicked = st.button("🔍  Predict", key="predict_btn",
                                    type="primary", use_container_width=True)
                if clicked:
                    d, c = run_inference(st.session_state.upload_img)
                    if d is not None:
                        st.session_state.upload_result = (d, c)

                if st.session_state.upload_result is not None:
                    d, c = st.session_state.upload_result
                    if st.session_state._last_saved != st.session_state.upload_result:
                        top = max(d, c)
                        lbl = "unknown" if top < THRESHOLD else ("dog" if d >= c else "cat")
                        add_to_history(st.session_state.upload_img, lbl, d, c)
                        st.session_state._last_saved = st.session_state.upload_result
                    show_result(st.session_state.upload_img, d, c, save_history=False)

    with tab_cam:
        st.markdown("""
        <div class="info-box">
            <span class="info-box-icon">📷</span>
            <div class="info-box-text">
                Point your camera at a <strong>dog</strong> or <strong>cat</strong>
                and take a photo — it will be classified instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        facing = st.radio("Camera direction",
                          ["🔭  Back camera", "🤳  Front camera"],
                          horizontal=True, key="cam_facing")
        facing_mode = "environment" if "Back" in facing else "user"

        st.components.v1.html(f"""
        <script>
        (function() {{
            function apply() {{
                const vids = window.parent.document.querySelectorAll('video');
                if (!vids.length) {{ setTimeout(apply, 300); return; }}
                navigator.mediaDevices.getUserMedia({{
                    video: {{ facingMode: {{ ideal: '{facing_mode}' }} }}
                }}).then(s => {{
                    vids.forEach(v => {{
                        if (v.srcObject) v.srcObject.getTracks().forEach(t => t.stop());
                        v.srcObject = s;
                    }});
                }}).catch(e => console.warn(e));
            }}
            apply();
        }})();
        </script>
        """, height=0)

        photo = st.camera_input("", key=f"cam_{facing_mode}",
                                label_visibility="collapsed")
        if photo is not None:
            cam_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
            st.session_state.camera_img = cv.imdecode(cam_bytes, cv.IMREAD_COLOR)

        if st.session_state.camera_img is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            ci, cr = st.columns([1, 1], gap="medium")
            with ci:
                cam_rgb = cv.cvtColor(st.session_state.camera_img, cv.COLOR_BGR2RGB)
                st.image(cam_rgb, use_container_width=True)
            with cr:
                d, c = run_inference(st.session_state.camera_img)
                if d is not None:
                    show_result(st.session_state.camera_img, d, c, save_history=True)

# ════════════════════════
# RIGHT — history
# ════════════════════════
with right:
    h_left, h_right = st.columns([3, 1])
    with h_left:
        st.markdown("""
        <div class="sect-head">
            <span class="sect-title">Prediction History</span>
            <span class="sect-line"></span>
        </div>
        """, unsafe_allow_html=True)
    with h_right:
        if st.session_state.history:
            if st.button("Clear", key="clear_hist"):
                st.session_state.history = []
                st.rerun()

    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">🐾</span>
            <div class="empty-state-title">No predictions yet</div>
            <div class="empty-state-sub">
                Upload a photo of a dog or cat<br>and click Predict to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        dogs  = sum(1 for h in st.session_state.history if h["label"] == "dog")
        cats  = sum(1 for h in st.session_state.history if h["label"] == "cat")
        unkns = sum(1 for h in st.session_state.history if h["label"] == "unknown")
        total = len(st.session_state.history)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-chip chip-dog">
                <span class="stat-chip-num">{dogs}</span>
                <span class="stat-chip-lbl">Dogs</span>
            </div>
            <div class="stat-chip chip-cat">
                <span class="stat-chip-num">{cats}</span>
                <span class="stat-chip-lbl">Cats</span>
            </div>
            <div class="stat-chip chip-unk">
                <span class="stat-chip-num">{unkns}</span>
                <span class="stat-chip-lbl">Unknown</span>
            </div>
            <div class="stat-chip chip-tot">
                <span class="stat-chip-num">{total}</span>
                <span class="stat-chip-lbl">Total</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for entry in st.session_state.history:
            lbl     = entry["label"]
            emoji   = "🐶" if lbl == "dog" else ("🐱" if lbl == "cat" else "🚫")
            lbl_txt = "Dog" if lbl == "dog" else ("Cat" if lbl == "cat" else "Not Detected")
            conf    = f"{max(entry['dog'], entry['cat']):.0f}% conf · {entry['time']}"
            b64     = base64.b64encode(entry["bytes"]).decode()
            dw      = entry["dog"]
            cw      = entry["cat"]
            item_cls = f"hist-item is-{lbl}" if lbl != "unknown" else "hist-item"

            st.markdown(f"""
            <div class="{item_cls}">
                <img class="hist-thumb" src="data:image/jpeg;base64,{b64}" />
                <div class="hist-body">
                    <div class="hist-pred {lbl}">{emoji} {lbl_txt}</div>
                    <div class="hist-sub">{conf}</div>
                </div>
                <div class="hist-minibars">
                    <div class="mini-row">
                        <div class="mini-track">
                            <div class="mini-fill fill-dog" style="width:{dw:.0f}%"></div>
                        </div>
                    </div>
                    <div class="mini-row">
                        <div class="mini-track">
                            <div class="mini-fill fill-cat" style="width:{cw:.0f}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)