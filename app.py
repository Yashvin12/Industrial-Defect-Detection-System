"""
Industrial Defect Detection — Multi-Page Streamlit Dashboard
KaggleHacX '26 | Team: [YourTeamName]
Run: streamlit run app.py

Design guided by ui-ux.md skill:
  - Inter font >= 16px base, line-height >= 1.5
  - Contrast >= 4.5:1, semantic color tokens
  - Meaningful animations (150-300ms transforms)
  - Active state navigation indication
  - Proper chart legends & tooltips
  - Loading feedback on all actions
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "model.keras"
CLASSES = ['crack', 'hole', 'normal', 'rust', 'scratch']

CLASS_INFO = {
    'crack':   {'icon': '⚡', 'color': '#FF4B4B', 'severity': 'CRITICAL',
                'desc': 'Linear surface fracture — structural stress or fatigue.',
                'action': '🚨 Remove from production immediately. Conduct full stress analysis before reuse.'},
    'hole':    {'icon': '🕳️', 'color': '#4B7BFF', 'severity': 'CRITICAL',
                'desc': 'Circular or irregular void — corrosion or impact damage.',
                'action': '🚨 Isolate part. Assess structural integrity and determine root cause.'},
    'normal':  {'icon': '✅', 'color': '#00CC66', 'severity': 'PASS',
                'desc': 'No defect detected — surface within acceptable quality bounds.',
                'action': '✅ Surface passes quality check. Cleared for next production stage.'},
    'rust':    {'icon': '🟠', 'color': '#FF8C00', 'severity': 'WARNING',
                'desc': 'Oxidation present — moisture exposure or poor coating.',
                'action': '⚠️ Apply anti-rust treatment. Inspect coating integrity and review storage.'},
    'scratch': {'icon': '✏️', 'color': '#CC44CC', 'severity': 'WARNING',
                'desc': 'Surface abrasion — handling or mechanical contact.',
                'action': '⚠️ Assess depth and impact on specifications. Review handling procedures.'},
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Industrial Defect Detector", page_icon="🏭",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# CSS — Inter font, 16px base, semantic tokens,
# 8px spacing grid, 200ms transitions, 4.5:1 contrast
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }

/* Hero */
.hero-title {
    font-size: 3rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #FF4B4B, #C850C0 50%, #4B7BFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding: 1rem 0 0.2rem; letter-spacing: -1px; line-height: 1.2;
}
.hero-sub {
    text-align: center; color: #999; font-size: 1.1rem;
    margin-bottom: 2.5rem; line-height: 1.6;
}

/* Step Cards */
.step-card {
    text-align: center; padding: 2rem 1.5rem; border-radius: 16px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    transition: transform 200ms ease, box-shadow 200ms ease;
    min-height: 200px;
}
.step-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
.step-num {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 2px;
    color: #C850C0; text-transform: uppercase; margin-bottom: 0.5rem;
}
.step-icon { font-size: 2.5rem; margin: 0.5rem 0; }
.step-title { font-size: 1.1rem; font-weight: 700; color: #EEE; margin-bottom: 0.3rem; }
.step-desc { font-size: 0.9rem; color: #888; line-height: 1.5; }

/* Section Title */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #DDD;
    margin: 1.5rem 0 0.8rem; letter-spacing: 0.3px;
}

/* Result Card */
.result-card {
    border-radius: 16px; padding: 2rem 1.5rem; text-align: center;
    margin: 0.5rem 0 1rem; backdrop-filter: blur(12px);
    transition: transform 200ms ease;
}
.result-card:hover { transform: translateY(-2px); }
.result-icon { font-size: 3.2rem; margin-bottom: 0.3rem; }
.result-class { font-size: 2rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 0.4rem; }
.result-desc { font-size: 0.95rem; color: #AAA; line-height: 1.6; margin-top: 0.6rem; }
.severity-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-top: 0.5rem;
}
.conf-label {
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: #888; margin: 1rem 0 0.4rem;
}
.conf-bar-bg { width: 100%; height: 10px; background: #1a1a2e; border-radius: 5px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 5px; transition: width 800ms ease; }
.conf-value { font-size: 2.2rem; font-weight: 800; margin-top: 0.3rem; }

/* Action Card */
.action-card {
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: 0.5rem 0;
    font-size: 1rem; line-height: 1.6;
}

/* Landing Class Cards */
.class-card {
    text-align: center; padding: 1.2rem 0.8rem; border-radius: 14px;
    transition: transform 200ms ease;
}
.class-card:hover { transform: translateY(-4px); }
.class-card-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
.class-card-name { font-weight: 700; font-size: 0.95rem; }

/* Stat pill */
.stat-pill {
    text-align: center; padding: 1rem; border-radius: 12px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
}
.stat-value { font-size: 1.6rem; font-weight: 800; }
.stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1a, #111122); }

/* Hide radio button circles for clean nav */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer;
    transition: background 200ms ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06);
}

/* Specs table styling */
.specs-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
.specs-table td {
    padding: 0.45rem 0.6rem; font-size: 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.specs-table td:first-child { color: #888; font-weight: 500; }
.specs-table td:last-child { color: #DDD; font-weight: 700; text-align: right; }

/* Hide branding */
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL & INFERENCE
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

def preprocess_image(img_pil):
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    img = img_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    arr = preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)

def predict(model, img_pil):
    img_array = preprocess_image(img_pil)
    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], float(probs[pred_idx]), probs, img_array


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model):
    try:
        # The model consists of: Input -> Base (EfficientNetV2B0) -> Head (GAP + Dense)
        base_model = model.layers[1]
        
        # Find the last conv layer in the base model (for EfficientNetV2B0 it's usually 'top_conv')
        last_conv = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        
        if last_conv is None:
            return None

        # 1. Create a model that maps from base input to BOTH the target conv layer and the base output
        base_grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[base_model.get_layer(last_conv).output, base_model.output]
        )

        # 2. Create a model for the custom head (everything after the base model)
        head_input = tf.keras.Input(shape=base_model.output_shape[1:])
        x = head_input
        for layer in model.layers[2:]:
            x = layer(x)
        head_model = tf.keras.models.Model(inputs=head_input, outputs=x)

        # 3. Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            
            # Forward pass through the base model
            conv_out, base_out = base_grad_model(inputs, training=False)
            tape.watch(conv_out)
            
            # Forward pass through the head
            preds = head_model(base_out, training=False)
            class_idx = tf.argmax(preds[0])
            class_ch = preds[:, class_idx]

        # 4. Compute gradients
        grads = tape.gradient(class_ch, conv_out)
        if grads is None:
            print("Grad-CAM error: gradients are None")
            return None
            
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def overlay_gradcam(img_pil, heatmap):
    """Superimpose Grad-CAM heatmap on the original image."""
    img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    # Fix: mixed-precision models output float16 — cv2.resize requires float32
    heatmap = np.squeeze(np.array(heatmap, dtype=np.float32))
    hm = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(img, 0.6, hm_color, 0.4, 0))


# ─────────────────────────────────────────────
# PLOTLY CHART
# ─────────────────────────────────────────────
def build_probability_chart(all_probs, pred_class):
    labels = [f"{CLASS_INFO[c]['icon']}  {c.capitalize()}" for c in CLASSES]
    values = [round(p * 100, 2) for p in all_probs]
    colors = [CLASS_INFO[c]['color'] if c == pred_class else 'rgba(255,255,255,0.12)' for c in CLASSES]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{v:.1f}%" for v in values], textposition='outside',
        textfont=dict(color='#CCC', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>',
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, max(values) + 15], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(color='#CCC', size=13, family='Inter'),
                   autorange='reversed'),
        margin=dict(l=10, r=30, t=10, b=10), height=220,
        hoverlabel=dict(bgcolor='#1a1a2e', font_size=13, font_family='Inter'),
    )
    return fig


# ═══════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════
def render_home_page():
    st.markdown('<div class="hero-title">KaggleHacX \'26<br>Industrial Defect Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-sub">
        AI-powered metal surface quality inspection system built with
        <strong>EfficientNetV2B0</strong> transfer learning,
        mixed-precision training, and <strong>Grad-CAM</strong> explainability.
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    stats = [("5", "Defect Classes"), ("224px", "Input Resolution"),
             ("FP16", "Mixed Precision"), ("Grad-CAM", "Explainability")]
    for col, (val, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # How it Works
    st.markdown('<div class="section-title">⚙️  How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [
        ("Step 1", "📤", "Upload Image", "Capture or upload a photo of the metal surface you want to inspect."),
        ("Step 2", "🧠", "AI Analysis", "EfficientNetV2B0 classifies the surface and Grad-CAM highlights the defect region."),
        ("Step 3", "📋", "Instant Report", "Get the defect class, confidence score, probability breakdown, and recommended action."),
    ]
    for col, (num, icon, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Detectable Classes
    st.markdown('<div class="section-title">🏷️  Detectable Defect Classes</div>', unsafe_allow_html=True)
    cols = st.columns(5, gap="medium")
    for i, (cls, info) in enumerate(CLASS_INFO.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="class-card" style="
                background: {info['color']}0D; border: 1px solid {info['color']}33;
                box-shadow: 0 4px 20px {info['color']}10;">
                <div class="class-card-icon">{info['icon']}</div>
                <div class="class-card-name" style="color:{info['color']};">{cls.capitalize()}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    _, cta_col, _ = st.columns([1, 2, 1])
    with cta_col:
        st.info("👈 Select **🔍 Detection Tool** from the sidebar to start inspecting.")


# ═══════════════════════════════════════════════
# PAGE: DETECTION TOOL
# ═══════════════════════════════════════════════
def render_detection_page():
    st.markdown('<div class="hero-title" style="font-size:2.2rem;">🔬 Defect Detection Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem;">Upload a metal surface image for real-time AI inspection</div>', unsafe_allow_html=True)

    model = load_model()
    if not model:
        st.error(f"❌ Model file `{MODEL_PATH}` not found. Train the model first with `trainkaggle.py`.")
        return

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed",
                                      help="Supported: JPG, JPEG, PNG")

    show_gradcam = st.session_state.get('show_gradcam', True)

    if not uploaded_file:
        st.info("👆 Upload a metal surface image above to begin inspection.")
        return

    img_pil = Image.open(uploaded_file).convert('RGB')

    # ── Inference + Grad-CAM under one spinner ──
    with st.spinner("🔍 Scanning surface integrity and running Grad-CAM analysis..."):
        pred_class, confidence, all_probs, img_array = predict(model, img_pil)
        heatmap = make_gradcam_heatmap(img_array, model) if show_gradcam else None

    info = CLASS_INFO[pred_class]
    conf_pct = confidence * 100

    # Severity styling
    sev_map = {'CRITICAL': ('#FF4B4B', 'rgba(255,75,75,0.2)'),
               'WARNING':  ('#FF8C00', 'rgba(255,140,0,0.2)'),
               'PASS':     ('#00CC66', 'rgba(0,204,102,0.2)')}
    sev_color, sev_bg = sev_map[info['severity']]

    # ── Two-Column Layout ──
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">🖼️  Input Image</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🎯  Detection Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card" style="
            background: {info['color']}11; border: 1.5px solid {info['color']}55;
            box-shadow: 0 0 30px {info['color']}15, inset 0 0 60px {info['color']}08;">
            <div class="result-icon">{info['icon']}</div>
            <div class="result-class" style="color:{info['color']};">{pred_class.upper()}</div>
            <span class="severity-badge" style="background:{sev_bg}; color:{sev_color};">{info['severity']}</span>
            <div class="result-desc">{info['desc']}</div>
            <div class="conf-label">Confidence</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{conf_pct:.0f}%;
                    background: linear-gradient(90deg, {info['color']}88, {info['color']});"></div>
            </div>
            <div class="conf-value" style="color:{info['color']};">{conf_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Probability Distribution (Plotly) ──
    st.markdown('<div class="section-title">📊  Class Probability Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(build_probability_chart(all_probs, pred_class),
                    use_container_width=True, config={'displayModeBar': False})

    # ── Grad-CAM ──
    if show_gradcam:
        st.markdown('<div class="section-title">🔥  Grad-CAM — Defect Region Visualization</div>', unsafe_allow_html=True)
        if heatmap is not None:
            gc1, gc2 = st.columns(2, gap="medium")
            with gc1:
                st.image(img_pil.resize((IMG_SIZE, IMG_SIZE)), caption="Original", use_container_width=True)
            with gc2:
                st.image(overlay_gradcam(img_pil, heatmap), caption="Grad-CAM Overlay", use_container_width=True)
            st.info("🔴 **Warmer regions** (red/yellow) show areas that most influenced the prediction.")
        else:
            st.warning("⚠️ Grad-CAM could not be generated for this image.")

    # ── Recommended Action ──
    st.markdown('<div class="section-title">💡  Recommended Action</div>', unsafe_allow_html=True)
    act_border = sev_color
    act_bg = sev_bg.replace('0.2', '0.08')
    st.markdown(f"""
    <div class="action-card" style="background:{act_bg}; border-left:4px solid {act_border};">
        {info['action']}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SIDEBAR NAVIGATION + ROUTING
# ═══════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("##### NAVIGATE")
        page = st.radio("nav", ["Home", "Detection Tool"],
                         label_visibility="collapsed",
                         format_func=lambda x: f"{'🏠' if x == 'Home' else '🔍'}  {x}")

        st.divider()
        st.markdown("##### MODEL SPECS")
        st.markdown("""
        <table class="specs-table">
            <tr><td>Architecture</td><td>EfficientNetV2B0</td></tr>
            <tr><td>Input Size</td><td>224 × 224 px</td></tr>
            <tr><td>Precision</td><td>Mixed FP16</td></tr>
            <tr><td>Classes</td><td>5</td></tr>
            <tr><td>Explainability</td><td>Grad-CAM</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.divider()
        st.toggle("🔥 Grad-CAM Overlay", value=True, key="show_gradcam",
                  help="Highlight the regions that influenced the prediction")

        st.markdown("")
        st.caption("KaggleHacX '26 · Data Sprint to the Peak")

    if page == "Home":
        render_home_page()
    elif page == "Detection Tool":
        render_detection_page()


if __name__ == '__main__':
    main()