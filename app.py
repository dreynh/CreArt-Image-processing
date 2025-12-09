# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2

from processing import stylization, halftone, image_blending

st.set_page_config(page_title="CreArt — Vintage Image Studio", layout="wide")

# LOAD CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# HEADER
st.markdown("<h1>CREART — Image Processing Studio</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Stylize • Halftone • Blend</div>", unsafe_allow_html=True)
st.markdown("---")

# SIDEBAR
st.sidebar.markdown("<div class='sidebar-title'>CHOOSE A FILTER</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Select a filter to transform your image.</div>", unsafe_allow_html=True)

feature = st.sidebar.selectbox("Select A Filter:", ["Stylization", "Halftoning", "Image Blending"])

# MAIN LAYOUT (LEFT = upload/process, RIGHT = preview/result)
left_col, right_col = st.columns([1.3, 2])

# ================= LEFT COLUMN ==================
with left_col:

    st.markdown("### Upload Your Photo")
    uploaded = st.file_uploader("Image File (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if feature == "Image Blending":
        uploaded2 = st.file_uploader("Second Image", type=["jpg", "jpeg", "png"], key="blend")
        alpha = st.slider("Blend Intensity", 0.0, 1.0, 0.5)
    else:
        uploaded2 = None
        alpha = 0.5

    process_btn = st.button("▶️ PROCESS IMAGE")

    if process_btn:
        if uploaded is None:
            st.warning("Please upload an image first!")
        else:
            img_rgb = np.array(Image.open(uploaded))

            # Convert RGB to BGR for processing funcs
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if feature == "Stylization":
                out = stylization(img_bgr)

            elif feature == "Halftoning":
                out_g = halftone(img_bgr)
                out = cv2.cvtColor(out_g, cv2.COLOR_GRAY2RGB)

            elif feature == "Image Blending":
                if uploaded2 is None:
                    st.warning("Second image is required for blending.")
                    out = None
                else:
                    img2_rgb = np.array(Image.open(uploaded2))
                    img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
                    out = image_blending(img_bgr, img2_bgr, alpha)

            if out is not None:
                st.session_state["result"] = out

                buf = io.BytesIO()
                Image.fromarray(out).save(buf, format="PNG")
                st.session_state["byte"] = buf.getvalue()

# ================= RIGHT COLUMN ==================
with right_col:

    st.markdown("<h3 style='text-align:center;'>📸 Preview & Result</h3>", unsafe_allow_html=True)

    if uploaded:
        st.image(np.array(Image.open(uploaded)), caption="Preview — Original Image")

    if "result" in st.session_state:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.image(st.session_state["result"], caption="Result — CreArt")
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "📥 Download Result",
            data=st.session_state["byte"],
            file_name="creart_result.png",
            mime="image/png"
        )
