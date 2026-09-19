import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Food Freshness Detector",
    page_icon="🍎",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD V2 MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/best_food_freshness_model_v2.keras"
    )


model = load_model()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🍎 AI Food Freshness Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether your food is Fresh or Rotten using AI'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# UPLOAD SECTION
# ==========================================

st.subheader("📷 Upload Food Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food Image",
        width="stretch"
    )

    # Resize image
    resized_image = image.resize((160, 160))

    # Convert to array
    img_array = np.array(resized_image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    # The V2 model already performs MobileNetV2 preprocessing
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]


    # ======================================
    # RESULT
    # ======================================

    if prediction >= 0.5:

        result = "ROTTEN"
        confidence = prediction * 100

        st.error(
            "⚠️ FOOD APPEARS TO BE ROTTEN"
        )

    else:

        result = "FRESH"
        confidence = (1 - prediction) * 100

        st.success(
            "✅ FOOD APPEARS TO BE FRESH"
        )


    # ======================================
    # CONFIDENCE
    # ======================================

    st.subheader("📊 Prediction Confidence")

    st.progress(
        int(confidence)
    )

    st.write(
        f"**Confidence: {confidence:.2f}%**"
    )


    # ======================================
    # RESULT DETAILS
    # ======================================

    st.divider()

    st.subheader("🔍 Prediction Details")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction",
            result
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

st.divider()

st.subheader("📌 About the Project")

st.write(
    "This project uses a MobileNetV2-based "
    "Convolutional Neural Network to classify "
    "food images as Fresh or Rotten."
)

st.write(
    "**Model:** MobileNetV2"
)

st.write(
    "**Test Accuracy:** 95.32%"
)

st.write(
    "**Dataset:** Fruits and Vegetables Dataset (12,000 images)"
)

st.divider()

st.caption(
    "AI-Based Food Freshness Detection Using CNN"
)