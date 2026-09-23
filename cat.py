import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff7ed, #eff6ff);
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 30px;
}

.upload-box {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-top: 25px;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
    color: #1e293b;
}

.confidence {
    font-size: 18px;
    color: #475569;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model("cat_dog_cnn.h5")


model = load_cnn_model()


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🐱 🐶 Cat vs Dog Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image and let the CNN model identify it</div>',
    unsafe_allow_html=True
)


# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📷 Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    resized_image = image.resize((128, 128))

    image_array = np.array(resized_image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Result
    if prediction >= 0.5:
        animal = "🐶 Dog"
        confidence = prediction * 100
    else:
        animal = "🐱 Cat"
        confidence = (1 - prediction) * 100

    # Result box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="result-title">Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h2>{animal}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">Confidence: <b>{confidence:.2f}%</b></div>',
        unsafe_allow_html=True
    )

    st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Built using Python • TensorFlow • CNN • Streamlit</div>',
    unsafe_allow_html=True
)