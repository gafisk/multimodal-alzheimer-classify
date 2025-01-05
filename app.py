import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit_antd_components as sac
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Fungsi untuk memuat dan memproses gambar
def load_and_preprocess_image(filepath, target_size=(224, 224)):
    img = load_img(filepath, color_mode="grayscale", target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalisasi
    return np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch


# Fungsi untuk memproses data tabular
def preprocess_tabular(sex, age):
    sex_value = 1 if sex == "Male" else 0  # Male = 1, Female = 0
    age_value = float(age)
    return np.array([[sex_value, age_value]], dtype=np.float32)


def load_model():
    # Model DenseNet untuk gambar
    base_efficientNetB1 = EfficientNetB1(weights=None, include_top=False, input_tensor=gambar_input)

    # Tambahkan layer tambahan untuk gambar
    x = Flatten()(base_efficientNetB1.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Input untuk data tabular (2 kolom)
    tabular_input = Input(shape=(2,), name="tabular_input")

    # Layer untuk tabular data
    t = Dense(64, activation="relu")(tabular_input)
    t = Dense(32, activation="relu")(t)

    # Gabungkan data gambar dan tabular
    combined = Concatenate()([x, t])

    # Layer akhir untuk prediksi
    output = Dense(len(categorical_labels[0]), activation="softmax", name="output")(combined)

    # Buat model multimodal
    model = Model(inputs=[gambar_input, tabular_input], outputs=output)

    # Kompilasi model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights('Bobot_66.h5')
    return model


model = load_model()

st.set_page_config(
    page_title="Alzheimer's Classification",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    selected = sac.menu(
        [
            sac.MenuItem("Home", icon="house"),
            sac.MenuItem("Prediksi", icon="bi bi-bar-chart-line-fill"),
        ],
        open_all=False,
    )

if selected == "Home":
    col1, col2, col3 = st.columns([2.3, 8.7, 1])
    with col1:
        st.image("Home-logo.png", width=125)
    with col2:
        col2.write("\n")
        st.title("Alzheimer's Disease")
    st.write(
        "Alzheimer Disease (Penyakit Alzheimer) adalah bentuk paling umum dari demensia yang menyebabkan penurunan fungsi otak secara progresif. Penyakit ini memengaruhi ingatan, kemampuan berpikir, bahasa, dan perilaku. Biasanya, Alzheimer menyerang individu berusia lanjut, tetapi dapat pula terjadi pada usia muda (early-onset Alzheimer)."
    )

if selected == "Prediksi":
    st.title("Prediksi Alzheimer Disease")

    # Input gambar
    uploaded_file = st.file_uploader(
        "Unggah gambar MRI otak (format: PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"]
    )

    # Input data tabular
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
    with col2:
        age = st.number_input("Usia:", min_value=1, max_value=120, value=50)

    classify_button = st.button("Klasifikasi")

    if classify_button:
        try:
            # Validasi jika file belum diunggah
            if not uploaded_file:
                st.warning("Harap unggah gambar MRI sebelum melakukan klasifikasi!")
                st.stop()

            # Proses input
            image = load_and_preprocess_image(uploaded_file)
            tabular_data = preprocess_tabular(sex, age)

            # Prediksi
            hasil = model.predict([image, tabular_data])
            predicted_class = np.argmax(hasil)
            probabilities = hasil[0]

            # Tampilkan hasil prediksi
            if predicted_class == 0:
                st.error(f"Hasil Klasifikasi: MCI (Mild Cognitive Impairment)")
            else:
                st.success(f"Hasil Klasifikasi: CN (Cognitively Normal)")

            st.write(
                f"Probabilitas Kelas MCI (Mild Cognitive Impairment): {probabilities[0] * 100:.2f}%"
            )
            st.write(
                f"Probabilitas Kelas CN (Cognitively Normal): {probabilities[1] * 100:.2f}%"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
