import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit as st
st.write(st.__version__)
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# LOAD MODEL & SCALER
# =========================================
with open("random_forest_model.sav", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================================
# FITUR INPUT
# =========================================
selected_features = [
    "age",
    "height_cm",
    "weight_kg",
    "heart_rate",
    "blood_pressure",
    "sleep_hours",
    "nutrition_quality",
    "activity_index",
    "smokes",
    "gender"
]

# =========================================
# JUDUL
# =========================================
st.title("🏃 Prediksi Tingkat Kebugaran")
st.write("Masukkan data kesehatan berikut untuk melakukan prediksi tingkat kebugaran.")
st.caption("⚠️ Gunakan tanda titik (.) untuk angka desimal.")
st.write(st.__version__)

# =========================================
# INFORMASI INPUT
# =========================================
feature_info = {
    "age": {
        "desc": "Masukkan usia",
        "range": "Contoh: 18 - 60 tahun"
    },
    "height_cm": {
        "desc": "Masukkan tinggi badan",
        "range": "Contoh: 150 - 190 cm"
    },
    "weight_kg": {
        "desc": "Masukkan berat badan",
        "range": "Contoh: 40 - 120 kg"
    },
    "heart_rate": {
        "desc": "Masukkan detak jantung",
        "range": "Contoh: 60 - 120 bpm"
    },
    "blood_pressure": {
        "desc": "Masukkan tekanan darah",
        "range": "Contoh: 80 - 180"
    },
    "sleep_hours": {
        "desc": "Masukkan rata-rata jam tidur",
        "range": "Contoh: 4 - 10 jam"
    },
    "nutrition_quality": {
        "desc": "Masukkan kualitas nutrisi",
        "range": "Contoh: 1 - 10"
    },
    "activity_index": {
        "desc": "Masukkan indeks aktivitas fisik",
        "range": "Contoh: 1 - 10"
    },
    "smokes": {
        "desc": "Apakah merokok",
        "range": "Pilih: Tidak (0) / Ya (1)"
    },
    "gender": {
        "desc": "Jenis kelamin",
        "range": "Pilih: Perempuan (0) / Laki-laki (1)"
    }
}

# =========================================
# FORM INPUT
# =========================================
user_input = {}

st.markdown("### 🧾 Form Input Data")

for feature in selected_features:

    if feature in feature_info:
        st.markdown(
            f"**{feature}**  \nℹ️ {feature_info[feature]['desc']} — {feature_info[feature]['range']}"
        )

    # INPUT KHUSUS
    if feature == "smokes":

        pilihan = st.selectbox(
            feature,
            ["Pilih...", "Tidak (0)", "Ya (1)"],
            label_visibility="collapsed",
            key=feature
        )

        user_input[feature] = None if pilihan == "Pilih..." else (
            1.0 if "Ya" in pilihan else 0.0
        )

    elif feature == "gender":

        pilihan = st.selectbox(
            feature,
            ["Pilih...", "Perempuan (0)", "Laki-laki (1)"],
            label_visibility="collapsed",
            key=feature
        )

        user_input[feature] = None if pilihan == "Pilih..." else (
            1.0 if "Laki-laki" in pilihan else 0.0
        )

    else:

        val = st.text_input(
            feature,
            "",
            label_visibility="collapsed",
            key=feature
        )

        if val.strip() == "":
            user_input[feature] = None

        else:
            try:
                user_input[feature] = float(val.replace(",", "."))

            except ValueError:
                st.error(f"Input {feature} harus berupa angka!")
                user_input[feature] = None

# =========================================
# BUTTON PREDIKSI
# =========================================
pred_btn = st.button("🔍 Prediksi")

# =========================================
# PREDIKSI
# =========================================
if pred_btn:

    if any(v is None for v in user_input.values()):

        st.warning("⚠️ Harap isi semua data terlebih dahulu.")

    else:
        # Membuat dataframe dari input pengguna
        input_df = pd.DataFrame(
            [user_input],
            columns=selected_features
        )
        
        # Scaling data sesuai proses training
        input_scaled = scaler.transform(input_df)
        
        # Prediksi menggunakan data yang sudah discale
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # =========================================
        # DATA YANG DIINPUT
        # =========================================
        st.subheader("📋 Data yang Diuji")

        satuan_map = {
            "age": "tahun",
            "height_cm": "cm",
            "weight_kg": "kg",
            "heart_rate": "bpm",
            "blood_pressure": "mmHg",
            "sleep_hours": "jam"
        }

        for feature, value in user_input.items():

            if feature == "smokes":
                st.write(f"**{feature}:** {int(value)} (0=Tidak, 1=Ya)")

            elif feature == "gender":

                gender_text = "Laki-laki" if value == 1 else "Perempuan"

                st.write(f"**{feature}:** {gender_text}")

            else:

                st.write(
                    f"**{feature}:** {value} {satuan_map.get(feature, '')}"
                )

        # =========================================
        # HASIL PREDIKSI
        # =========================================
        st.markdown("---")

        if prediction == 1:

            st.markdown(
                f"""
                <h2 style='text-align:center;color:#1E90FF;'>
                ✅ Hasil Prediksi: FIT
                </h2>

                <h3 style='text-align:center;'>
                Probabilitas: {probabilities[1]:.2%}
                </h3>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="
                    background-color: #ECFCFF;
                    padding: 16px;
                    border-radius: 10px;
                    border-left: 6px solid #72E4F9;
                    margin-top: 16px;">
                    
                    🏃 <strong>Rekomendasi Sistem:</strong><br><br>
                    
                    Kondisi kebugaran tergolong baik. Tetap jaga pola hidup sehat,
                    aktivitas fisik rutin, dan pola makan seimbang.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <h2 style='text-align:center;color:#FF4500;'>
                ⚠️ Hasil Prediksi: NOT FIT
                </h2>

                <h3 style='text-align:center;'>
                Probabilitas: {probabilities[0]:.2%}
                </h3>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
            <div style="background-color:#FFEAEF;padding:16px;border-radius:10px;border-left:6px solid #FF7497;">
            💡 <b>Rekomendasi Sistem:</b><br>
            Disarankan meningkatkan aktivitas fisik, menjaga pola makan,
            serta memperbaiki kualitas tidur untuk meningkatkan kebugaran tubuh.
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.caption(
            "⚠️ Sistem ini hanya berfungsi sebagai alat bantu prediksi, "
            "bukan hasil diagnosis medis."
        )

        # =========================================
        # VISUALISASI PROBABILITAS
        # =========================================
        st.subheader("📊 Visualisasi Probabilitas")

        fig, ax = plt.subplots()

        labels = ["NOT FIT", "FIT"]
        colors = ["#FFD3DD", "#BCF3FF"]

        ax.bar(labels, probabilities, color=colors)

        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)

        for i, v in enumerate(probabilities):

            ax.text(
                i,
                v + 0.02,
                f"{v:.2%}",
                ha="center",
                fontsize=10
            )

        st.pyplot(fig, clear_figure=True)
