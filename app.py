import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Rumah Depok", page_icon="🏠", layout="wide")

@st.cache_resource
def muat_model():
    return joblib.load("model_rumah.joblib"), json.load(open("meta.json"))

@st.cache_data
def muat_median():
    return pd.read_csv("median_kec.csv", index_col=0)["median_harga"]

def rupiah(x):
    if x >= 1e9:
        return f"Rp {x/1e9:,.2f} Miliar".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {x/1e6:,.0f} Juta".replace(",", ".")

model, meta = muat_model()
median_kec = muat_median()

st.title("🏠 Prediksi Harga Rumah Depok dan Sekitarnya")
st.caption(f"Model: {meta['model']} (dilatih pada data listing rumah hasil scraping, setelah dibersihkan). "
           f"Pada data uji: MAE ≈ {rupiah(meta['mae_uji'])}, MAPE ≈ {meta['mape_uji']*100:.0f}%, R² ≈ {meta['r2_uji']:.2f}.")

tab1, tab2, tab3 = st.tabs(["Prediksi", "Penjelasan Fitur", "Visualisasi"])

with tab1:
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        kec = c1.selectbox("Kecamatan / lokasi", meta["kecamatan"])
        lt = c1.number_input("Luas tanah (m²)", min_value=float(meta["lt_range"][0]), max_value=float(meta["lt_range"][1]), value=100.0, step=5.0)
        lb = c1.number_input("Luas bangunan (m²)", min_value=float(meta["lb_range"][0]), max_value=float(meta["lb_range"][1]), value=90.0, step=5.0)
        kt = c2.number_input("Jumlah kamar tidur", min_value=1, max_value=10, value=3)
        km = c2.number_input("Jumlah kamar mandi", min_value=1, max_value=10, value=2)
        kirim = st.form_submit_button("Prediksi harga")
    if kirim:
        if lb > 3 * lt:
            st.warning("Luas bangunan lebih dari 3× luas tanah. Kombinasi ini jarang dan di luar pola data latih, hasil kurang andal.")
        x = pd.DataFrame([{"kt": kt, "km": km, "log_lt": np.log1p(lt), "log_lb": np.log1p(lb),
                           "lb_per_lt": lb / lt, "kec": kec}])
        harga = float(np.expm1(model.predict(x)[0]))
        m = meta["mape_uji"]
        st.metric("Estimasi harga", rupiah(harga))
        st.write(f"Perkiraan rentang wajar: **{rupiah(harga*(1-m))} – {rupiah(harga*(1+m))}** "
                 f"(±{m*100:.0f}%, berdasarkan MAPE model pada data uji).")
        st.caption("Ini estimasi harga penawaran listing berdasarkan pola data historis, bukan penilaian resmi (appraisal).")

with tab2:
    st.markdown("""
| Fitur | Penjelasan |
|---|---|
| Kecamatan / lokasi | Wilayah listing. Lokasi menentukan level harga. Wilayah dengan data sangat sedikit digabung menjadi "Lainnya". |
| Luas tanah (m²) | Penentu harga terkuat. Model memakai skala logaritma karena hubungannya tidak linear. |
| Luas bangunan (m²) | Penentu harga terkuat kedua, juga dalam skala logaritma. |
| Kamar tidur | Jumlah kamar tidur. Berkorelasi dengan ukuran rumah. |
| Kamar mandi | Jumlah kamar mandi. |
| Rasio bangunan/tanah | Dihitung otomatis (luas bangunan ÷ luas tanah), menggambarkan kepadatan bangunan. |
""")
    st.info("Fitur garasi tidak dipakai: pada seleksi fitur, kontribusinya paling kecil dan membuangnya tidak menurunkan performa.")

with tab3:
    st.subheader("Median harga per kecamatan")
    st.bar_chart((median_kec / 1e9).rename("Median harga (Miliar Rp)"))
    try:
        prep, rf = model.named_steps["prep"], model.named_steps["model"]
        imp = pd.Series(rf.feature_importances_, index=prep.get_feature_names_out()).sort_values()
        st.subheader("Tingkat kepentingan fitur pada model")
        st.bar_chart(imp)
    except Exception:
        pass
