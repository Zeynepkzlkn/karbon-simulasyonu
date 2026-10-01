import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. STREAMLIT SAYFA YAPILANDIRMASI
st.set_page_config(
    page_title="Küresel Karbon Ayak İzi Simülatörü",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Simülatörü")
st.markdown("""
**Açıklanabilir Yapay Zeka (SVR & Monte Carlo)** destekli bu simülatör, makroekonomik ve dijital politika senaryolarının talep tabanlı seragazı emisyonları üzerindeki duyarlılığını canlı olarak öngörür.
""")


# 2. MODEL VE ÖLÇEKLENDİRİCİLERİ YÜKLEME
@st.cache_resource
def load_models():
  svr = joblib.load("svr_model.pkl")
  scaler = joblib.load("scaler.pkl")
  pca = joblib.load("pca.pkl")

  y_scaler = None
  if os.path.exists("y_scaler.pkl"):
    y_scaler = joblib.load("y_scaler.pkl")

  return svr, scaler, pca, y_scaler


try:
  svr_model, scaler, pca, y_scaler = load_models()
except Exception as e:
  st.error(
      "Model dosyaları yüklenemedi! Lütfen 'svr_model.pkl', 'scaler.pkl' ve"
      f" 'pca.pkl' dosyalarının GitHub reponuzda yüklü olduğundan emin olun. Hata"
      f" detayı: {e}"
  )
  st.stop()

# 3. YAN MENÜ (POLİTİKA VE EKONOMİ GİRDİLERİ)
st.sidebar.header("🎛️ Politika ve Ekonomi Girdileri")

gdp = st.sidebar.slider(
    "Kişi Başı GSYH ($)", 1000, 80000, 25000, step=1000, key="gdp"
)
energy_intensity = st.sidebar.slider(
    "Enerji Yoğunluğu (MJ/$)",
    1.0,
    15.0,
    5.2,
    step=0.1,
    key="energy_intensity",
)
gvc_output = st.sidebar.slider(
    "GVC Çıktısı ($)", 100.0, 5000.0, 1200.0, step=50.0, key="gvc_output"
)
trade_openness = st.sidebar.slider(
    "Ticari Açıklık (% GSYH)",
    20.0,
    200.0,
    85.0,
    step=1.0,
    key="trade_openness",
)
manufacturing = st.sidebar.slider(
    "İmalat Sanayi Payı (% GSYH)",
    5.0,
    45.0,
    18.0,
    step=0.5,
    key="manufacturing",
)
renewable_energy = st.sidebar.slider(
    "Yenilenebilir Enerji Payı (%)",
    0.0,
    80.0,
    22.0,
    step=1.0,
    key="renewable_energy",
)

st.sidebar.subheader("📱 Dijitalleşme Altyapı Göstergeleri")
broadband = st.sidebar.slider(
    "Sabit Geniş Bant Aboneliği (100 Kişide)",
    0.0,
    50.0,
    20.0,
    step=0.5,
    key="broadband",
)
internet_users = st.sidebar.slider(
    "İnternet Kullanıcı Oranı (%)",
    10.0,
    100.0,
    75.0,
    step=1.0,
    key="internet_users",
)
mobile_sub = st.sidebar.slider(
    "Mobil Abonelik (100 Kişide)",
    30.0,
    200.0,
    110.0,
    step=1.0,
    key="mobile_sub",
)

# 4. TAHMİN VE HESAPLAMA BUTONU
if st.button("🚀 Senaryo Tahminini ve Risk Analizini Çalıştır"):
  # A. PCA ile Dijitalleşme İndeksi Hesaplama
  dig_inputs = np.array([[broadband, internet_users, mobile_sub]])
  try:
    dig_index_arr = pca.transform(dig_inputs)
    dig_index_val = float(dig_index_arr.ravel())
  except Exception:
    dig_index_val = 0.0

  # B. Model Girdi Matrisinin Doğru Sırayla Oluşturulması:
  # 1: GDP per capita
  # 2: Energy intensity
  # 3: GVC-related Output
  # 4: Trade Openness
  # 5: Manufacturing
  # 6: Renewable Energy
  # 7: Digitalization Index
  raw_features = np.array([[
      gdp,
      energy_intensity,
      gvc_output,
      trade_openness,
      manufacturing,
      renewable_energy,
      dig_index_val,
  ]])

  # C. Ölçeklendirme ve SVR Tahmini
  scaled_features = scaler.transform(raw_features)
  pred_scaled = svr_model.predict(scaled_features)

  # D. Gerçek Emisyon Ölçeğine Dönüştürme (Mt CO₂eq)
  if y_scaler is not None:
    pred_emission = float(
        y_scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))
    )
  else:
    pred_emission = float(pred_scaled)

  # E. Monte Carlo Simülasyonu (10.000 İterasyon, RMSE = 101.24 Mt CO₂eq)
  np.random.seed(42)
  residuals = np.random.normal(0, 101.24, 10000)
  mc_distribution = pred_emission + residuals
  lower_bound = float(np.percentile(mc_distribution, 5))
  upper_bound = float(np.percentile(mc_distribution, 95))

  # F. Sonuç Kartları (Metrics)
  st.subheader("📊 Tahmin ve Risk Analizi Sonuçları")
  col1, col2, col3 = st.columns(3)
  col1.metric("Tahmini Karbon Ayak İzi", f"{pred_emission:.2f} Mt CO₂eq")
  col2.metric("Alt Güven Sınırı (%5)", f"{lower_bound:.2f} Mt CO₂eq")
  col3.metric("Üst Güven Sınırı (%95)", f"{upper_bound:.2f} Mt CO₂eq")

  # G. Olasılık Dağılımı Grafiği (Plotly)
  fig = go.Figure()
  fig.add_trace(
      go.Histogram(
          x=mc_distribution,
          nbinsx=50,
          name="Monte Carlo Dağılımı",
          marker_color="#1f77b4",
          opacity=0.75,
      )
  )
  fig.add_vline(
      x=pred_emission,
      line_width=3,
      line_dash="dash",
      line_color="red",
      annotation_text=f"Nokta Tahmin ({pred_emission:.1f})",
      annotation_position="top right",
  )
  fig.update_layout(
      title="10.000 İterasyonlu Monte Carlo Olasılık Dağılım Grafiği",
      xaxis_title="Tahmini Talep Tabanlı GHG Emisyonu (Mt CO₂eq)",
      yaxis_title="Frekans",
      template="plotly_white",
  )

  st.plotly_chart(fig, use_container_width=True)