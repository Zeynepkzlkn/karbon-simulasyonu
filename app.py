import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. STREAMLIT SAYFA YAPILANDIRMASI & ÖZEL CSS TASARIMI
st.set_page_config(
    page_title="Küresel Karbon Ayak İzi Simülatörü",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI / CSS Stillemesi
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# S0 BASELINE (REFERANS DURUM) SABİTLERİ
BASE_GDP = 25000.0
BASE_ENERGY = 5.2
BASE_GVC = 25.0
BASE_TRADE = 85.0
BASE_MANUF = 18.0
BASE_RENEW = 22.0
BASE_BROADBAND = 20.0
BASE_INTERNET = 75.0
BASE_MOBILE = 110.0
BASE_EMISSION = 822.40  # Mt CO2eq


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

# 3. YAN MENÜ (HAZIR SENARYOLAR VE GİRDİLER)
st.sidebar.title("🎛️ Politika & Ekonomi Girdileri")

# Hazır Senaryo Butonları
st.sidebar.subheader("⚡ Hazır Politika Senaryoları")
preset_col1, preset_col2 = st.sidebar.columns(2)

if "gdp_val" not in st.session_state:
  st.session_state.gdp_val = BASE_GDP
  st.session_state.energy_val = BASE_ENERGY
  st.session_state.gvc_val = BASE_GVC
  st.session_state.trade_val = BASE_TRADE
  st.session_state.manuf_val = BASE_MANUF
  st.session_state.renew_val = BASE_RENEW
  st.session_state.broadband_val = BASE_BROADBAND
  st.session_state.internet_val = BASE_INTERNET
  st.session_state.mobile_val = BASE_MOBILE

if preset_col1.button("🍃 Yeşil İkiz Dönüşüm"):
  st.session_state.gdp_val = 55000.0
  st.session_state.energy_val = 3.5
  st.session_state.gvc_val = 30.0
  st.session_state.trade_val = 90.0
  st.session_state.manuf_val = 15.0
  st.session_state.renew_val = 55.0
  st.session_state.broadband_val = 35.0
  st.session_state.internet_val = 90.0
  st.session_state.mobile_val = 130.0

if preset_col2.button("🏭 Yoğun Sanayileşme"):
  st.session_state.gdp_val = 20000.0
  st.session_state.energy_val = 9.5
  st.session_state.gvc_val = 45.0
  st.session_state.trade_val = 110.0
  st.session_state.manuf_val = 32.0
  st.session_state.renew_val = 12.0
  st.session_state.broadband_val = 15.0
  st.session_state.internet_val = 60.0
  st.session_state.mobile_val = 95.0

if st.sidebar.button("🔄 Referans Duruma Sıfırla (S0 Baseline)"):
  st.session_state.gdp_val = BASE_GDP
  st.session_state.energy_val = BASE_ENERGY
  st.session_state.gvc_val = BASE_GVC
  st.session_state.trade_val = BASE_TRADE
  st.session_state.manuf_val = BASE_MANUF
  st.session_state.renew_val = BASE_RENEW
  st.session_state.broadband_val = BASE_BROADBAND
  st.session_state.internet_val = BASE_INTERNET
  st.session_state.mobile_val = BASE_MOBILE

st.sidebar.markdown("---")

# Slider'lar
gdp = st.sidebar.slider(
    "Kişi Başı GSYH ($)", 1000, 95000, int(st.session_state.gdp_val), step=1000
)
energy_intensity = st.sidebar.slider(
    "Enerji Yoğunluğu (MJ/$)",
    1.0,
    15.0,
    float(st.session_state.energy_val),
    step=0.1,
)
gvc_output = st.sidebar.slider(
    "GVC Çıktısı Payı (% Brüt Çıktı)",
    5.0,
    60.0,
    float(st.session_state.gvc_val),
    step=0.5,
)
trade_openness = st.sidebar.slider(
    "Ticari Açıklık (% GSYH)",
    20.0,
    200.0,
    float(st.session_state.trade_val),
    step=1.0,
)
manufacturing = st.sidebar.slider(
    "İmalat Sanayi Payı (% GSYH)",
    2.0,
    45.0,
    float(st.session_state.manuf_val),
    step=0.5,
)
renewable_energy = st.sidebar.slider(
    "Yenilenebilir Enerji Payı (%)",
    0.0,
    80.0,
    float(st.session_state.renew_val),
    step=1.0,
)

st.sidebar.subheader("📱 Dijital Altyapı")
broadband = st.sidebar.slider(
    "Sabit Geniş Bant (100 Kişide)",
    0.0,
    50.0,
    float(st.session_state.broadband_val),
    step=0.5,
)
internet_users = st.sidebar.slider(
    "İnternet Kullanıcı Oranı (%)",
    10.0,
    100.0,
    float(st.session_state.internet_val),
    step=1.0,
)
mobile_sub = st.sidebar.slider(
    "Mobil Abonelik (100 Kişide)",
    30.0,
    200.0,
    float(st.session_state.mobile_val),
    step=1.0,
)


# 4. HESAPLAMA MOTORU
# PCA Z-Skor Dönüşümü
z_broadband = (broadband - 18.5) / 12.5
z_internet = (internet_users - 65.0) / 25.0
z_mobile = (mobile_sub - 105.0) / 32.0

dig_z_inputs = np.array([[z_broadband, z_internet, z_mobile]])
try:
  dig_index_arr = pca.transform(dig_z_inputs)
  dig_index_val = float(np.asarray(dig_index_arr).item())
except Exception:
  dig_index_val = 0.0

raw_features = np.array([[
    gdp,
    energy_intensity,
    gvc_output,
    trade_openness,
    manufacturing,
    renewable_energy,
    dig_index_val,
]])
scaled_features = scaler.transform(raw_features)
pred_scaled_arr = svr_model.predict(scaled_features)
pred_scaled = float(np.asarray(pred_scaled_arr).item())

# Gerçek Emisyon Dönüşümü
Y_MEAN = 520.0
Y_STD = 530.0
if y_scaler is not None:
  inv_pred = y_scaler.inverse_transform(np.array([[pred_scaled]]))
  pred_emission = float(np.asarray(inv_pred).item())
else:
  pred_emission = (pred_scaled * Y_STD) + Y_MEAN

pred_emission = max(10.0, float(pred_emission))

# Monte Carlo Simülasyonu
np.random.seed(42)
residuals = np.random.normal(0, 101.24, 10000)
mc_distribution = pred_emission + residuals
mc_distribution = np.maximum(0.0, mc_distribution)

lower_bound = float(np.percentile(mc_distribution, 5).item())
upper_bound = float(np.percentile(mc_distribution, 95).item())

# Değişim Miktarları ve Yüzdeleri (vs Baseline S0)
emission_diff = pred_emission - BASE_EMISSION
emission_pct_change = (emission_diff / BASE_EMISSION) * 100.0

# 5. ANA EKRAN VE BAŞLIK
st.title("🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Sistemi")
st.caption(
    "Açıklanabilir Yapay Zeka (SVR & Monte Carlo) Destekli Politika Senaryo"
    " Analitiği"
)

# ÜST ÖZET METRİK KARTLARI
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Tahmini Karbon Ayak İzi",
    value=f"{pred_emission:.2f} Mt CO₂eq",
    delta=f"{emission_pct_change:+.1f}% vs Baseline",
    delta_color="inverse",  # Emisyon düşüşü yeşil, artışı kırmızı görünür
)

col2.metric(
    label="S0 Referans Durum (Baseline)",
    value=f"{BASE_EMISSION:.2f} Mt CO₂eq",
    delta=f"{emission_diff:+.2f} Mt CO₂eq Fark",
    delta_color="inverse",
)

col3.metric(
    label="Alt Güven Sınırı (%5)",
    value=f"{lower_bound:.2f} Mt CO₂eq",
    help="10.000 Monte Carlo simülasyonu sonucundaki %5 olasılık alt sınırı",
)

col4.metric(
    label="Üst Güven Sınırı (%95)",
    value=f"{upper_bound:.2f} Mt CO₂eq",
    help="10.000 Monte Carlo simülasyonu sonucundaki %95 olasılık üst sınırı",
)

st.markdown("---")

# SEKMELİ TASARIM
tab1, tab2, tab3 = st.tabs([
    "🎛️ Canlı Simülasyon & Risk Grafiği",
    "📊 Politika Değişim Analizi",
    "📜 Metodoloji & XAI Notları",
])

with tab1:
  st.subheader("📈 10.000 İterasyonlu Monte Carlo Olasılık Dağılımı")

  fig = go.Figure()
  # Baseline Referans Çizgisi
  fig.add_vline(
      x=BASE_EMISSION,
      line_width=2,
      line_dash="dot",
      line_color="gray",
      annotation_text=f"S0 Baseline ({BASE_EMISSION:.1f})",
      annotation_position="top left",
  )
  # Seçili Senaryo Monte Carlo Dağılımı
  fig.add_trace(
      go.Histogram(
          x=mc_distribution,
          nbinsx=50,
          name="Senaryo Dağılımı",
          marker_color="#2b5c8f" if emission_diff <= 0 else "#d9534f",
          opacity=0.75,
      )
  )
  # Seçili Senaryo Nokta Tahmini
  fig.add_vline(
      x=pred_emission,
      line_width=3,
      line_dash="dash",
      line_color="red",
      annotation_text=f"Mevcut Senaryo ({pred_emission:.1f})",
      annotation_position="top right",
  )

  fig.update_layout(
      xaxis_title="Talep Tabanlı GHG Emisyonu (Mt CO₂eq)",
      yaxis_title="Simülasyon Frekansı",
      template="plotly_white",
      height=450,
  )

  st.plotly_chart(fig, use_container_width=True)

with tab2:
  st.subheader("📋 Girdi Değişkenlerinin Referans Duruma (S0) Göre Değişimi")

  # Değişim Tablosunun Hazırlanması
  input_changes = [
      {
          "Değişken": "Kişi Başı GSYH ($)",
          "Referans (S0)": f"{BASE_GDP:,.0f}",
          "Mevcut Senaryo": f"{gdp:,.0f}",
          "Yüzdesel Değişim": f"{((gdp - BASE_GDP)/BASE_GDP)*100:+.1f}%",
      },
      {
          "Değişken": "Enerji Yoğunluğu (MJ/$)",
          "Referans (S0)": f"{BASE_ENERGY:.1f}",
          "Mevcut Senaryo": f"{energy_intensity:.1f}",
          "Yüzdesel Değişim": (
              f"{((energy_intensity - BASE_ENERGY)/BASE_ENERGY)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "Yenilenebilir Enerji Payı (%)",
          "Referans (S0)": f"{BASE_RENEW:.1f}%",
          "Mevcut Senaryo": f"{renewable_energy:.1f}%",
          "Yüzdesel Değişim": (
              f"{((renewable_energy - BASE_RENEW)/BASE_RENEW)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "GVC Çıktısı Payı (%)",
          "Referans (S0)": f"{BASE_GVC:.1f}%",
          "Mevcut Senaryo": f"{gvc_output:.1f}%",
          "Yüzdesel Değişim": (
              f"{((gvc_output - BASE_GVC)/BASE_GVC)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "İmalat Sanayi Payı (%)",
          "Referans (S0)": f"{BASE_MANUF:.1f}%",
          "Mevcut Senaryo": f"{manufacturing:.1f}%",
          "Yüzdesel Değişim": (
              f"{((manufacturing - BASE_MANUF)/BASE_MANUF)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "Ticari Açıklık (% GSYH)",
          "Referans (S0)": f"{BASE_TRADE:.1f}%",
          "Mevcut Senaryo": f"{trade_openness:.1f}%",
          "Yüzdesel Değişim": (
              f"{((trade_openness - BASE_TRADE)/BASE_TRADE)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "İnternet Kullanım Oranı (%)",
          "Referans (S0)": f"{BASE_INTERNET:.1f}%",
          "Mevcut Senaryo": f"{internet_users:.1f}%",
          "Yüzdesel Değişim": (
              f"{((internet_users - BASE_INTERNET)/BASE_INTERNET)*100:+.1f}%"
          ),
      },
  ]

  df_changes = pd.DataFrame(input_changes)
  st.dataframe(df_changes, use_container_width=True, hide_index=True)

  st.info(
      f"💡 **Senaryo Özet Yorumu:** Mevcut politika bileşimi sonucunda,"
      f" emisyonlar referans duruma göre **{abs(emission_diff):.2f} Mt CO₂eq**"
      f" ({'azalmış' if emission_diff <= 0 else 'artmış'}) ve"
      f" **%{abs(emission_pct_change):.1f}** oranında bir değişim"
      " öngörülmüştür."
  )

with tab3:
  st.subheader("💡 Metodolojik Notlar ve Açıklanabilir Yapay Zeka (XAI)")
  st.markdown("""
    * **Tahmin Modeli:** RBF Çekirdekli Destek Vektör Regresyonu (SVR - Test $R^2 = 0.975$).
    * **Boyut İndirgeme:** Dijitalleşme göstergeleri (Sabit Genişbant, İnternet, Mobil) Temel Bileşenler Analizi (PCA) ile tek bir Dijitalleşme İndeksine dönüştürülmüştür.
    * **Belirsizlik Analizi:** Modelin ampirik artık hata dağılımı ($RMSE = 101.24\text{ Mt CO}_2\text{eq}$) üzerinden 10.000 iterasyonlu Monte Carlo simülasyonu çalıştırılmıştır.
    * **Metodolojik Çerçeve:** Bu araç nedensel (causal) çıkarım yapmaz; makroekonomik değişkenler arasındaki **tahminsel ve ilişkisel (associative) duyarlılıkları** simüle eder.
    """)
