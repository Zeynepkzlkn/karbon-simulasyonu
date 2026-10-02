import json
import os
import urllib.request
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
# Modern UI / CSS Stillemesi (React DOM Uyumlu Güvenli CSS)
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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* STREAMLIT ÜST MENÜ VE ÜST BARI GÜVENLİ GİZLEME */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }
    footer {
        visibility: hidden;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)# Modern UI / CSS Stillemesi (React DOM Uyumlu Güvenli CSS)
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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* STREAMLIT ÜST MENÜ VE ÜST BARI GÜVENLİ GİZLEME */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }
    footer {
        visibility: hidden;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
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

# CANLI ZİYARETÇİ SAYACI (Oturum Başında 1 Kez Çalışır)
if "visited" not in st.session_state:
  st.session_state.visited = True
  try:
    url = "https://api.counterapi.dev/v1/karbon-simulasyonu-zeynep/visits/up"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=3) as response:
      data = json.loads(response.read().decode())
      st.session_state.visitor_count = data.get("count", "---")
  except Exception:
    st.session_state.visitor_count = "---"


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

# SESSION STATE İLKELENDİRME
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

# 3. YAN MENÜ (ÜLKE TİPOLOJİLERİ VE GİRDİLER)
st.sidebar.title("🌍 Ülke Tipolojisi & Profiller")

typology = st.sidebar.selectbox(
    "Hazır Ülke Profilini Seçin:",
    [
        "--- Özel / Elle Ayarla ---",
        "🇪🇺 AB Yeşil Mutabakat Ülkesi",
        "🏭 Gelişmekte Olan Sanayi Ekonomisi",
        "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi",
        "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke",
        "🍃 Yeşil İkiz Dönüşüm Öncüsü",
        "🏛️ S0 Referans Küresel Durum (Baseline)",
    ],
)

if typology == "🇪🇺 AB Yeşil Mutabakat Ülkesi":
  st.session_state.gdp_val = 48000.0
  st.session_state.energy_val = 3.2
  st.session_state.gvc_val = 32.0
  st.session_state.trade_val = 85.0
  st.session_state.manuf_val = 14.0
  st.session_state.renew_val = 58.0
  st.session_state.broadband_val = 38.0
  st.session_state.internet_val = 92.0
  st.session_state.mobile_val = 135.0
elif typology == "🏭 Gelişmekte Olan Sanayi Ekonomisi":
  st.session_state.gdp_val = 8500.0
  st.session_state.energy_val = 8.8
  st.session_state.gvc_val = 22.0
  st.session_state.trade_val = 55.0
  st.session_state.manuf_val = 28.0
  st.session_state.renew_val = 14.0
  st.session_state.broadband_val = 12.0
  st.session_state.internet_val = 58.0
  st.session_state.mobile_val = 95.0
elif typology == "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi":
  st.session_state.gdp_val = 35000.0
  st.session_state.energy_val = 5.5
  st.session_state.gvc_val = 42.0
  st.session_state.trade_val = 130.0
  st.session_state.manuf_val = 26.0
  st.session_state.renew_val = 22.0
  st.session_state.broadband_val = 44.0
  st.session_state.internet_val = 96.0
  st.session_state.mobile_val = 160.0
elif typology == "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke":
  st.session_state.gdp_val = 3200.0
  st.session_state.energy_val = 12.5
  st.session_state.gvc_val = 12.0
  st.session_state.trade_val = 35.0
  st.session_state.manuf_val = 20.0
  st.session_state.renew_val = 8.0
  st.session_state.broadband_val = 4.0
  st.session_state.internet_val = 32.0
  st.session_state.mobile_val = 65.0
elif typology == "🍃 Yeşil İkiz Dönüşüm Öncüsü":
  st.session_state.gdp_val = 55000.0
  st.session_state.energy_val = 2.8
  st.session_state.gvc_val = 30.0
  st.session_state.trade_val = 90.0
  st.session_state.manuf_val = 15.0
  st.session_state.renew_val = 65.0
  st.session_state.broadband_val = 42.0
  st.session_state.internet_val = 95.0
  st.session_state.mobile_val = 140.0
elif typology == "🏛️ S0 Referans Küresel Durum (Baseline)":
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
st.sidebar.subheader("🎛️ Politika & Ekonomi Girdileri")

# Slider'lar
gdp = st.sidebar.slider(
    "Kişi Başı GSYH (\$)", 1000, 95000, int(st.session_state.gdp_val), step=1000
)
energy_intensity = st.sidebar.slider(
    "Enerji Yoğunluğu (MJ/\$)",
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

st.sidebar.subheader("📱 Dijital Altyapı Göstergeleri")
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

# Yan Menü Altı: Canlı Ziyaretçi Sayacı
st.sidebar.markdown("---")
count_display = st.session_state.get("visitor_count", "---")
st.sidebar.markdown(f"👁️ **Toplam Ziyaret Sayısı:** `{count_display}`")


# HESAPLAMA VE TAHMİN FONKSİYONU
def predict_emissions(
    gdp_i, energy_i, gvc_i, trade_i, manuf_i, renew_i, bb_i, net_i, mob_i
):
  z_bb = (bb_i - 18.5) / 12.5
  z_net = (net_i - 65.0) / 25.0
  z_mob = (mob_i - 105.0) / 32.0

  dig_z = np.array([[z_bb, z_net, z_mob]])
  try:
    dig_idx = float(np.asarray(pca.transform(dig_z)).item())
  except Exception:
    dig_idx = 0.0

  raw_feats = np.array(
      [[gdp_i, energy_i, gvc_i, trade_i, manuf_i, renew_i, dig_idx]]
  )
  scaled_feats = scaler.transform(raw_feats)
  pred_scaled = float(np.asarray(svr_model.predict(scaled_feats)).item())

  Y_MEAN = 520.0
  Y_STD = 530.0
  if y_scaler is not None:
    inv_p = y_scaler.inverse_transform(np.array([[pred_scaled]]))
    pred_e = float(np.asarray(inv_p).item())
  else:
    pred_e = (pred_scaled * Y_STD) + Y_MEAN
  return max(10.0, float(pred_e))


# Mevcut Senaryo Tahmini
pred_emission = predict_emissions(
    gdp,
    energy_intensity,
    gvc_output,
    trade_openness,
    manufacturing,
    renewable_energy,
    broadband,
    internet_users,
    mobile_sub,
)

# MARJİNAL ETKİ / YEREL SHAP KATKILARI (S0 Baseline Karşılaştırmalı)
c_gdp = (
    predict_emissions(
        gdp,
        BASE_ENERGY,
        BASE_GVC,
        BASE_TRADE,
        BASE_MANUF,
        BASE_RENEW,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_energy = (
    predict_emissions(
        BASE_GDP,
        energy_intensity,
        BASE_GVC,
        BASE_TRADE,
        BASE_MANUF,
        BASE_RENEW,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_gvc = (
    predict_emissions(
        BASE_GDP,
        BASE_ENERGY,
        gvc_output,
        BASE_TRADE,
        BASE_MANUF,
        BASE_RENEW,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_trade = (
    predict_emissions(
        BASE_GDP,
        BASE_ENERGY,
        BASE_GVC,
        trade_openness,
        BASE_MANUF,
        BASE_RENEW,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_manuf = (
    predict_emissions(
        BASE_GDP,
        BASE_ENERGY,
        BASE_GVC,
        BASE_TRADE,
        manufacturing,
        BASE_RENEW,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_renew = (
    predict_emissions(
        BASE_GDP,
        BASE_ENERGY,
        BASE_GVC,
        BASE_TRADE,
        BASE_MANUF,
        renewable_energy,
        BASE_BROADBAND,
        BASE_INTERNET,
        BASE_MOBILE,
    )
    - BASE_EMISSION
)
c_dig = (
    predict_emissions(
        BASE_GDP,
        BASE_ENERGY,
        BASE_GVC,
        BASE_TRADE,
        BASE_MANUF,
        BASE_RENEW,
        broadband,
        internet_users,
        mobile_sub,
    )
    - BASE_EMISSION
)

feature_names = [
    "Kişi Başı GSYH",
    "Enerji Yoğunluğu",
    "GVC Çıktısı",
    "Ticari Açıklık",
    "İmalat Sanayi",
    "Yenilenebilir Enerji",
    "Dijital Altyapı",
]
feature_contribs = [c_gdp, c_energy, c_gvc, c_trade, c_manuf, c_renew, c_dig]

# Monte Carlo Simülasyonu
np.random.seed(42)
residuals = np.random.normal(0, 101.24, 10000)
mc_distribution = pred_emission + residuals
mc_distribution = np.maximum(0.0, mc_distribution)

lower_bound = float(np.percentile(mc_distribution, 5).item())
upper_bound = float(np.percentile(mc_distribution, 95).item())

emission_diff = pred_emission - BASE_EMISSION
emission_pct_change = (emission_diff / BASE_EMISSION) * 100.0

# 5. ANA EKRAN
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
    delta_color="inverse",
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
    help="%5 olasılık alt sınırı",
)

col4.metric(
    label="Üst Güven Sınırı (%95)",
    value=f"{upper_bound:.2f} Mt CO₂eq",
    help="%95 olasılık üst sınırı",
)

st.markdown("---")

# SEKMELİ TASARIM
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Canlı Simülasyon & Risk Grafiği",
    "🧩 Yerel SHAP / Politika Katkı Analizi",
    "📊 Değişim & Tipoloji Özeti",
    "📜 Metodoloji & XAI Notları",
])

with tab1:
  st.subheader("📈 10.000 İterasyonlu Monte Carlo Olasılık Dağılımı")

  fig = go.Figure()
  fig.add_vline(
      x=BASE_EMISSION,
      line_width=2,
      line_dash="dot",
      line_color="gray",
      annotation_text=f"S0 Baseline ({BASE_EMISSION:.1f})",
      annotation_position="top left",
  )
  fig.add_trace(
      go.Histogram(
          x=mc_distribution,
          nbinsx=50,
          name="Senaryo Dağılımı",
          marker_color="#2b5c8f" if emission_diff <= 0 else "#d9534f",
          opacity=0.75,
      )
  )
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
  st.subheader("🧩 Yerel Politika Katkı (SHAP Waterfall) Grafiği")
  st.markdown("""
    Bu grafik, **S0 Referans Durumuna (822.4 Mt CO₂eq)** göre seçilen politika bileşenlerinin karbon ayak izi tahminini **ne kadar artırdığını (+ Kırmızı)** veya **ne kadar düşürdüğünü (- Yeşil)** tek tek ayrıştırır.
    """)

  fig_waterfall = go.Figure(
      go.Waterfall(
          name="Politika Katkısı",
          orientation="v",
          measure=["relative"] * len(feature_contribs) + ["total"],
          x=feature_names + ["Net Tahmin"],
          textposition="outside",
          text=[f"{c:+.1f}" for c in feature_contribs]
          + [f"{pred_emission:.1f}"],
          y=feature_contribs + [pred_emission],
          base=BASE_EMISSION,
          connector={"line": {"color": "rgb(63, 63, 63)"}},
          decreasing={"marker": {"color": "#2ca02c"}},  # Emisyon düşüşü YEŞİL
          increasing={"marker": {"color": "#d62728"}},  # Emisyon artışı KIRMIZI
          totals={"marker": {"color": "#1f77b4"}},
      )
  )

  fig_waterfall.update_layout(
      title="S0 Baseline (822.4 Mt) Üzerine Değişkenlerin Marjinal Etkileri",
      yaxis_title="Katkı Miktarı (Mt CO₂eq)",
      template="plotly_white",
      height=500,
  )
  st.plotly_chart(fig_waterfall, use_container_width=True)

  max_reducer_idx = np.argmin(feature_contribs)
  max_increaser_idx = np.argmax(feature_contribs)

  col_w1, col_w2 = st.columns(2)
  col_w1.success(
      f"🌱 **En Güçlü Karbon Düşürücü Etken:** {feature_names[max_reducer_idx]}"
      f" ({feature_contribs[max_reducer_idx]:+.2f} Mt CO₂eq)"
  )
  col_w2.error(
      f"🔥 **En Yüksek Emisyon Artırıcı Etken:**"
      f" {feature_names[max_increaser_idx]}"
      f" ({feature_contribs[max_increaser_idx]:+.2f} Mt CO₂eq)"
  )

with tab3:
  st.subheader("📋 Girdi Değişkenlerinin Referans Duruma (S0) Göre Değişimi")

  input_changes = [
      {
          "Değişken": "Kişi Başı GSYH (\$)",
          "Referans (S0)": f"{BASE_GDP:,.0f}",
          "Mevcut Senaryo": f"{gdp:,.0f}",
          "Yüzdesel Değişim": f"{((gdp-BASE_GDP)/BASE_GDP)*100:+.1f}%",
      },
      {
          "Değişken": "Enerji Yoğunluğu (MJ/\$)",
          "Referans (S0)": f"{BASE_ENERGY:.1f}",
          "Mevcut Senaryo": f"{energy_intensity:.1f}",
          "Yüzdesel Değişim": (
              f"{((energy_intensity-BASE_ENERGY)/BASE_ENERGY)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "Yenilenebilir Enerji Payı (%)",
          "Referans (S0)": f"{BASE_RENEW:.1f}%",
          "Mevcut Senaryo": f"{renewable_energy:.1f}%",
          "Yüzdesel Değişim": (
              f"{((renewable_energy-BASE_RENEW)/BASE_RENEW)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "GVC Çıktısı Payı (%)",
          "Referans (S0)": f"{BASE_GVC:.1f}%",
          "Mevcut Senaryo": f"{gvc_output:.1f}%",
          "Yüzdesel Değişim": (
              f"{((gvc_output-BASE_GVC)/BASE_GVC)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "İmalat Sanayi Payı (%)",
          "Referans (S0)": f"{BASE_MANUF:.1f}%",
          "Mevcut Senaryo": f"{manufacturing:.1f}%",
          "Yüzdesel Değişim": (
              f"{((manufacturing-BASE_MANUF)/BASE_MANUF)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "Ticari Açıklık (% GSYH)",
          "Referans (S0)": f"{BASE_TRADE:.1f}%",
          "Mevcut Senaryo": f"{trade_openness:.1f}%",
          "Yüzdesel Değişim": (
              f"{((trade_openness-BASE_TRADE)/BASE_TRADE)*100:+.1f}%"
          ),
      },
      {
          "Değişken": "İnternet Kullanım Oranı (%)",
          "Referans (S0)": f"{BASE_INTERNET:.1f}%",
          "Mevcut Senaryo": f"{internet_users:.1f}%",
          "Yüzdesel Değişim": (
              f"{((internet_users-BASE_INTERNET)/BASE_INTERNET)*100:+.1f}%"
          ),
      },
  ]

  df_changes = pd.DataFrame(input_changes)
  st.dataframe(df_changes, use_container_width=True, hide_index=True)

  st.info(
      f"💡 **Seçilen Tipoloji:**"
      f" {typology if typology != '--- Özel / Elle Ayarla ---' else 'Özel Politika Senaryosu'}\n\nMevcut"
      " politika bileşimi sonucunda, emisyonlar referans duruma göre"
      f" **{abs(emission_diff):.2f} Mt CO₂eq**"
      f" ({'azalmış' if emission_diff <= 0 else 'artmış'}) ve"
      f" **%{abs(emission_pct_change):.1f}** oranında bir değişim"
      " öngörülmüştür."
  )

with tab4:
  st.subheader("💡 Metodolojik Notlar ve Açıklanabilir Yapay Zeka (XAI)")
  st.markdown("""
    * **Tahmin Modeli:** RBF Çekirdekli Destek Vektör Regresyonu (SVR - Test \\(R^2 = 0.975\\)).
    * **Boyut İndirgeme:** Dijitalleşme göstergeleri (Sabit Genişbant, İnternet, Mobil) Temel Bileşenler Analizi (PCA) ile tek bir Dijitalleşme İndeksine dönüştürülmüştür.
    * **Yerel XAI Katkı Yöntemi:** Her bir makroekonomik değişkenin tahmine olan marjinal katkısı, diğer değişkenler S0 Baseline seviyesinde sabit tutularak SVR karar yüzeyi üzerinde tekil duyarlılık adımları ile ayrıştırılmıştır.
    * **Belirsizlik Analizi:** Modelin ampirik artık hata dağılımı (\\(RMSE = 101.24\\text{ Mt CO}_2\\text{eq}\\)) üzerinden 10.000 iterasyonlu Monte Carlo simülasyonu çalıştırılmıştır.
    * **Metodolojik Çerçeve:** Bu araç nedensel (causal) çıkarım yapmaz; makroekonomik değişkenler arasındaki **tahminsel ve ilişkisel (associative) duyarlılıkları** simüle eder.
    """)
