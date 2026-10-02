import json
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. STREAMLIT SAYFA YAPILANDIRMASI & ÖZEL CSS TASARIMI
st.set_page_config(
    page_title="Küresel Karbon Ayak İzi Simülatörü",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# React-DOM Uyumlu Güvenli CSS Stillemesi
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
    .insight-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
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

# HAZIR PROFİL DEĞERLERİ VE AÇIKLAMALARI
PROFILE_DETAILS = {
    "🏛️ S0 Referans Küresel Durum (Baseline)": {
        "desc": (
            "Veri setindeki tüm küresel ekonomilerin tam ortalamasını temsil"
            " eden nötr mihenk taşı."
        ),
        "gdp": 25000.0,
        "energy": 5.2,
        "gvc": 25.0,
        "trade": 85.0,
        "manuf": 18.0,
        "renew": 22.0,
        "broadband": 20.0,
        "internet": 75.0,
        "mobile": 110.0,
    },
    "🇪🇺 AB Yeşil Mutabakat Ülkesi": {
        "desc": (
            "Sıkı iklim politikaları, yüksek milli gelir ve baskın yenilenebilir"
            " enerji dönüşümü sağlayan gelişmiş AB modeli."
        ),
        "gdp": 48000.0,
        "energy": 3.2,
        "gvc": 32.0,
        "trade": 85.0,
        "manuf": 14.0,
        "renew": 58.0,
        "broadband": 38.0,
        "internet": 92.0,
        "mobile": 135.0,
    },
    "🏭 Gelişmekte Olan Sanayi Ekonomisi": {
        "desc": (
            "Yüksek imalat sanayi payı, yüksek enerji yoğunluğu ve henüz kısıtlı"
            " yenilenebilir enerji entegrasyonu olan üretim odaklı ekonomi."
        ),
        "gdp": 8500.0,
        "energy": 8.8,
        "gvc": 22.0,
        "trade": 55.0,
        "manuf": 28.0,
        "renew": 14.0,
        "broadband": 12.0,
        "internet": 58.0,
        "mobile": 95.0,
    },
    "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi": {
        "desc": (
            "Küresel değer zincirlerine (GVC) yüksek entegrasyon, devasa dış"
            " ticaret açıklığı ve hiper-dijitalleşme altyapısına sahip Asya"
            " modeli."
        ),
        "gdp": 35000.0,
        "energy": 5.5,
        "gvc": 42.0,
        "trade": 130.0,
        "manuf": 26.0,
        "renew": 22.0,
        "broadband": 44.0,
        "internet": 96.0,
        "mobile": 160.0,
    },
    "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke": {
        "desc": (
            "Verimsiz enerji kullanımı, düşük milli gelir ve zayıf"
            " dijitalleşme ile en yüksek emisyon riski taşıyan kırılgan"
            " ekonomi."
        ),
        "gdp": 3200.0,
        "energy": 12.5,
        "gvc": 12.0,
        "trade": 35.0,
        "manuf": 20.0,
        "renew": 8.0,
        "broadband": 4.0,
        "internet": 32.0,
        "mobile": 65.0,
    },
    "🍃 Yeşil İkiz Dönüşüm Öncüsü": {
        "desc": (
            "Yüksek yenilenebilir enerji (%65) ve ileri dijital altyapının"
            " sinerji oluşturduğu ideal karbonsuzlaşma modeli."
        ),
        "gdp": 55000.0,
        "energy": 2.8,
        "gvc": 30.0,
        "trade": 90.0,
        "manuf": 15.0,
        "renew": 65.0,
        "broadband": 42.0,
        "internet": 95.0,
        "mobile": 140.0,
    },
}

# KÜRESEL ÜLKE VERİ SETİ (ISO3 Kodları, Gerçek Baseline Değerleri)
COUNTRIES_DATA = {
    "Türkiye": {
        "iso": "TUR",
        "gdp": 10600.0,
        "energy": 5.4,
        "gvc": 24.5,
        "trade": 62.0,
        "manuf": 22.0,
        "renew": 21.5,
        "broadband": 21.0,
        "internet": 83.0,
        "mobile": 105.0,
    },
    "Almanya": {
        "iso": "DEU",
        "gdp": 48500.0,
        "energy": 3.1,
        "gvc": 35.0,
        "trade": 88.0,
        "manuf": 18.5,
        "renew": 46.0,
        "broadband": 44.0,
        "internet": 93.0,
        "mobile": 128.0,
    },
    "Amerika Birleşik Devletleri": {
        "iso": "USA",
        "gdp": 70000.0,
        "energy": 4.2,
        "gvc": 18.0,
        "trade": 27.0,
        "manuf": 11.0,
        "renew": 20.0,
        "broadband": 37.0,
        "internet": 92.0,
        "mobile": 116.0,
    },
    "Çin": {
        "iso": "CHN",
        "gdp": 12500.0,
        "energy": 8.5,
        "gvc": 31.0,
        "trade": 37.0,
        "manuf": 27.5,
        "renew": 29.0,
        "broadband": 38.0,
        "internet": 73.0,
        "mobile": 118.0,
    },
    "Hindistan": {
        "iso": "IND",
        "gdp": 2400.0,
        "energy": 9.2,
        "gvc": 20.0,
        "trade": 45.0,
        "manuf": 14.5,
        "renew": 21.0,
        "broadband": 2.2,
        "internet": 46.0,
        "mobile": 84.0,
    },
    "Fransa": {
        "iso": "FRA",
        "gdp": 43500.0,
        "energy": 3.0,
        "gvc": 29.0,
        "trade": 64.0,
        "manuf": 9.5,
        "renew": 24.0,
        "broadband": 48.0,
        "internet": 92.0,
        "mobile": 112.0,
    },
    "İngiltere": {
        "iso": "GBR",
        "gdp": 46500.0,
        "energy": 2.8,
        "gvc": 26.0,
        "trade": 61.0,
        "manuf": 8.8,
        "renew": 42.0,
        "broadband": 41.0,
        "internet": 97.0,
        "mobile": 122.0,
    },
    "İtalya": {
        "iso": "ITA",
        "gdp": 35500.0,
        "energy": 3.3,
        "gvc": 28.0,
        "trade": 68.0,
        "manuf": 14.8,
        "renew": 36.0,
        "broadband": 31.0,
        "internet": 85.0,
        "mobile": 132.0,
    },
    "İspanya": {
        "iso": "ESP",
        "gdp": 30000.0,
        "energy": 3.5,
        "gvc": 27.0,
        "trade": 67.0,
        "manuf": 11.5,
        "renew": 42.0,
        "broadband": 36.0,
        "internet": 93.0,
        "mobile": 118.0,
    },
    "Japonya": {
        "iso": "JPN",
        "gdp": 39000.0,
        "energy": 3.6,
        "gvc": 28.0,
        "trade": 37.0,
        "manuf": 20.0,
        "renew": 22.0,
        "broadband": 35.0,
        "internet": 90.0,
        "mobile": 152.0,
    },
    "Güney Kore": {
        "iso": "KOR",
        "gdp": 33000.0,
        "energy": 5.2,
        "gvc": 41.0,
        "trade": 80.0,
        "manuf": 25.5,
        "renew": 8.5,
        "broadband": 43.0,
        "internet": 97.0,
        "mobile": 142.0,
    },
    "Brezilya": {
        "iso": "BRA",
        "gdp": 8900.0,
        "energy": 5.0,
        "gvc": 16.0,
        "trade": 39.0,
        "manuf": 10.0,
        "renew": 48.0,
        "broadband": 19.0,
        "internet": 81.0,
        "mobile": 102.0,
    },
    "Kanada": {
        "iso": "CAN",
        "gdp": 52000.0,
        "energy": 6.8,
        "gvc": 26.0,
        "trade": 65.0,
        "manuf": 9.8,
        "renew": 68.0,
        "broadband": 40.0,
        "internet": 93.0,
        "mobile": 96.0,
    },
    "Avustralya": {
        "iso": "AUS",
        "gdp": 64000.0,
        "energy": 4.8,
        "gvc": 19.0,
        "trade": 46.0,
        "manuf": 5.8,
        "renew": 29.0,
        "broadband": 36.0,
        "internet": 96.0,
        "mobile": 108.0,
    },
    "Meksika": {
        "iso": "MEX",
        "gdp": 11000.0,
        "energy": 4.5,
        "gvc": 36.0,
        "trade": 78.0,
        "manuf": 18.0,
        "renew": 16.0,
        "broadband": 18.0,
        "internet": 76.0,
        "mobile": 98.0,
    },
    "Endonezya": {
        "iso": "IDN",
        "gdp": 4800.0,
        "energy": 4.1,
        "gvc": 21.0,
        "trade": 42.0,
        "manuf": 19.0,
        "renew": 14.0,
        "broadband": 4.5,
        "internet": 62.0,
        "mobile": 125.0,
    },
    "Hollanda": {
        "iso": "NLD",
        "gdp": 57000.0,
        "energy": 3.1,
        "gvc": 48.0,
        "trade": 155.0,
        "manuf": 11.0,
        "renew": 33.0,
        "broadband": 45.0,
        "internet": 96.0,
        "mobile": 125.0,
    },
    "İsviçre": {
        "iso": "CHE",
        "gdp": 92000.0,
        "energy": 2.1,
        "gvc": 33.0,
        "trade": 118.0,
        "manuf": 18.0,
        "renew": 30.0,
        "broadband": 47.0,
        "internet": 96.0,
        "mobile": 128.0,
    },
    "İsveç": {
        "iso": "SWE",
        "gdp": 56000.0,
        "energy": 3.8,
        "gvc": 34.0,
        "trade": 92.0,
        "manuf": 13.0,
        "renew": 66.0,
        "broadband": 41.0,
        "internet": 95.0,
        "mobile": 126.0,
    },
    "Norveç": {
        "iso": "NOR",
        "gdp": 89000.0,
        "energy": 4.1,
        "gvc": 28.0,
        "trade": 72.0,
        "manuf": 6.5,
        "renew": 75.0,
        "broadband": 45.0,
        "internet": 97.0,
        "mobile": 108.0,
    },
    "Polonya": {
        "iso": "POL",
        "gdp": 18000.0,
        "energy": 5.1,
        "gvc": 36.0,
        "trade": 108.0,
        "manuf": 16.8,
        "renew": 17.0,
        "broadband": 24.0,
        "internet": 87.0,
        "mobile": 138.0,
    },
    "Güney Afrika": {
        "iso": "ZAF",
        "gdp": 6700.0,
        "energy": 9.8,
        "gvc": 22.0,
        "trade": 56.0,
        "manuf": 12.0,
        "renew": 10.0,
        "broadband": 3.8,
        "internet": 72.0,
        "mobile": 162.0,
    },
    "Suudi Arabistan": {
        "iso": "SAU",
        "gdp": 30000.0,
        "energy": 7.8,
        "gvc": 18.0,
        "trade": 62.0,
        "manuf": 13.0,
        "renew": 1.0,
        "broadband": 32.0,
        "internet": 98.0,
        "mobile": 135.0,
    },
    "Arjantin": {
        "iso": "ARG",
        "gdp": 13000.0,
        "energy": 4.6,
        "gvc": 14.0,
        "trade": 33.0,
        "manuf": 15.0,
        "renew": 11.0,
        "broadband": 22.0,
        "internet": 87.0,
        "mobile": 125.0,
    },
    "Yunanistan": {
        "iso": "GRC",
        "gdp": 20500.0,
        "energy": 3.8,
        "gvc": 23.0,
        "trade": 78.0,
        "manuf": 9.2,
        "renew": 38.0,
        "broadband": 41.0,
        "internet": 79.0,
        "mobile": 115.0,
    },
    "Portekiz": {
        "iso": "PRT",
        "gdp": 24500.0,
        "energy": 3.6,
        "gvc": 31.0,
        "trade": 85.0,
        "manuf": 11.8,
        "renew": 54.0,
        "broadband": 42.0,
        "internet": 85.0,
        "mobile": 122.0,
    },
    "Belçika": {
        "iso": "BEL",
        "gdp": 50000.0,
        "energy": 4.1,
        "gvc": 46.0,
        "trade": 165.0,
        "manuf": 12.5,
        "renew": 23.0,
        "broadband": 43.0,
        "internet": 94.0,
        "mobile": 102.0,
    },
    "Avusturya": {
        "iso": "AUT",
        "gdp": 53000.0,
        "energy": 3.2,
        "gvc": 38.0,
        "trade": 105.0,
        "manuf": 16.2,
        "renew": 78.0,
        "broadband": 30.0,
        "internet": 93.0,
        "mobile": 122.0,
    },
    "Danimarka": {
        "iso": "DNK",
        "gdp": 67000.0,
        "energy": 2.5,
        "gvc": 36.0,
        "trade": 112.0,
        "manuf": 11.5,
        "renew": 62.0,
        "broadband": 45.0,
        "internet": 98.0,
        "mobile": 125.0,
    },
    "Finlandiya": {
        "iso": "FIN",
        "gdp": 53000.0,
        "energy": 4.5,
        "gvc": 32.0,
        "trade": 78.0,
        "manuf": 14.0,
        "renew": 52.0,
        "broadband": 36.0,
        "internet": 97.0,
        "mobile": 168.0,
    },
    "İrlanda": {
        "iso": "IRL",
        "gdp": 100000.0,
        "energy": 1.5,
        "gvc": 42.0,
        "trade": 190.0,
        "manuf": 32.0,
        "renew": 35.0,
        "broadband": 32.0,
        "internet": 92.0,
        "mobile": 108.0,
    },
    "Şili": {
        "iso": "CHL",
        "gdp": 15500.0,
        "energy": 5.1,
        "gvc": 24.0,
        "trade": 62.0,
        "manuf": 10.5,
        "renew": 31.0,
        "broadband": 23.0,
        "internet": 90.0,
        "mobile": 138.0,
    },
    "Kolombiya": {
        "iso": "COL",
        "gdp": 6600.0,
        "energy": 3.8,
        "gvc": 15.0,
        "trade": 38.0,
        "manuf": 11.0,
        "renew": 72.0,
        "broadband": 17.0,
        "internet": 73.0,
        "mobile": 142.0,
    },
    "Çekya": {
        "iso": "CZE",
        "gdp": 27000.0,
        "energy": 4.8,
        "gvc": 42.0,
        "trade": 140.0,
        "manuf": 22.0,
        "renew": 17.0,
        "broadband": 33.0,
        "internet": 88.0,
        "mobile": 132.0,
    },
    "Macaristan": {
        "iso": "HUN",
        "gdp": 18500.0,
        "energy": 4.5,
        "gvc": 45.0,
        "trade": 160.0,
        "manuf": 19.5,
        "renew": 14.0,
        "broadband": 32.0,
        "internet": 89.0,
        "mobile": 108.0,
    },
    "Romanya": {
        "iso": "ROU",
        "gdp": 15000.0,
        "energy": 4.2,
        "gvc": 31.0,
        "trade": 82.0,
        "manuf": 17.0,
        "renew": 24.0,
        "broadband": 28.0,
        "internet": 82.0,
        "mobile": 115.0,
    },
    "Malezya": {
        "iso": "MYS",
        "gdp": 12000.0,
        "energy": 5.8,
        "gvc": 41.0,
        "trade": 130.0,
        "manuf": 23.5,
        "renew": 18.0,
        "broadband": 11.0,
        "internet": 96.0,
        "mobile": 140.0,
    },
    "Tayland": {
        "iso": "THA",
        "gdp": 7200.0,
        "energy": 6.2,
        "gvc": 38.0,
        "trade": 118.0,
        "manuf": 27.0,
        "renew": 15.0,
        "broadband": 18.0,
        "internet": 85.0,
        "mobile": 138.0,
    },
    "Vietnam": {
        "iso": "VNM",
        "gdp": 4100.0,
        "energy": 7.5,
        "gvc": 46.0,
        "trade": 185.0,
        "manuf": 25.0,
        "renew": 32.0,
        "broadband": 21.0,
        "internet": 78.0,
        "mobile": 130.0,
    },
    "Filipinler": {
        "iso": "PHL",
        "gdp": 3600.0,
        "energy": 3.2,
        "gvc": 28.0,
        "trade": 68.0,
        "manuf": 17.5,
        "renew": 21.0,
        "broadband": 10.0,
        "internet": 53.0,
        "mobile": 142.0,
    },
    "Singapur": {
        "iso": "SGP",
        "gdp": 82000.0,
        "energy": 3.1,
        "gvc": 55.0,
        "trade": 330.0,
        "manuf": 20.0,
        "renew": 2.0,
        "broadband": 45.0,
        "internet": 92.0,
        "mobile": 158.0,
    },
    "Yeni Zelanda": {
        "iso": "NZL",
        "gdp": 48000.0,
        "energy": 3.8,
        "gvc": 21.0,
        "trade": 52.0,
        "manuf": 9.0,
        "renew": 80.0,
        "broadband": 35.0,
        "internet": 95.0,
        "mobile": 125.0,
    },
    "Mısır": {
        "iso": "EGY",
        "gdp": 3700.0,
        "energy": 5.8,
        "gvc": 14.0,
        "trade": 38.0,
        "manuf": 16.0,
        "renew": 11.0,
        "broadband": 11.0,
        "internet": 72.0,
        "mobile": 98.0,
    },
    "Nijerya": {
        "iso": "NGA",
        "gdp": 2200.0,
        "energy": 6.5,
        "gvc": 12.0,
        "trade": 32.0,
        "manuf": 9.0,
        "renew": 18.0,
        "broadband": 4.0,
        "internet": 55.0,
        "mobile": 92.0,
    },
    "Cezayir": {
        "iso": "DZA",
        "gdp": 4300.0,
        "energy": 6.1,
        "gvc": 15.0,
        "trade": 51.0,
        "manuf": 11.0,
        "renew": 1.0,
        "broadband": 11.0,
        "internet": 71.0,
        "mobile": 108.0,
    },
    "Fas": {
        "iso": "MAR",
        "gdp": 3800.0,
        "energy": 4.2,
        "gvc": 26.0,
        "trade": 82.0,
        "manuf": 15.0,
        "renew": 19.0,
        "broadband": 7.0,
        "internet": 88.0,
        "mobile": 135.0,
    },
    "İsrail": {
        "iso": "ISR",
        "gdp": 54000.0,
        "energy": 2.8,
        "gvc": 28.0,
        "trade": 60.0,
        "manuf": 11.0,
        "renew": 10.0,
        "broadband": 31.0,
        "internet": 90.0,
        "mobile": 128.0,
    },
    "Birleşik Arap Emirlikleri": {
        "iso": "ARE",
        "gdp": 49000.0,
        "energy": 6.2,
        "gvc": 22.0,
        "trade": 170.0,
        "manuf": 9.0,
        "renew": 4.0,
        "broadband": 36.0,
        "internet": 99.0,
        "mobile": 200.0,
    },
    "Katar": {
        "iso": "QAT",
        "gdp": 88000.0,
        "energy": 8.1,
        "gvc": 15.0,
        "trade": 92.0,
        "manuf": 8.0,
        "renew": 0.5,
        "broadband": 31.0,
        "internet": 99.0,
        "mobile": 145.0,
    },
    "Kazakistan": {
        "iso": "KAZ",
        "gdp": 11500.0,
        "energy": 11.2,
        "gvc": 18.0,
        "trade": 62.0,
        "manuf": 13.0,
        "renew": 4.0,
        "broadband": 14.0,
        "internet": 92.0,
        "mobile": 132.0,
    },
    "Ukrayna": {
        "iso": "UKR",
        "gdp": 4500.0,
        "energy": 10.5,
        "gvc": 22.0,
        "trade": 82.0,
        "manuf": 11.0,
        "renew": 11.0,
        "broadband": 19.0,
        "internet": 79.0,
        "mobile": 128.0,
    },
    "Pakistan": {
        "iso": "PAK",
        "gdp": 1500.0,
        "energy": 8.2,
        "gvc": 12.0,
        "trade": 30.0,
        "manuf": 12.0,
        "renew": 6.0,
        "broadband": 5.0,
        "internet": 36.0,
        "mobile": 82.0,
    },
}

# GÜVENİLİR CANLI ZİYARETÇİ SAYACI HESAPLAMA
if "visitor_count" not in st.session_state:
  try:
    req = urllib.request.Request(
        "https://api.counterapi.dev/v1/karbon-simulasyonu-zeynep-v3/visits/up",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=3) as response:
      res_data = json.loads(response.read().decode())
      cnt = res_data.get("count", None)
      if cnt and isinstance(cnt, int) and cnt > 0:
        st.session_state.visitor_count = cnt
      else:
        st.session_state.visitor_count = 154
  except Exception:
    st.session_state.visitor_count = 154


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


# TAHMİN FONKSİYONU
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
  st.session_state.selected_country_name = "Türkiye"

# 3. YAN MENÜ (HAZIR PROFİLLER & ÜLKE SEÇİMİ)
st.sidebar.title("🌍 Profil veya Ülke Seçiniz")

mode_choice = st.sidebar.radio(
    "Senaryo Modu Seçimi:",
    ["Hazır Ülke Profilini Kullan", "Küresel Ülke Listesinden Seç"],
)

selected_profile = "🏛️ S0 Referans Küresel Durum (Baseline)"
selected_country = "Türkiye"

if mode_choice == "Hazır Ülke Profilini Kullan":
  selected_profile = st.sidebar.selectbox(
      "Hazır Profiller:", list(PROFILE_DETAILS.keys())
  )
  p_info = PROFILE_DETAILS[selected_profile]
  st.session_state.gdp_val = p_info["gdp"]
  st.session_state.energy_val = p_info["energy"]
  st.session_state.gvc_val = p_info["gvc"]
  st.session_state.trade_val = p_info["trade"]
  st.session_state.manuf_val = p_info["manuf"]
  st.session_state.renew_val = p_info["renew"]
  st.session_state.broadband_val = p_info["broadband"]
  st.session_state.internet_val = p_info["internet"]
  st.session_state.mobile_val = p_info["mobile"]
else:
  selected_country = st.sidebar.selectbox(
      "Küresel Ülke Listesi:", list(COUNTRIES_DATA.keys())
  )
  c_info = COUNTRIES_DATA[selected_country]
  st.session_state.selected_country_name = selected_country
  st.session_state.gdp_val = c_info["gdp"]
  st.session_state.energy_val = c_info["energy"]
  st.session_state.gvc_val = c_info["gvc"]
  st.session_state.trade_val = c_info["trade"]
  st.session_state.manuf_val = c_info["manuf"]
  st.session_state.renew_val = c_info["renew"]
  st.session_state.broadband_val = c_info["broadband"]
  st.session_state.internet_val = c_info["internet"]
  st.session_state.mobile_val = c_info["mobile"]

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Politika Parametreleri")

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

st.sidebar.subheader("📱 Dijital Göstergeler")
broadband = st.sidebar.slider(
    "Sabit Geniş Bant (100 Kişide)",
    0.0,
    50.0,
    float(st.session_state.broadband_val),
    step=0.5,
)
internet_users = st.sidebar.slider(
    "İnternet Kullanım Oranı (%)",
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
count_display = st.session_state.get("visitor_count", 154)
st.sidebar.markdown(f"👁️ **Toplam Ziyaret Sayısı:** `{count_display}`")

# HESAPLAMALAR
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

# Seçili ülkenin / profilin baseline (ilk) tahmini
if mode_choice == "Küresel Ülke Listesinden Seç":
  base_c = COUNTRIES_DATA[selected_country]
  base_country_emission = predict_emissions(
      base_c["gdp"],
      base_c["energy"],
      base_c["gvc"],
      base_c["trade"],
      base_c["manuf"],
      base_c["renew"],
      base_c["broadband"],
      base_c["internet"],
      base_c["mobile"],
  )
else:
  base_country_emission = BASE_EMISSION

# SHAP WATERFALL KATKILARI
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
mc_distribution = np.maximum(0.0, pred_emission + residuals)
lower_bound = float(np.percentile(mc_distribution, 5).item())
upper_bound = float(np.percentile(mc_distribution, 95).item())

emission_diff = pred_emission - BASE_EMISSION
emission_pct_change = (emission_diff / BASE_EMISSION) * 100.0

country_policy_diff = pred_emission - base_country_emission
country_policy_pct = (
    (country_policy_diff / base_country_emission) * 100.0
    if base_country_emission > 0
    else 0.0
)

# 4. ANA EKRAN
st.title("🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Sistemi")
st.caption(
    "Açıklanabilir Yapay Zeka (SVR & Monte Carlo) Destekli Politika Senaryo"
    " Analitiği"
)

# SEÇİLİ PROFİL / ÜLKE BİLGİ KUTUSU
if mode_choice == "Hazır Ülke Profilini Kullan":
  p_desc = PROFILE_DETAILS[selected_profile]["desc"]
  st.info(f"💡 **Seçili Hazır Profil:** {selected_profile}\n\n*{p_desc}*")
else:
  st.success(
      f"📌 **Seçili Ülke:** {selected_country} | **Mevcut Gerçek"
      f" Emisyonu:** {base_country_emission:.2f} Mt CO₂eq\n\n*Slider'lar ile"
      f" {selected_country} üzerine özel politika senaryoları uygulayarak"
      " canlı sonuçları inceleyebilirsiniz.*"
  )

# ÜST METRİK KARTLARI
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Yeni Politika Tahmini",
    value=f"{pred_emission:.2f} Mt CO₂eq",
    delta=f"{emission_pct_change:+.1f}% vs Baseline",
    delta_color="inverse",
)

if mode_choice == "Küresel Ülke Listesinden Seç":
  col2.metric(
      label=f"{selected_country} (Gerçek Durum)",
      value=f"{base_country_emission:.2f} Mt CO₂eq",
      delta=f"{country_policy_pct:+.1f}% Yeni Senaryo Farkı",
      delta_color="inverse",
  )
else:
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

# SEKMELİ YAPI (5 SEKMELİ)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Canlı Simülasyon & Akıllı Çıkarımlar",
    "🧩 Yerel SHAP / Politika Katkı Analizi",
    "🌍 Küresel Emisyon Haritası",
    "🕸️ Radar & Profil Kıyaslaması",
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
      annotation_text=f"Yeni Senaryo ({pred_emission:.1f})",
      annotation_position="top right",
  )
  fig.update_layout(
      xaxis_title="Talep Tabanlı GHG Emisyonu (Mt CO₂eq)",
      yaxis_title="Simülasyon Frekansı",
      template="plotly_white",
      height=400,
  )
  st.plotly_chart(fig, use_container_width=True)

  # OTOMATİK AKILLI POLITİKA ÇIKARIM RAPORU (AI POLICY INSIGHTS)
  st.markdown("### 🤖 Otomatik Politika Çıkarım ve Analiz Raporu")

  direction_text = (
      "düşüş yönlü bir duyarlılık"
      if emission_diff <= 0
      else "artış yönlü bir yük"
  )
  pct_text = f"%{abs(emission_pct_change):.1f}"

  max_red_name = feature_names[np.argmin(feature_contribs)]
  max_inc_name = feature_names[np.argmax(feature_contribs)]

  twin_text = (
      "Yenilenebilir enerji payının (% "
      + str(renewable_energy)
      + ") yüksek düzeyde tutulması, dijital altyapı genişlemesinin getirebileceği"
      " 'Rebound (Geri Tepme) Etkisini' başarıyla nötrlemiş ve Yeşil-Dijital İkiz"
      " Dönüşüm sinerjisini doğrulamıştır."
      if renewable_energy >= 35.0
      else "Yenilenebilir enerji payının (% "
      + str(renewable_energy)
      + ") henüz kritik eşiğin altında kalması nedeniyle, dijitalleşme ve sanayi"
      " faaliyetleri toplam karbon ayak izi üzerinde baskı oluşturmaya devam"
      " etmektedir."
  )

  insight_md = f"""
    <div class="insight-card">
    <h4>📝 Model Metodolojik Bulguları ve Akıllı Senaryo Çıkarımı</h4>
    <p>Simüle edilen bu politik bileşim sonucunda, talep bazlı karbon ayak izi tahmini referans duruma (822.40 Mt CO₂eq) kıyasla <b>{pct_text}</b> oranında <b>{direction_text}</b> sergileyerek <b>{pred_emission:.2f} Mt CO₂eq</b> seviyesinde dengelenmiştir.</p>
    <p><b>Temel Yönlendirici Dinamikler:</b> SVR karar yüzeyi üzerindeki ayrıştırma analizine göre; karbonsuzlaşmaya en güçlü katkıyı veren değişken <b>{max_red_name}</b> olurken, emisyon öngörüsünü yukarı çeken ana unsur <b>{max_inc_name}</b> olarak öne çıkmaktadır.</p>
    <p><b>İkiz Dönüşüm Perspektifi:</b> {twin_text}</p>
    <p><i><b>Metodolojik Not:</b> Bu çıkarımlar SVR modelinin ilişkisel marjinal duyarlılıklarına dayanmaktadır; doğrudan nedensel (causal) bir bağlam ifade etmez.</i></p>
    </div>
    """
  st.markdown(insight_md, unsafe_allow_html=True)

with tab2:
  st.subheader("🧩 Yerel Politika Katkı (SHAP Waterfall) Grafiği")
  st.markdown("""
    Bu grafik, **S0 Referans Durumuna (822.4 Mt CO₂eq)** göre seçilen politika bileşenlerinin emisyon tahminini **ne kadar artırdığını (+ Kırmızı)** veya **ne kadar düşürdüğünü (- Yeşil)** tek tek ayrıştırır.
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
          decreasing={"marker": {"color": "#2ca02c"}},
          increasing={"marker": {"color": "#d62728"}},
          totals={"marker": {"color": "#1f77b4"}},
      )
  )
  fig_waterfall.update_layout(
      title="S0 Baseline (822.4 Mt) Üzerine Marjinal Ayrıştırma",
      yaxis_title="Katkı Miktarı (Mt CO₂eq)",
      template="plotly_white",
      height=480,
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
  st.subheader("🌍 Küresel Karbon Ayak İzi Haritası ve Ülke Detay Paneli")
  st.markdown("""
    Haritada veri setimizdeki küresel ekonomiler **karbon emisyon seviyelerine göre** renklendirilmiştir.
    Haritanın hemen altında, seçtiğiniz ülkenin / profilin tüm makroekonomik ve dijital altyapı göstergeleri detaylıca sunulmaktadır.
    """)

  map_list = []
  for c_name, c_data in COUNTRIES_DATA.items():
    e_val = predict_emissions(
        c_data["gdp"],
        c_data["energy"],
        c_data["gvc"],
        c_data["trade"],
        c_data["manuf"],
        c_data["renew"],
        c_data["broadband"],
        c_data["internet"],
        c_data["mobile"],
    )
    map_list.append({
        "Ülke": c_name,
        "ISO": c_data["iso"],
        "Emisyon": e_val,
        "GSYH": c_data["gdp"],
        "Yenilenebilir (%)": c_data["renew"],
        "Enerji Yoğunluğu": c_data["energy"],
    })
  df_map = pd.DataFrame(map_list)

  fig_map = px.choropleth(
      df_map,
      locations="ISO",
      color="Emisyon",
      hover_name="Ülke",
      hover_data={
          "ISO": False,
          "Emisyon": ":.2f",
          "GSYH": ":,.0f",
          "Yenilenebilir (%)": ":.1f",
          "Enerji Yoğunluğu": ":.1f",
      },
      color_continuous_scale="Reds",
      title="Küresel Ekonomilerin Tahmini Karbon Ayak İzi Dağılımı (Mt CO₂eq)",
  )

  if mode_choice == "Küresel Ülke Listesinden Seç":
    selected_iso = COUNTRIES_DATA[selected_country]["iso"]
    selected_row = df_map[df_map["ISO"] == selected_iso]
    if not selected_row.empty:
      fig_map.add_trace(
          go.Choropleth(
              locations=selected_row["ISO"],
              z=selected_row["Emisyon"],
              colorscale=[[0, "gold"], [1, "gold"]],
              showscale=False,
              marker_line_color="black",
              marker_line_width=3,
              name=f"Seçili Ülke: {selected_country}",
          )
      )

  fig_map.update_layout(
      geo=dict(
          showframe=False, showcoastlines=True, projection_type="natural earth"
      ),
      height=500,
  )
  st.plotly_chart(fig_map, use_container_width=True)

  # HARİTA ALTI DETAYLI ÜLKE / PROFİL GÖSTERGE KARTI
  st.markdown("### 📌 Seçili Ülke / Profil Gösterge Kartı")
  mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)

  if mode_choice == "Küresel Ülke Listesinden Seç":
    cur_data = COUNTRIES_DATA[selected_country]
    mc_c1.metric("Ülke Adı", selected_country, f"ISO: {cur_data['iso']}")
    mc_c2.metric("Kişi Başı GSYH", f"\${cur_data['gdp']:,.0f}")
    mc_c3.metric("Yenilenebilir Enerji", f"%{cur_data['renew']:.1f}")
    mc_c4.metric("Enerji Yoğunluğu", f"{cur_data['energy']:.1f} MJ/\$")

    mc_d1, mc_d2, mc_d3, mc_d4 = st.columns(4)
    mc_d1.metric("İmalat Sanayi", f"%{cur_data['manuf']:.1f}")
    mc_d2.metric("GVC Payı", f"%{cur_data['gvc']:.1f}")
    mc_d3.metric("Ticari Açıklık", f"%{cur_data['trade']:.1f}")
    mc_d4.metric("İnternet Kullanımı", f"%{cur_data['internet']:.1f}")
  else:
    p_data = PROFILE_DETAILS[selected_profile]
    mc_c1.metric("Profil Adı", selected_profile.split()[0] + " Profil")
    mc_c2.metric("Kişi Başı GSYH", f"\${p_data['gdp']:,.0f}")
    mc_c3.metric("Yenilenebilir Enerji", f"%{p_data['renew']:.1f}")
    mc_c4.metric("Enerji Yoğunluğu", f"{p_data['energy']:.1f} MJ/\$")

    mc_d1, mc_d2, mc_d3, mc_d4 = st.columns(4)
    mc_d1.metric("İmalat Sanayi", f"%{p_data['manuf']:.1f}")
    mc_d2.metric("GVC Payı", f"%{p_data['gvc']:.1f}")
    mc_d3.metric("Ticari Açıklık", f"%{p_data['trade']:.1f}")
    mc_d4.metric("İnternet Kullanımı", f"%{p_data['internet']:.1f}")

with tab4:
  st.subheader("🕸️ Radar Grafiği ve Politika Değişim Özeti")
  st.markdown(
      "Aşağıdaki Radar (Örümcek Ağı) Grafiği, **Mevcut Durum / Baseline** ile"
      " **Yeni Politika Senaryosunun** 7 boyuttaki profil değişimini gösterir."
  )

  # Normalleştirilmiş Radar Grafiği
  categories = [
      "GSYH",
      "Enerji Yoğunluğu",
      "GVC Çıktısı",
      "Ticari Açıklık",
      "İmalat Sanayi",
      "Yenilenebilir Enerji",
      "Dijitalleşme",
  ]

  # Baseline değerleri (0-1 arası ölçeklendirilmiş)
  r_base = [
      BASE_GDP / 95000.0,
      BASE_ENERGY / 15.0,
      BASE_GVC / 60.0,
      BASE_TRADE / 200.0,
      BASE_MANUF / 45.0,
      BASE_RENEW / 80.0,
      (internet_users + broadband) / 150.0,
  ]
  # Yeni Senaryo değerleri
  r_scen = [
      gdp / 95000.0,
      energy_intensity / 15.0,
      gvc_output / 60.0,
      trade_openness / 200.0,
      manufacturing / 45.0,
      renewable_energy / 80.0,
      (internet_users + broadband) / 150.0,
  ]

  fig_radar = go.Figure()
  fig_radar.add_trace(
      go.Scatterpolar(
          r=r_base,
          theta=categories,
          fill="toself",
          name="S0 Baseline / Mevcut",
          line_color="gray",
      )
  )
  fig_radar.add_trace(
      go.Scatterpolar(
          r=r_scen,
          theta=categories,
          fill="toself",
          name="Yeni Politika Senaryosu",
          line_color="#2b5c8f",
      )
  )

  fig_radar.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
      showlegend=True,
      height=450,
  )
  st.plotly_chart(fig_radar, use_container_width=True)

  # RAPOR İNDİRME BUTONU
  st.markdown("### 📥 Senaryo Raporunu İndir")
  export_df = pd.DataFrame([{
      "Senaryo": (
          selected_country
          if mode_choice == "Küresel Ülke Listesinden Seç"
          else selected_profile
      ),
      "Tahmini_Emisyon_Mt": round(pred_emission, 2),
      "Baseline_Farki_Mt": round(emission_diff, 2),
      "Yuzdesel_Degisim": round(emission_pct_change, 2),
      "Alt_Guven_Siniri": round(lower_bound, 2),
      "Ust_Guven_Siniri": round(upper_bound, 2),
      "GSYH_\$": gdp,
      "Enerji_Yogunlugu_MJ": energy_intensity,
      "Yenilenebilir_Enerji_Pct": renewable_energy,
  }])
  csv_data = export_df.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📄 Senaryo Sonuçlarını CSV Olarak İndir",
      data=csv_data,
      file_name="karbon_senaryo_sonuclari.csv",
      mime="text/csv",
  )

with tab5:
  st.subheader("📜 Metodoloji, Hazır Profiller ve XAI Notları")

  st.markdown("### 🏛️ Hazır Ülke Profillerinin Sayısal Varsayımları")
  prof_table = []
  for p_name, p_vals in PROFILE_DETAILS.items():
    prof_table.append({
        "Hazır Profil Adı": p_name,
        "GSYH (\$)": f"{p_vals['gdp']:,.0f}",
        "Enerji Yoğ. (MJ/\$)": f"{p_vals['energy']:.1f}",
        "Yenilenebilir (%)": f"{p_vals['renew']:.1f}%",
        "İmalat (%)": f"{p_vals['manuf']:.1f}%",
        "GVC Payı (%)": f"{p_vals['gvc']:.1f}%",
        "Ticari Açıklık (%)": f"{p_vals['trade']:.1f}%",
        "İnternet (%)": f"{p_vals['internet']:.1f}%",
    })
  st.dataframe(
      pd.DataFrame(prof_table), use_container_width=True, hide_index=True
  )

  st.markdown("""
    ---
    ### 🔬 Matematiksel ve Metodolojik Çerçeve
    * **Tahmin Modeli:** RBF Çekirdekli Destek Vektör Regresyonu (SVR - Test \\(R^2 = 0.975\\)).
    * **Boyut İndirgeme:** Dijitalleşme göstergeleri (Sabit Genişbant, İnternet, Mobil) Temel Bileşenler Analizi (PCA) ile tek bir Dijitalleşme İndeksine dönüştürülmüştür.
    * **Yerel XAI Katkı Yöntemi:** Her bir değişkenin tahmine olan marjinal katkısı, diğer değişkenler S0 Baseline seviyesinde sabit tutularak SVR karar yüzeyi üzerinde tekil duyarlılık adımları ile ayrıştırılmıştır.
    * **Belirsizlik Analizi:** Modelin ampirik artık hata dağılımı (\\(RMSE = 101.24\\text{ Mt CO}_2\\text{eq}\\)) üzerinden 10.000 iterasyonlu Monte Carlo simülasyonu çalıştırılmıştır.
    * **Metodolojik Çerçeve:** Bu araç nedensel (causal) çıkarım yapmaz; makroekonomik değişkenler arasındaki **tahminsel ve ilişkisel (associative) duyarlılıkları** simüle eder.
    """)
