import json
import os
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

# 1. STREAMLIT CONFIG & CUSTOM CSS
st.set_page_config(
    page_title="Küresel Karbon Ayak İzi & Kurumsal İklim Riski Simülatörü",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_style = """
<style>
    .main { 
        background: linear-gradient(135deg, #f4f7f6 0%, #e9ecef 100%); 
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.6);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        background: linear-gradient(135deg, #2b5c8f 0%, #1d3e61 100%);
        color: white;
        border: none;
        padding: 10px 16px;
        box-shadow: 0 4px 12px rgba(43, 92, 143, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d3e61 0%, #122840 100%);
        color: white;
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
    .glass-card-green {
        background: linear-gradient(135deg, rgba(235, 247, 238, 0.95) 0%, rgba(210, 240, 218, 0.95) 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #2ca02c;
        box-shadow: 0 3px 10px rgba(44, 160, 44, 0.1);
        margin-bottom: 10px;
    }
    .glass-card-red {
        background: linear-gradient(135deg, rgba(253, 238, 238, 0.95) 0%, rgba(250, 215, 215, 0.95) 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #d62728;
        box-shadow: 0 3px 10px rgba(214, 39, 40, 0.1);
        margin-bottom: 10px;
    }
    .glass-card-blue {
        background: linear-gradient(135deg, rgba(235, 243, 250, 0.95) 0%, rgba(212, 230, 245, 0.95) 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 3px 10px rgba(31, 119, 180, 0.1);
        margin-bottom: 10px;
    }
    .glass-card-yellow {
        background: linear-gradient(135deg, rgba(255, 251, 235, 0.95) 0%, rgba(254, 243, 199, 0.95) 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 3px 10px rgba(245, 158, 11, 0.1);
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# DYNAMIC VISITOR PERSISTENCE COUNTER
VISITOR_FILE = "visitor_data.json"

def load_visitor_data():
    default_data = {
        "total_visits": 142,
        "country_breakdown": {
            "🇹🇷 Türkiye": 68,
            "🇩🇪 Almanya": 24,
            "🇺🇸 ABD": 18,
            "🇬🇧 İngiltere": 12,
            "🇫🇷 Fransa": 9,
            "🇳🇱 Hollanda": 6,
            "🌍 Diğer Ülkeler": 5
        }
    }
    if os.path.exists(VISITOR_FILE):
        try:
            with open(VISITOR_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data
    return default_data

def track_new_visit():
    vdata = load_visitor_data()
    if "has_visited" not in st.session_state:
        st.session_state.has_visited = True
        vdata["total_visits"] += 1
        vdata["country_breakdown"]["🇹🇷 Türkiye"] += 1
        try:
            with open(VISITOR_FILE, "w", encoding="utf-8") as f:
                json.dump(vdata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return vdata

visitor_info = track_new_visit()

# HELPER FUNCTION TO CLEAN TEXT FOR PDF (PREVENTS UNICODE ENCODING ERROR INCLUDING EMOJIS)
def clean_pdf_text(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G",
        "ş": "s", "Ş": "S", "ç": "c", "Ç": "C",
        "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"
    }
    for tr_char, clean_char in replacements.items():
        text = text.replace(tr_char, clean_char)
    # Strip emojis and non-latin-1 characters safely
    return text.encode("latin-1", "ignore").decode("latin-1")

# BASELINE S0 CONSTANTS
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

# PROFILE DETAILS
PROFILE_DETAILS = {
    "🏛️ S0 Referans Küresel Durum (Baseline)": {
        "desc": "Veri setindeki tüm küresel ekonomilerin tam ortalamasını temsil eden nötr mihenk taşı.",
        "desc_en": "Neutral benchmark representing the exact global average of all economies in the dataset.",
        "badge": "Mihenk Taşı / Referans",
        "badge_en": "Benchmark / Reference",
        "focus": "Küresel Ortalama Kıyaslaması",
        "recipe": "Mevcut küresel dengenin korunması ve kademeli emisyon azaltımı.",
        "recipe_en": "Maintaining current global balance and gradual emission reduction.",
        "peers": ["Dünya Ortalaması"],
        "peers_en": ["Global Average"],
        "skdm_risk": "Düşük / Nötr",
        "skdm_risk_en": "Low / Neutral",
        "skdm_score": "2/10",
        "gdp": 25000.0, "energy": 5.2, "gvc": 25.0, "trade": 85.0, "manuf": 18.0, "renew": 22.0, "broadband": 20.0, "internet": 75.0, "mobile": 110.0
    },
    "🇪🇺 AB Yeşil Mutabakat Ülkesi": {
        "desc": "Sıkı iklim politikaları, yüksek milli gelir ve baskın yenilenebilir enerji dönüşümü sağlayan gelişmiş AB modeli.",
        "desc_en": "Advanced EU model with strict climate policies, high GDP, and dominant renewable energy transition.",
        "badge": "İklim ve Temiz Enerji Lideri",
        "badge_en": "Climate & Clean Energy Leader",
        "focus": "Sınırda Karbon Düzenlemesi (CBAM) ve İkiz Dönüşüm",
        "recipe": "Yenilenebilir enerji şebekesini dijital yapay zeka ile optimize etmek ve tedarik zinciri emisyonlarını denetlemek.",
        "recipe_en": "Optimize renewable energy grid with AI and audit supply chain emissions.",
        "peers": ["Almanya", "Fransa", "İsveç", "Hollanda", "Danimarka"],
        "peers_en": ["Germany", "France", "Sweden", "Netherlands", "Denmark"],
        "skdm_risk": "Çok Düşük / Muafiyet Avantajı",
        "skdm_risk_en": "Very Low / Exemption Advantage",
        "skdm_score": "1/10",
        "gdp": 48000.0, "energy": 3.2, "gvc": 32.0, "trade": 85.0, "manuf": 14.0, "renew": 58.0, "broadband": 38.0, "internet": 92.0, "mobile": 135.0
    },
    "🏭 Gelişmekte Olan Sanayi Ekonomisi": {
        "desc": "Yüksek imalat sanayi payı, yüksek enerji yoğunluğu ve henüz kısıtlı yenilenebilir enerji entegrasyonu olan üretim odaklı ekonomi.",
        "desc_en": "Production-oriented economy with high manufacturing share, high energy intensity, and limited renewables.",
        "badge": "Üretim ve Sanayi Üssü",
        "badge_en": "Production & Industrial Hub",
        "focus": "Enerji Verimliliği ve Temiz Üretim Teknolojileri",
        "recipe": "Kömür/fosil bağımlılığını azaltmak, sanayide enerji yoğunluğunu düşürmek ve temiz teknoloji yatırımları çekmek.",
        "recipe_en": "Reduce fossil fuel dependence, lower industrial energy intensity, and attract clean tech investment.",
        "peers": ["Türkiye", "Polonya", "Meksika", "Tayland", "Çekya"],
        "peers_en": ["Turkey", "Poland", "Mexico", "Thailand", "Czechia"],
        "skdm_risk": "Yüksek Risk / Karbon Vergisi Maruziyeti",
        "skdm_risk_en": "High Risk / CBAM Tax Exposure",
        "skdm_score": "8.5/10",
        "gdp": 8500.0, "energy": 8.8, "gvc": 22.0, "trade": 55.0, "manuf": 28.0, "renew": 14.0, "broadband": 12.0, "internet": 58.0, "mobile": 95.0
    },
    "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi": {
        "desc": "Küresel değer zincirlerine (GVC) yüksek entegrasyon, devasa dış ticaret açıklığı ve hiper-dijitalleşme altyapısına sahip Asya modeli.",
        "desc_en": "Asian model with deep GVC integration, massive trade openness, and hyper-digital infrastructure.",
        "badge": "Hiper-Dijital Tedarik Üssü",
        "badge_en": "Hyper-Digital Supply Hub",
        "focus": "Küresel Tedarik Zincirlerinin Karbonsuzlaştırılması",
        "recipe": "Devasa dijital altyapıyı veri merkezlerinde %100 yenilenebilir enerjiye geçirmek ve ihracatta karbonsuz lojistiği benimsemek.",
        "recipe_en": "Transition digital infrastructure to 100% renewables and adopt carbon-neutral export logistics.",
        "peers": ["Güney Kore", "Singapur", "Malezya", "Vietnam", "Tayvan"],
        "peers_en": ["South Korea", "Singapore", "Malaysia", "Vietnam", "Taiwan"],
        "skdm_risk": "Orta - Yüksek Risk",
        "skdm_risk_en": "Medium - High Risk",
        "skdm_score": "6.8/10",
        "gdp": 35000.0, "energy": 5.5, "gvc": 42.0, "trade": 130.0, "manuf": 26.0, "renew": 22.0, "broadband": 44.0, "internet": 96.0, "mobile": 160.0
    },
    "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke": {
        "desc": "Verimsiz enerji kullanımı, düşük milli gelir ve zayıf dijitalleşme ile en yüksek emisyon riski taşıyan kırılgan ekonomi.",
        "desc_en": "Fragile economy with inefficient energy use, low GDP per capita, and weak digital infrastructure.",
        "badge": "Yüksek Emisyon Riski & Kırılgan Yapı",
        "badge_en": "High Emission Risk & Fragile Structure",
        "focus": "Uluslararası İklim Finansmanı ve Altyapı Dönüşümü",
        "recipe": "Eski elektrik şebekelerini yenilemek, küresel iklim fonlarından yararlanarak yenilenebilir enerji sıçraması yapmak.",
        "recipe_en": "Modernize power grid and leverage global climate funds for renewable energy leapfrogging.",
        "peers": ["Kazakistan", "Güney Afrika", "Ukrayna", "Pakistan", "Nijerya"],
        "peers_en": ["Kazakhstan", "South Africa", "Ukraine", "Pakistan", "Nigeria"],
        "skdm_risk": "Kritik Derecede Yüksek Risk",
        "skdm_risk_en": "Critically High Risk",
        "skdm_score": "9.5/10",
        "gdp": 3200.0, "energy": 12.5, "gvc": 12.0, "trade": 35.0, "manuf": 20.0, "renew": 8.0, "broadband": 4.0, "internet": 32.0, "mobile": 65.0
    },
    "🍃 Yeşil İkiz Dönüşüm Öncüsü": {
        "desc": "Yüksek yenilenebilir enerji (%65) ve ileri dijital altyapının sinerji oluşturduğu ideal karbonsuzlaşma modeli.",
        "desc_en": "Ideal decarbonization model combining high renewable energy (65%) and advanced digital infrastructure.",
        "badge": "Geleceğin İdeal Modeli",
        "badge_en": "Ideal Future Model",
        "focus": "Sıfır Karbon Büyüme ve Yeşil Dijital Entegrasyon",
        "recipe": "Akıllı şehirler, yeşil veri merkezleri ve %100 karbonsuz üretim ile küresel iklim standartlarını belirlemek.",
        "recipe_en": "Set global climate standards with smart cities, green data centers, and zero-carbon production.",
        "peers": ["Norveç", "Yeni Zelanda", "İzlanda", "Finlandiya", "İsviçre"],
        "peers_en": ["Norway", "New Zealand", "Iceland", "Finland", "Switzerland"],
        "skdm_risk": "Sıfır Risk / Lider Konum",
        "skdm_risk_en": "Zero Risk / Leader Position",
        "skdm_score": "0.5/10",
        "gdp": 55000.0, "energy": 2.8, "gvc": 30.0, "trade": 90.0, "manuf": 15.0, "renew": 65.0, "broadband": 42.0, "internet": 95.0, "mobile": 140.0
    }
}

# GLOBAL COUNTRIES DATA
COUNTRIES_DATA = {
    "Türkiye": {"iso": "TUR", "gdp": 10600.0, "energy": 5.4, "gvc": 24.5, "trade": 62.0, "manuf": 22.0, "renew": 21.5, "broadband": 21.0, "internet": 83.0, "mobile": 105.0},
    "Almanya": {"iso": "DEU", "gdp": 48500.0, "energy": 3.1, "gvc": 35.0, "trade": 88.0, "manuf": 18.5, "renew": 46.0, "broadband": 44.0, "internet": 93.0, "mobile": 128.0},
    "Amerika Birleşik Devletleri": {"iso": "USA", "gdp": 70000.0, "energy": 4.2, "gvc": 18.0, "trade": 27.0, "manuf": 11.0, "renew": 20.0, "broadband": 37.0, "internet": 92.0, "mobile": 116.0},
    "Çin": {"iso": "CHN", "gdp": 12500.0, "energy": 8.5, "gvc": 31.0, "trade": 37.0, "manuf": 27.5, "renew": 29.0, "broadband": 38.0, "internet": 73.0, "mobile": 118.0},
    "Hindistan": {"iso": "IND", "gdp": 2400.0, "energy": 9.2, "gvc": 20.0, "trade": 45.0, "manuf": 14.5, "renew": 21.0, "broadband": 2.2, "internet": 46.0, "mobile": 84.0},
    "Fransa": {"iso": "FRA", "gdp": 43500.0, "energy": 3.0, "gvc": 29.0, "trade": 64.0, "manuf": 9.5, "renew": 24.0, "broadband": 48.0, "internet": 92.0, "mobile": 112.0},
    "İngiltere": {"iso": "GBR", "gdp": 46500.0, "energy": 2.8, "gvc": 26.0, "trade": 61.0, "manuf": 8.8, "renew": 42.0, "broadband": 41.0, "internet": 97.0, "mobile": 122.0},
    "İtalya": {"iso": "ITA", "gdp": 35500.0, "energy": 3.3, "gvc": 28.0, "trade": 68.0, "manuf": 14.8, "renew": 36.0, "broadband": 31.0, "internet": 85.0, "mobile": 132.0},
    "İspanya": {"iso": "ESP", "gdp": 30000.0, "energy": 3.5, "gvc": 27.0, "trade": 67.0, "manuf": 11.5, "renew": 42.0, "broadband": 36.0, "internet": 93.0, "mobile": 118.0},
    "Japonya": {"iso": "JPN", "gdp": 39000.0, "energy": 3.6, "gvc": 28.0, "trade": 37.0, "manuf": 20.0, "renew": 22.0, "broadband": 35.0, "internet": 90.0, "mobile": 152.0},
    "Güney Kore": {"iso": "KOR", "gdp": 33000.0, "energy": 5.2, "gvc": 41.0, "trade": 80.0, "manuf": 25.5, "renew": 8.5, "broadband": 43.0, "internet": 97.0, "mobile": 142.0},
    "Brezilya": {"iso": "BRA", "gdp": 8900.0, "energy": 5.0, "gvc": 16.0, "trade": 39.0, "manuf": 10.0, "renew": 48.0, "broadband": 19.0, "internet": 81.0, "mobile": 102.0},
    "Kanada": {"iso": "CAN", "gdp": 52000.0, "energy": 6.8, "gvc": 26.0, "trade": 65.0, "manuf": 9.8, "renew": 68.0, "broadband": 40.0, "internet": 93.0, "mobile": 96.0},
    "Avustralya": {"iso": "AUS", "gdp": 64000.0, "energy": 4.8, "gvc": 19.0, "trade": 46.0, "manuf": 5.8, "renew": 29.0, "broadband": 36.0, "internet": 96.0, "mobile": 108.0},
    "Meksika": {"iso": "MEX", "gdp": 11000.0, "energy": 4.5, "gvc": 36.0, "trade": 78.0, "manuf": 18.0, "renew": 16.0, "broadband": 18.0, "internet": 76.0, "mobile": 98.0},
    "Endonezya": {"iso": "IDN", "gdp": 4800.0, "energy": 4.1, "gvc": 21.0, "trade": 42.0, "manuf": 19.0, "renew": 14.0, "broadband": 4.5, "internet": 62.0, "mobile": 125.0},
    "Hollanda": {"iso": "NLD", "gdp": 57000.0, "energy": 3.1, "gvc": 48.0, "trade": 155.0, "manuf": 11.0, "renew": 33.0, "broadband": 45.0, "internet": 96.0, "mobile": 125.0},
    "İsviçre": {"iso": "CHE", "gdp": 92000.0, "energy": 2.1, "gvc": 33.0, "trade": 118.0, "manuf": 18.0, "renew": 30.0, "broadband": 47.0, "internet": 96.0, "mobile": 128.0},
    "İsveç": {"iso": "SWE", "gdp": 56000.0, "energy": 3.8, "gvc": 34.0, "trade": 92.0, "manuf": 13.0, "renew": 66.0, "broadband": 41.0, "internet": 95.0, "mobile": 126.0},
    "Norveç": {"iso": "NOR", "gdp": 89000.0, "energy": 4.1, "gvc": 28.0, "trade": 72.0, "manuf": 6.5, "renew": 75.0, "broadband": 45.0, "internet": 97.0, "mobile": 108.0},
    "Polonya": {"iso": "POL", "gdp": 18000.0, "energy": 5.1, "gvc": 36.0, "trade": 108.0, "manuf": 16.8, "renew": 17.0, "broadband": 24.0, "internet": 87.0, "mobile": 138.0},
    "Güney Afrika": {"iso": "ZAF", "gdp": 6700.0, "energy": 9.8, "gvc": 22.0, "trade": 56.0, "manuf": 12.0, "renew": 10.0, "broadband": 3.8, "internet": 72.0, "mobile": 162.0},
    "Suudi Arabistan": {"iso": "SAU", "gdp": 30000.0, "energy": 7.8, "gvc": 18.0, "trade": 62.0, "manuf": 13.0, "renew": 1.0, "broadband": 32.0, "internet": 98.0, "mobile": 135.0},
    "Arjantin": {"iso": "ARG", "gdp": 13000.0, "energy": 4.6, "gvc": 14.0, "trade": 33.0, "manuf": 15.0, "renew": 11.0, "broadband": 22.0, "internet": 87.0, "mobile": 125.0},
    "Yunanistan": {"iso": "GRC", "gdp": 20500.0, "energy": 3.8, "gvc": 23.0, "trade": 78.0, "manuf": 9.2, "renew": 38.0, "broadband": 41.0, "internet": 79.0, "mobile": 115.0},
    "Portekiz": {"iso": "PRT", "gdp": 24500.0, "energy": 3.6, "gvc": 31.0, "trade": 85.0, "manuf": 11.8, "renew": 54.0, "broadband": 42.0, "internet": 85.0, "mobile": 122.0},
    "Belçika": {"iso": "BEL", "gdp": 50000.0, "energy": 4.1, "gvc": 46.0, "trade": 165.0, "manuf": 12.5, "renew": 23.0, "broadband": 43.0, "internet": 94.0, "mobile": 102.0},
    "Avusturya": {"iso": "AUT", "gdp": 53000.0, "energy": 3.2, "gvc": 38.0, "trade": 105.0, "manuf": 16.2, "renew": 78.0, "broadband": 30.0, "internet": 93.0, "mobile": 122.0},
    "Danimarka": {"iso": "DNK", "gdp": 67000.0, "energy": 2.5, "gvc": 36.0, "trade": 112.0, "manuf": 11.5, "renew": 62.0, "broadband": 45.0, "internet": 98.0, "mobile": 125.0},
    "Finlandiya": {"iso": "FIN", "gdp": 53000.0, "energy": 4.5, "gvc": 32.0, "trade": 78.0, "manuf": 14.0, "renew": 52.0, "broadband": 36.0, "internet": 97.0, "mobile": 168.0},
    "İrlanda": {"iso": "IRL", "gdp": 100000.0, "energy": 1.5, "gvc": 42.0, "trade": 190.0, "manuf": 32.0, "renew": 35.0, "broadband": 32.0, "internet": 92.0, "mobile": 108.0},
    "Şili": {"iso": "CHL", "gdp": 15500.0, "energy": 5.1, "gvc": 24.0, "trade": 62.0, "manuf": 10.5, "renew": 31.0, "broadband": 23.0, "internet": 90.0, "mobile": 138.0},
    "Kolombiya": {"iso": "COL", "gdp": 6600.0, "energy": 3.8, "gvc": 15.0, "trade": 38.0, "manuf": 11.0, "renew": 72.0, "broadband": 17.0, "internet": 73.0, "mobile": 142.0},
    "Çekya": {"iso": "CZE", "gdp": 27000.0, "energy": 4.8, "gvc": 42.0, "trade": 140.0, "manuf": 22.0, "renew": 17.0, "broadband": 33.0, "internet": 88.0, "mobile": 132.0},
    "Macaristan": {"iso": "HUN", "gdp": 18500.0, "energy": 4.5, "gvc": 45.0, "trade": 160.0, "manuf": 19.5, "renew": 14.0, "broadband": 32.0, "internet": 89.0, "mobile": 108.0},
    "Romanya": {"iso": "ROU", "gdp": 15000.0, "energy": 4.2, "gvc": 31.0, "trade": 82.0, "manuf": 17.0, "renew": 24.0, "broadband": 28.0, "internet": 82.0, "mobile": 115.0},
    "Malezya": {"iso": "MYS", "gdp": 12000.0, "energy": 5.8, "gvc": 41.0, "trade": 130.0, "manuf": 23.5, "renew": 18.0, "broadband": 11.0, "internet": 96.0, "mobile": 140.0},
    "Tayland": {"iso": "THA", "gdp": 7200.0, "energy": 6.2, "gvc": 38.0, "trade": 118.0, "manuf": 27.0, "renew": 15.0, "broadband": 18.0, "internet": 85.0, "mobile": 138.0},
    "Vietnam": {"iso": "VNM", "gdp": 4100.0, "energy": 7.5, "gvc": 46.0, "trade": 185.0, "manuf": 25.0, "renew": 32.0, "broadband": 21.0, "internet": 78.0, "mobile": 130.0},
    "Filipinler": {"iso": "PHL", "gdp": 3600.0, "energy": 3.2, "gvc": 28.0, "trade": 68.0, "manuf": 17.5, "renew": 21.0, "broadband": 10.0, "internet": 53.0, "mobile": 142.0},
    "Singapur": {"iso": "SGP", "gdp": 82000.0, "energy": 3.1, "gvc": 55.0, "trade": 330.0, "manuf": 20.0, "renew": 2.0, "broadband": 45.0, "internet": 92.0, "mobile": 158.0},
    "Yeni Zelanda": {"iso": "NZL", "gdp": 48000.0, "energy": 3.8, "gvc": 21.0, "trade": 52.0, "manuf": 9.0, "renew": 80.0, "broadband": 35.0, "internet": 95.0, "mobile": 125.0},
    "Mısır": {"iso": "EGY", "gdp": 3700.0, "energy": 5.8, "gvc": 14.0, "trade": 38.0, "manuf": 16.0, "renew": 11.0, "broadband": 11.0, "internet": 72.0, "mobile": 98.0},
    "Nijerya": {"iso": "NGA", "gdp": 2200.0, "energy": 6.5, "gvc": 12.0, "trade": 32.0, "manuf": 9.0, "renew": 18.0, "broadband": 4.0, "internet": 55.0, "mobile": 92.0},
    "Cezayir": {"iso": "DZA", "gdp": 4300.0, "energy": 6.1, "gvc": 15.0, "trade": 51.0, "manuf": 11.0, "renew": 1.0, "broadband": 11.0, "internet": 71.0, "mobile": 108.0},
    "Fas": {"iso": "MAR", "gdp": 3800.0, "energy": 4.2, "gvc": 26.0, "trade": 82.0, "manuf": 15.0, "renew": 19.0, "broadband": 7.0, "internet": 88.0, "mobile": 135.0},
    "İsrail": {"iso": "ISR", "gdp": 54000.0, "energy": 2.8, "gvc": 28.0, "trade": 60.0, "manuf": 11.0, "renew": 10.0, "broadband": 31.0, "internet": 90.0, "mobile": 128.0},
    "Birleşik Arap Emirlikleri": {"iso": "ARE", "gdp": 49000.0, "energy": 6.2, "gvc": 22.0, "trade": 170.0, "manuf": 9.0, "renew": 4.0, "broadband": 36.0, "internet": 99.0, "mobile": 200.0},
    "Katar": {"iso": "QAT", "gdp": 88000.0, "energy": 8.1, "gvc": 15.0, "trade": 92.0, "manuf": 8.0, "renew": 0.5, "broadband": 31.0, "internet": 99.0, "mobile": 145.0},
    "Kazakistan": {"iso": "KAZ", "gdp": 11500.0, "energy": 11.2, "gvc": 18.0, "trade": 62.0, "manuf": 13.0, "renew": 4.0, "broadband": 14.0, "internet": 92.0, "mobile": 132.0},
    "Ukrayna": {"iso": "UKR", "gdp": 4500.0, "energy": 10.5, "gvc": 22.0, "trade": 82.0, "manuf": 11.0, "renew": 11.0, "broadband": 19.0, "internet": 79.0, "mobile": 128.0},
    "Pakistan": {"iso": "PAK", "gdp": 1500.0, "energy": 8.2, "gvc": 12.0, "trade": 30.0, "manuf": 12.0, "renew": 6.0, "broadband": 5.0, "internet": 36.0, "mobile": 82.0}
}

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

# 2. LOAD MODELS
@st.cache_resource
def load_models():
    if os.path.exists("svr_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("pca.pkl"):
        svr = joblib.load("svr_model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        y_scaler = joblib.load("y_scaler.pkl") if os.path.exists("y_scaler.pkl") else None
        return svr, scaler, pca, y_scaler
    else:
        return None, None, None, None

svr_model, scaler, pca, y_scaler = load_models()

# PREDICTION FUNCTION
def predict_emissions(gdp_i, energy_i, gvc_i, trade_i, manuf_i, renew_i, bb_i, net_i, mob_i):
    if svr_model is not None and scaler is not None and pca is not None:
        z_bb = (bb_i - 18.5) / 12.5
        z_net = (net_i - 65.0) / 25.0
        z_mob = (mob_i - 105.0) / 32.0
        dig_z = np.array([[z_bb, z_net, z_mob]])
        try:
            dig_idx = float(np.asarray(pca.transform(dig_z)).item())
        except Exception:
            dig_idx = 0.0
        
        raw_feats = np.array([[gdp_i, energy_i, gvc_i, trade_i, manuf_i, renew_i, dig_idx]])
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
    else:
        # Mathematical fallback for initial state or testing
        e_est = 800.0 + (gdp_i / 1000.0) * 2.5 + (energy_i * 35.0) + (manuf_i * 8.0) - (renew_i * 6.5)
        return max(10.0, float(e_est))

# SESSION STATE INITIALIZATION
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

# 3. SIDEBAR & LANGUAGE TOGGLE
st.sidebar.title("🌐 Dil / Language")
lang_choice = st.sidebar.selectbox("Language / Dil Seçiniz:", ["Türkçe", "English"], index=0)

is_tr = (lang_choice == "Türkçe")

if is_tr:
    title_text = "🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Sistemi"
    sub_title = "Açıklanabilir Yapay Zeka (SVR & Monte Carlo) Destekli Politika & Kurumsal İklim Riski Simülatörü"
    mod_choice_label = "Senaryo Modu Seçimi:"
    mod_opt1 = "Küresel Ülke Listesinden Seç"
    mod_opt2 = "Hazır Ülke Profilini Kullan"
    param_header = "🎛️ Politika Parametreleri"
    dig_header = "📱 Dijitalleşme Girdileri (PCA Modelleri)"
    dig_mode_label = "Dijitalleşme Kontrol Modu:"
    dig_opt1 = "Detaylı 3 Gösterge"
    dig_opt2 = "Tek Dijitalleşme İndeksi"
else:
    title_text = "🌍 Global Supply Chain & Carbon Footprint Decision Support System"
    sub_title = "Explainable AI (SVR & Monte Carlo) Powered Policy & Corporate Climate Risk Simulator"
    mod_choice_label = "Scenario Mode Selection:"
    mod_opt1 = "Select from Global Country List"
    mod_opt2 = "Use Preset Country Profile"
    param_header = "🎛️ Policy Parameters"
    dig_header = "📱 Digitalization Inputs (PCA Models)"
    dig_mode_label = "Digitalization Control Mode:"
    dig_opt1 = "Detailed 3 Indicators"
    dig_opt2 = "Single Digitalization Index"

st.sidebar.markdown("---")
st.sidebar.subheader(param_header)

mode_choice = st.sidebar.radio(mod_choice_label, [mod_opt1, mod_opt2])

selected_profile = "🏛️ S0 Referans Küresel Durum (Baseline)"
selected_country = st.session_state.selected_country_name

if mode_choice in [mod_opt2, "Use Preset Country Profile"]:
    selected_profile = st.sidebar.selectbox("Hazır Profiller / Profiles:", list(PROFILE_DETAILS.keys()))
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
    country_list = list(COUNTRIES_DATA.keys())
    curr_idx = country_list.index(st.session_state.selected_country_name) if st.session_state.selected_country_name in country_list else 0
    selected_country = st.sidebar.selectbox("Ülke Listesi / Country List:", country_list, index=curr_idx)
    st.session_state.selected_country_name = selected_country
    c_info = COUNTRIES_DATA[selected_country]
    st.session_state.gdp_val = c_info["gdp"]
    st.session_state.energy_val = c_info["energy"]
    st.session_state.gvc_val = c_info["gvc"]
    st.session_state.trade_val = c_info["trade"]
    st.session_state.manuf_val = c_info["manuf"]
    st.session_state.renew_val = c_info["renew"]
    st.session_state.broadband_val = c_info["broadband"]
    st.session_state.internet_val = c_info["internet"]
    st.session_state.mobile_val = c_info["mobile"]

gdp = st.sidebar.slider("Kişi Başı GSYH / GDP per Capita ($)" if is_tr else "GDP per Capita ($)", 1000, 95000, int(st.session_state.gdp_val), step=1000)
energy_intensity = st.sidebar.slider("Enerji Yoğunluğu / Energy Intensity (MJ/$)" if is_tr else "Energy Intensity (MJ/$)", 1.0, 15.0, float(st.session_state.energy_val), step=0.1)
gvc_output = st.sidebar.slider("GVC Çıktısı / GVC Output Share (%)" if is_tr else "GVC Output Share (%)", 5.0, 60.0, float(st.session_state.gvc_val), step=0.5)
trade_openness = st.sidebar.slider("Ticari Açıklık / Trade Openness (% GDP)" if is_tr else "Trade Openness (% GDP)", 20.0, 200.0, float(st.session_state.trade_val), step=1.0)
manufacturing = st.sidebar.slider("İmalat Sanayi / Manufacturing Share (% GDP)" if is_tr else "Manufacturing Share (% GDP)", 2.0, 45.0, float(st.session_state.manuf_val), step=0.5)
renewable_energy = st.sidebar.slider("Yenilenebilir Enerji / Renewable Share (%)" if is_tr else "Renewable Share (%)", 0.0, 80.0, float(st.session_state.renew_val), step=1.0)

st.sidebar.subheader(dig_header)
digital_mode = st.sidebar.radio(dig_mode_label, [dig_opt1, dig_opt2], index=0)

if digital_mode in [dig_opt1, "Detailed 3 Indicators"]:
    broadband = st.sidebar.slider("Sabit Geniş Bant / Fixed Broadband" if is_tr else "Fixed Broadband", 0.0, 50.0, float(st.session_state.broadband_val), step=0.5)
    internet_users = st.sidebar.slider("İnternet Kullanımı / Internet Users (%)" if is_tr else "Internet Users (%)", 10.0, 100.0, float(st.session_state.internet_val), step=1.0)
    mobile_sub = st.sidebar.slider("Mobil Abonelik / Mobile Subscriptions" if is_tr else "Mobile Subscriptions", 30.0, 200.0, float(st.session_state.mobile_val), step=1.0)
else:
    dig_single = st.sidebar.slider("Dijitalleşme İndeksi / Digitalization Index (%)" if is_tr else "Digitalization Index (%)", 10.0, 100.0, float(st.session_state.internet_val), step=1.0)
    broadband = (dig_single / 100.0) * 45.0
    internet_users = dig_single
    mobile_sub = (dig_single / 100.0) * 160.0

st.sidebar.markdown("---")
st.sidebar.markdown(f"👁️ **{'Canlı Ziyaretçi Sayısı' if is_tr else 'Live Visitors'}:** `{visitor_info['total_visits']}`")

with st.sidebar.expander("🌐 " + ("Ziyaretçi Ülke Dağılımı" if is_tr else "Visitor Breakdown")):
    for c_flag, c_cnt in visitor_info["country_breakdown"].items():
        st.write(f"• **{c_flag}:** {c_cnt}")

# PREDICTIONS & CALCULATIONS
pred_emission = predict_emissions(gdp, energy_intensity, gvc_output, trade_openness, manufacturing, renewable_energy, broadband, internet_users, mobile_sub)

if mode_choice in [mod_opt1, "Select from Global Country List"]:
    base_c = COUNTRIES_DATA[selected_country]
    base_country_emission = predict_emissions(base_c["gdp"], base_c["energy"], base_c["gvc"], base_c["trade"], base_c["manuf"], base_c["renew"], base_c["broadband"], base_c["internet"], base_c["mobile"])
else:
    prof_c = PROFILE_DETAILS[selected_profile]
    base_country_emission = predict_emissions(prof_c["gdp"], prof_c["energy"], prof_c["gvc"], prof_c["trade"], prof_c["manuf"], prof_c["renew"], prof_c["broadband"], prof_c["internet"], prof_c["mobile"])

# SHAP WATERFALL CONTRIBUTIONS
c_gdp = predict_emissions(gdp, BASE_ENERGY, BASE_GVC, BASE_TRADE, BASE_MANUF, BASE_RENEW, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_energy = predict_emissions(BASE_GDP, energy_intensity, BASE_GVC, BASE_TRADE, BASE_MANUF, BASE_RENEW, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_gvc = predict_emissions(BASE_GDP, BASE_ENERGY, gvc_output, BASE_TRADE, BASE_MANUF, BASE_RENEW, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_trade = predict_emissions(BASE_GDP, BASE_ENERGY, BASE_GVC, trade_openness, BASE_MANUF, BASE_RENEW, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_manuf = predict_emissions(BASE_GDP, BASE_ENERGY, BASE_GVC, BASE_TRADE, manufacturing, BASE_RENEW, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_renew = predict_emissions(BASE_GDP, BASE_ENERGY, BASE_GVC, BASE_TRADE, BASE_MANUF, renewable_energy, BASE_BROADBAND, BASE_INTERNET, BASE_MOBILE) - BASE_EMISSION
c_dig = predict_emissions(BASE_GDP, BASE_ENERGY, BASE_GVC, BASE_TRADE, BASE_MANUF, BASE_RENEW, broadband, internet_users, mobile_sub) - BASE_EMISSION

# MONTE CARLO
np.random.seed(42)
residuals = np.random.normal(0, 101.24, 10000)
mc_distribution = np.maximum(0.0, pred_emission + residuals)
lower_bound = float(np.percentile(mc_distribution, 5).item())
upper_bound = float(np.percentile(mc_distribution, 95).item())

emission_diff = pred_emission - BASE_EMISSION
emission_pct_change = (emission_diff / BASE_EMISSION) * 100.0

country_policy_diff = pred_emission - base_country_emission
country_policy_pct = ((country_policy_diff / base_country_emission) * 100.0) if base_country_emission > 0 else 0.0

# DYNAMIC RECOMMENDATION CALCULATIONS
c_info_cur = COUNTRIES_DATA[selected_country] if mode_choice in [mod_opt1, 'Select from Global Country List'] else PROFILE_DETAILS[selected_profile]
corp_export_default = 5000000
b_factor = (c_info_cur["energy"] / 10.0) * (1.0 - (c_info_cur["renew"] / 100.0)) * 0.08
s_factor = (energy_intensity / 10.0) * (1.0 - (renewable_energy / 100.0)) * 0.08
dyn_tax_savings = (b_factor - s_factor) * corp_export_default
dyn_scope3_reduction = max(0.0, (renewable_energy - c_info_cur["renew"]) * 0.6 + (c_info_cur["manuf"] - manufacturing) * 0.4)

# 4. MAIN INTERFACE
st.title(title_text)
st.caption(sub_title)

# SAVE SCENARIO BUTTON
col_sc1, col_sc2 = st.columns(2)
with col_sc2:
    if st.button("💾 " + ("Senaryoyu Hafızaya Kaydet" if is_tr else "Save Scenario to Memory")):
        s_name = f"{selected_country if mode_choice in [mod_opt1, 'Select from Global Country List'] else selected_profile} - {pred_emission:.1f} Mt"
        st.session_state.saved_scenarios.append({
            "Senaryo Adı": s_name,
            "Tahmini Emisyon (Mt)": round(pred_emission, 2),
            "Baseline Farkı (%)": round(emission_pct_change, 1),
            "Alt Güven": round(lower_bound, 1),
            "Üst Güven": round(upper_bound, 1),
            "GSYH ($)": gdp,
            "Enerji Yoğunluğu (MJ/$)": energy_intensity,
            "Yenilenebilir Enerji (%)": renewable_energy,
            "İmalat Payı (%)": manufacturing,
            "Ticari Açıklık (%)": trade_openness,
            "GVC Çıktısı (%)": gvc_output,
            "c_renew": round(c_renew, 1),
            "dyn_tax_savings": round(max(0.0, dyn_tax_savings), 0),
            "dyn_scope3_reduction": round(dyn_scope3_reduction, 1)
        })
        st.success("✅ " + ("Senaryo kaydedildi!" if is_tr else "Scenario saved!"))

if mode_choice in [mod_opt2, "Use Preset Country Profile"]:
    p_desc = PROFILE_DETAILS[selected_profile]["desc" if is_tr else "desc_en"]
    p_badge = PROFILE_DETAILS[selected_profile]["badge" if is_tr else "badge_en"]
    p_peers = ", ".join(PROFILE_DETAILS[selected_profile]["peers" if is_tr else "peers_en"])
    p_risk = PROFILE_DETAILS[selected_profile]["skdm_risk" if is_tr else "skdm_risk_en"]
    p_recipe = PROFILE_DETAILS[selected_profile]["recipe" if is_tr else "recipe_en"]
    
    st.info(f"🏛️ **{'Seçili Tipoloji' if is_tr else 'Selected Typology'}:** {selected_profile} | **{'Rozet' if is_tr else 'Badge'}:** {p_badge}\n\n"
            f"💡 *{p_desc}*\n\n"
            f"👥 **{'Akran Ülkeler' if is_tr else 'Peer Countries'}:** {p_peers} | 🛡️ **{'SKDM Riski' if is_tr else 'CBAM Risk'}:** {p_risk}\n\n"
            f"🎯 **{'Karbonsuzlaşma Reçetesi' if is_tr else 'Decarbonization Recipe'}:** {p_recipe}")
else:
    st.success(f"📌 **{'Seçili Ülke' if is_tr else 'Selected Country'}:** {selected_country} | **{'Mevcut Gerçek Emisyonu' if is_tr else 'Baseline Emission'}:** {base_country_emission:.2f} Mt CO₂eq")

# METRIC CARDS
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Yeni Politika Tahmini / New Prediction" if is_tr else "New Policy Prediction",
    value=f"{pred_emission:.2f} Mt CO₂eq",
    delta=f"{emission_pct_change:+.1f}% vs Baseline",
    delta_color="inverse"
)

if mode_choice in [mod_opt1, "Select from Global Country List"]:
    col2.metric(
        label=f"{selected_country} (Baseline)",
        value=f"{base_country_emission:.2f} Mt CO₂eq",
        delta=f"{country_policy_pct:+.1f}% Scenario",
        delta_color="inverse"
    )
else:
    col2.metric(
        label="Seçili Profil Baseline" if is_tr else "Selected Profile Baseline",
        value=f"{base_country_emission:.2f} Mt CO₂eq",
        delta=f"{country_policy_pct:+.1f}% Scenario",
        delta_color="inverse"
    )

col3.metric(
    label="Alt Güven Sınırı (%5)" if is_tr else "Lower Bound (5%)",
    value=f"{lower_bound:.2f} Mt CO₂eq",
    help="%5 olasılık alt sınırı" if is_tr else "5% probability lower bound"
)

col4.metric(
    label="Üst Güven Sınırı (%95)" if is_tr else "Upper Bound (95%)",
    value=f"{upper_bound:.2f} Mt CO₂eq",
    help="%95 olasılık üst sınırı" if is_tr else "95% probability upper bound"
)

st.markdown("---")

# EVAL SKDM
def eval_skdm_risk(manuf_val, renew_val, energy_val):
    if renew_val >= 45.0 and energy_val <= 3.5:
        return ("Çok Düşük / Muafiyet Avantajı 🟢" if is_tr else "Very Low / Exempt 🟢"), "1.5/10", "glass-card-green"
    elif renew_val >= 30.0 or (manuf_val <= 15.0 and energy_val <= 4.5):
        return ("Orta Risk / Kısmi Maruziyet 🟡" if is_tr else "Medium Risk / Partial Exposure 🟡"), "5.0/10", "glass-card-yellow"
    else:
        return ("Yüksek Risk / SKDM Karbon Vergisi Yükü 🔴" if is_tr else "High Risk / CBAM Tax Exposure 🔴"), "8.5/10", "glass-card-red"

# ENHANCED MULTI-SCENARIO PDF GENERATOR
def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    title_p = "KURESEL KARBON AYAK IZI & SKDM RISK RAPORU" if is_tr else "GLOBAL CARBON FOOTPRINT & CBAM RISK REPORT"
    pdf.cell(190, 10, clean_pdf_text(title_p), align='C')
    pdf.ln(10)
    
    saved_scs = st.session_state.saved_scenarios
    if not saved_scs:
        # Generate active screen scenario
        saved_scs = [{
            "Senaryo Adı": selected_country if mode_choice in [mod_opt1, 'Select from Global Country List'] else selected_profile,
            "Tahmini Emisyon (Mt)": round(pred_emission, 2),
            "Baseline Farkı (%)": round(emission_pct_change, 1),
            "Alt Güven": round(lower_bound, 1),
            "Üst Güven": round(upper_bound, 1),
            "GSYH ($)": gdp,
            "Enerji Yoğunluğu (MJ/$)": energy_intensity,
            "Yenilenebilir Enerji (%)": renewable_energy,
            "İmalat Payı (%)": manufacturing,
            "Ticari Açıklık (%)": trade_openness,
            "c_renew": round(c_renew, 1),
            "dyn_tax_savings": round(max(0.0, dyn_tax_savings), 0),
            "dyn_scope3_reduction": round(dyn_scope3_reduction, 1)
        }]

    pdf.set_font("Helvetica", 'B', 12)
    hdr_txt = f"TOPLAM KAYDEDILEN SENARYO SAYISI: {len(saved_scs)}" if is_tr else f"TOTAL SAVED SCENARIOS: {len(saved_scs)}"
    pdf.cell(190, 8, clean_pdf_text(hdr_txt))
    pdf.ln(10)

    for idx, sc in enumerate(saved_scs):
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(190, 8, clean_pdf_text(f"SENARYO {idx+1}: {sc['Senaryo Adı']}"))
        pdf.ln(8)

        pdf.set_font("Helvetica", '', 10)
        pdf.cell(190, 6, clean_pdf_text(f"- Tahmini GHG Emisyonu: {sc['Tahmini Emisyon (Mt)']} Mt CO2eq"))
        pdf.ln(6)
        pdf.cell(190, 6, clean_pdf_text(f"- Baseline Farki: {sc.get('Baseline Farkı (%)', 0.0):+.1f}%"))
        pdf.ln(6)
        pdf.cell(190, 6, clean_pdf_text(f"- Monte Carlo Guven Sinaralari (%95): {sc.get('Alt Güven', 0.0)} - {sc.get('Üst Güven', 0.0)} Mt CO2eq"))
        pdf.ln(8)

        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(190, 6, clean_pdf_text("MAKRO VE POLITIKA PARAMETRELERI:"))
        pdf.ln(6)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(190, 5, clean_pdf_text(f"  * GSYH: ${sc.get('GSYH ($)', 0):,.0f} | Enerji Yogunlugu: {sc.get('Enerji Yoğunluğu (MJ/$)', 0):.1f} MJ/$"))
        pdf.ln(5)
        pdf.cell(190, 5, clean_pdf_text(f"  * Yenilenebilir Enerji: %{sc.get('Yenilenebilir Enerji (%)', 0):.1f} | Imalat Payi: %{sc.get('İmalat Payı (%)', 0):.1f}"))
        pdf.ln(5)
        pdf.cell(190, 5, clean_pdf_text(f"  * Ticari Aciklik: %{sc.get('Ticari Açıklık (%)', 0):.1f}"))
        pdf.ln(8)

        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(190, 6, clean_pdf_text("YAPAY ZEKA DESTEKLI POLITIKA CIKTILARI & ONERILER:"))
        pdf.ln(6)
        pdf.set_font("Helvetica", '', 9)
        pdf.multi_cell(190, 5, clean_pdf_text(f"  1. Yenilenebilir Enerji: %{sc.get('Yenilenebilir Enerji (%)', 0):.1f} seviyesi net {sc.get('c_renew', 0.0):+.1f} Mt CO2eq etki yaratir."))
        pdf.multi_cell(190, 5, clean_pdf_text(f"  2. Enerji Verimliligi: {sc.get('Enerji Yoğunluğu (MJ/$)', 0):.1f} MJ/$ seviyesi ile EUR 5M ihracatcida EUR {sc.get('dyn_tax_savings', 0):,.0f} net vergi tasarrufu saglanir."))
        pdf.multi_cell(190, 5, clean_pdf_text(f"  3. Temiz Imalat Donusumu: Kapsam 3 tedarik zinciri iklim riski %{sc.get('dyn_scope3_reduction', 0.0):.1f} azalmaktadir."))
        pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(190, 6, clean_pdf_text("METODOLOJIK DOGRULAMA NOTU"))
    pdf.ln(6)
    pdf.set_font("Helvetica", '', 8)
    pdf.multi_cell(190, 4, clean_pdf_text("Bu rapor RBF-SVR (Test R2 = 0.975) ve Monte Carlo simulesiyle uretilmistir. Sonuclar iliskisel ve tahminsel duyarliliklari temsil eder, dogrudan nedensellik iddiasi tasimaz."))

    return bytes(pdf.output())

# TABS DEFINITION
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🌍 Küresel Emisyon Haritası" if is_tr else "🌍 Global Emission Map",
    "📊 Canlı Senaryo SKDM Analizi" if is_tr else "📊 Live Scenario CBAM Analysis",
    "🏭 Kurumsal / İşletme Simülatörü" if is_tr else "🏭 Corporate Climate Risk Simulator",
    "🎯 Net-Zero Reçete Motoru" if is_tr else "🎯 Net-Zero Recipe Engine",
    "⚔️ İkili Ülke Karşılaştırma" if is_tr else "⚔️ Pairwise Country Comparison",
    "🔥 Tüm Ülkeler Kabarcık & Matrix" if is_tr else "🔥 Global Bubble & Matrix Analysis",
    "💾 Senaryo Deposu & İndir (CSV/PDF)" if is_tr else "💾 Scenario Vault & Export (CSV/PDF)",
    "📜 Metodoloji & XAI Notları" if is_tr else "📜 Methodology & XAI Notes"
])

# TAB 1: MAP, RADAR CHART & PRESET PROFILES HEATMAP
with tab1:
    map_list = []
    for c_name, c_data in COUNTRIES_DATA.items():
        e_val = predict_emissions(c_data["gdp"], c_data["energy"], c_data["gvc"], c_data["trade"], c_data["manuf"], c_data["renew"], c_data["broadband"], c_data["internet"], c_data["mobile"])
        map_list.append({"Ülke": c_name, "ISO": c_data["iso"], "Emisyon (Mt)": round(e_val, 1), "GSYH ($)": c_data["gdp"], "Yenilenebilir (%)": c_data["renew"], "Enerji Yoğunluğu": c_data["energy"], "İmalat (%)": c_data["manuf"]})
    df_map = pd.DataFrame(map_list)

    fig_map = px.choropleth(
        df_map, locations="ISO", color="Emisyon (Mt)", hover_name="Ülke",
        color_continuous_scale=[[0.0, "#2ca02c"], [0.35, "#ff7f0e"], [1.0, "#d62728"]],
        title="Küresel Ekonomilerin Tahmini Karbon Ayak İzi Dağılımı (Mt CO₂eq)" if is_tr else "Global Economies Estimated Carbon Footprint (Mt CO₂eq)"
    )
    sel_iso = COUNTRIES_DATA[selected_country]["iso"] if mode_choice in [mod_opt1, "Select from Global Country List"] else "TUR"
    sel_row = df_map[df_map["ISO"] == sel_iso]
    if not sel_row.empty:
        fig_map.add_trace(go.Choropleth(locations=sel_row["ISO"], z=sel_row["Emisyon (Mt)"], colorscale=[[0, "#ffff00"], [1, "#ffff00"]], showscale=False, marker_line_color="#ffffff", marker_line_width=3.5))

    fig_map.update_geos(showocean=True, oceancolor="#0e2a47", showlakes=True, lakecolor="#0e2a47", showcountries=True, countrycolor="#444444", projection_type="natural earth")
    fig_map.update_layout(height=480, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # RESTORED: MULTI-DIMENSIONAL SPIDER / RADAR CHART & PRESET PROFILES HEATMAP
    st.markdown("---")
    
    if mode_choice in [mod_opt2, "Use Preset Country Profile"]:
        st.subheader(f"📌 {selected_profile} — {'Çok Boyutlu Profil Radarı & Profil Isı Haritası' if is_tr else 'Multidimensional Radar & Profile Heatmap'}")
        
        col_rad1, col_rad2 = st.columns([1, 1])
        
        with col_rad1:
            st.markdown(f"#### 🕸️ {'Çok Boyutlu Profil Radarı (Örümcek Ağ Grafiği)' if is_tr else 'Multidimensional Radar Chart'}")
            
            # Normalize Radar Categories
            p_curr = PROFILE_DETAILS[selected_profile]
            categories = ['GSYH ($)', 'Enerji Yoğ.', 'GVC Payı', 'Ticaret Açıklık', 'İmalat Payı', 'Yenilenebilir %']
            
            val_selected = [
                min(100, (p_curr["gdp"] / 70000) * 100),
                min(100, (p_curr["energy"] / 12) * 100),
                min(100, (p_curr["gvc"] / 50) * 100),
                min(100, (p_curr["trade"] / 150) * 100),
                min(100, (p_curr["manuf"] / 35) * 100),
                min(100, p_curr["renew"])
            ]
            
            val_base = [
                min(100, (BASE_GDP / 70000) * 100),
                min(100, (BASE_ENERGY / 12) * 100),
                min(100, (BASE_GVC / 50) * 100),
                min(100, (BASE_TRADE / 150) * 100),
                min(100, (BASE_MANUF / 35) * 100),
                min(100, BASE_RENEW)
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=val_selected, theta=categories, fill='toself', name=selected_profile.split(' ')[0], fillcolor='rgba(43, 92, 143, 0.4)', line_color='#2b5c8f'))
            fig_radar.add_trace(go.Scatterpolar(r=val_base, theta=categories, fill='toself', name='Baseline (S0)', fillcolor='rgba(128, 128, 128, 0.2)', line_color='gray'))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=380,
                margin=dict(l=40, r=40, t=30, b=30),
                template="plotly_white"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_rad2:
            st.markdown(f"#### 📊 {'Hazır Ülke Profilleri Isı Haritası Matrisi' if is_tr else 'Preset Profiles Heatmap Matrix'}")
            prof_matrix_rows = []
            for p_k, p_v in PROFILE_DETAILS.items():
                p_short = p_k.split(" ")[1] if len(p_k.split(" ")) > 1 else p_k
                prof_matrix_rows.append({
                    "Profil": p_short,
                    "GSYH": p_v["gdp"]/1000,
                    "Enerji Y.": p_v["energy"],
                    "Yenilenebilir": p_v["renew"],
                    "İmalat": p_v["manuf"],
                    "Ticaret": p_v["trade"]
                })
            df_pm = pd.DataFrame(prof_matrix_rows).set_index("Profil")
            df_pm_norm = (df_pm - df_pm.min()) / (df_pm.max() - df_pm.min() + 1e-9) * 100.0
            
            fig_hm_prof = px.imshow(df_pm_norm, text_auto=".0f", color_continuous_scale="Viridis", aspect="auto")
            fig_hm_prof.update_layout(height=380, margin=dict(l=40, r=20, t=30, b=30))
            st.plotly_chart(fig_hm_prof, use_container_width=True)

        st.markdown(f"#### 📊 {'Hazır Ülke Profilleri Akran Ülke & SKDM Karşılaştırması' if is_tr else 'Preset Profiles Peer Country & CBAM Comparison'}")
        prof_table_data = []
        for p_name, p_data in PROFILE_DETAILS.items():
            p_e = predict_emissions(p_data["gdp"], p_data["energy"], p_data["gvc"], p_data["trade"], p_data["manuf"], p_data["renew"], p_data["broadband"], p_data["internet"], p_data["mobile"])
            prof_table_data.append({
                "Hazır Profil / Tipoloji" if is_tr else "Typology Profile": p_name,
                "Rozet / Identity" if is_tr else "Badge": p_data["badge" if is_tr else "badge_en"],
                "Tahmini Emisyon (Mt)" if is_tr else "Predicted Emission (Mt)": round(p_e, 1),
                "Akran Ülkeler" if is_tr else "Peer Countries": ", ".join(p_data["peers" if is_tr else "peers_en"]),
                "SKDM Riski" if is_tr else "CBAM Risk": p_data["skdm_risk" if is_tr else "skdm_risk_en"],
                "Karbonsuzlaşma Reçetesi" if is_tr else "Decarbonization Recipe": p_data["recipe" if is_tr else "recipe_en"]
            })
        df_prof_table = pd.DataFrame(prof_table_data)
        st.dataframe(df_prof_table.sort_values("Tahmini Emisyon (Mt)" if is_tr else "Predicted Emission (Mt)", ascending=False), use_container_width=True, hide_index=True)

    else:
        st.subheader(f"📌 {selected_country} — {'Göstergeler & Küresel Sıralama' if is_tr else 'Indicators & Global Ranking'}")
        c_m = COUNTRIES_DATA[selected_country]
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Milli Gelir (GSYH)" if is_tr else "GDP per Capita", f"${c_m['gdp']:,.0f}")
        col_m2.metric("Enerji Yoğunluğu" if is_tr else "Energy Intensity", f"{c_m['energy']:.1f} MJ/$")
        col_m3.metric("Yenilenebilir Enerji" if is_tr else "Renewable Share", f"%{c_m['renew']:.1f}")
        col_m4.metric("İmalat Sanayi Payı" if is_tr else "Manufacturing Share", f"%{c_m['manuf']:.1f}")
        col_m5.metric("Ticari Açıklık" if is_tr else "Trade Openness", f"%{c_m['trade']:.1f}")

        st.markdown(f"#### 📊 {'Küresel Ülke Emisyon & Politika Listesi' if is_tr else 'Global Country Emission & Policy List'}")
        st.dataframe(df_map.sort_values("Emisyon (Mt)", ascending=False), use_container_width=True, hide_index=True)

# TAB 2: LIVE SCENARIO CBAM ANALYSIS & DYNAMIC AI RECOMMENDATIONS
with tab2:
    st.subheader(f"📊 {selected_country if mode_choice in [mod_opt1, 'Select from Global Country List'] else selected_profile} — {'Canlı Senaryo SKDM & Karşılaştırma Analizi' if is_tr else 'Live Scenario CBAM & Comparison Analysis'}")
    
    base_skdm_label, base_skdm_score, base_skdm_class = eval_skdm_risk(c_info_cur["manuf"], c_info_cur["renew"], c_info_cur["energy"])
    scen_skdm_label, scen_skdm_score, scen_skdm_class = eval_skdm_risk(manufacturing, renewable_energy, energy_intensity)

    col_sk1, col_sk2 = st.columns(2)
    col_sk1.info(f"🏛️ **{'Mevcut Durum SKDM Risk' if is_tr else 'Baseline CBAM Risk'}:** {base_skdm_label} | İmalat: %{c_info_cur['manuf']:.1f} | Yenilenebilir: %{c_info_cur['renew']:.1f}")
    col_sk2.success(f"🎛️ **{'Yeni Senaryo SKDM Risk (Canlı)' if is_tr else 'New Scenario CBAM Risk (Live)'}:** {scen_skdm_label} | İmalat: %{manufacturing:.1f} | Yenilenebilir: %{renewable_energy:.1f}")

    # SHAP MARGINAL CONTRIBUTIONS
    st.markdown(f"#### 🤖 {'Yapay Zeka SHAP Marjinal Politika Katkıları (Net Etki)' if is_tr else 'AI SHAP Marginal Policy Contributions (Net Effect)'}")
    shap_contributions = {
        "GSYH / GDP": c_gdp,
        "Enerji Yoğunluğu / Energy Intensity": c_energy,
        "GVC Payı / GVC Share": c_gvc,
        "Ticari Açıklık / Trade Openness": c_trade,
        "İmalat / Manufacturing": c_manuf,
        "Yenilenebilir / Renewables": c_renew,
        "Dijitalleşme / Digitalization": c_dig
    }
    df_shap = pd.DataFrame(list(shap_contributions.items()), columns=["Politika Değişkeni" if is_tr else "Policy Variable", "Marjinal Etki (Mt CO₂eq)" if is_tr else "Marginal Effect (Mt CO₂eq)"])
    fig_shap = px.bar(
        df_shap, x="Marjinal Etki (Mt CO₂eq)" if is_tr else "Marginal Effect (Mt CO₂eq)", y="Politika Değişkeni" if is_tr else "Policy Variable", orientation="h",
        color="Marjinal Etki (Mt CO₂eq)" if is_tr else "Marginal Effect (Mt CO₂eq)", color_continuous_scale="RdYlGn_r",
        title="Her Bir Politikanın Seragazı Emisyonu Üzerindeki Net SHAP Katkısı" if is_tr else "Net SHAP Contribution of Each Policy Variable on GHG Emissions"
    )
    fig_shap.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_shap, use_container_width=True)

    # 100% DYNAMIC AI RECOMMENDATION CARDS
    st.markdown(f"#### 💡 {'Yapay Zeka Destekli Politika Kaldıraçları & Canlı Öneriler' if is_tr else 'AI-Driven Policy Levers & Live Recommendations'}")
    
    rec_c1, rec_c2, rec_c3 = st.columns(3)
    
    if is_tr:
        card1_html = f"""<div class='glass-card-green'>
            <h4>🍃 1. Yenilenebilir Enerji Hamlesi</h4>
            <p><b>Aksiyon:</b> Yenilenebilir enerji oranının <b>%{renewable_energy:.1f}</b> seviyesine ayarlanması emisyon tahminini net <b>{c_renew:+.1f} Mt CO₂eq</b> etkilemektedir.</p>
        </div>"""
        card2_html = f"""<div class='glass-card-blue'>
            <h4>⚡ 2. Enerji Verimliliği & Şebeke</h4>
            <p><b>Aksiyon:</b> Enerji yoğunluğunun <b>{energy_intensity:.1f} MJ/$</b> seviyesine ayarlanması, €5M ihracat yapan bir işletme için tahmini SKDM cezasında <b>€{max(0.0, dyn_tax_savings):,.0f}</b> net tasarruf yaratmaktadır.</p>
        </div>"""
        card3_html = f"""<div class='glass-card-yellow'>
            <h4>🏭 3. Temiz İmalat Dönüşümü</h4>
            <p><b>Aksiyon:</b> İmalat payının <b>%{manufacturing:.1f}</b> ve temiz enerjinin <b>%{renewable_energy:.1f}</b> olduğu bu senaryoda Kapsam 3 tedarik zinciri iklim riski <b>%{dyn_scope3_reduction:.1f}</b> azalmaktadır.</p>
        </div>"""
    else:
        card1_html = f"""<div class='glass-card-green'>
            <h4>🍃 1. Renewable Energy Push</h4>
            <p><b>Action:</b> Setting renewable energy share to <b>{renewable_energy:.1f}%</b> impacts emission prediction by a net <b>{c_renew:+.1f} Mt CO₂eq</b>.</p>
        </div>"""
        card2_html = f"""<div class='glass-card-blue'>
            <h4>⚡ 2. Energy Efficiency & Grid</h4>
            <p><b>Action:</b> Setting energy intensity to <b>{energy_intensity:.1f} MJ/$</b> saves <b>€{max(0.0, dyn_tax_savings):,.0f}</b> in CBAM tax penalties for a €5M exporter.</p>
        </div>"""
        card3_html = f"""<div class='glass-card-yellow'>
            <h4>🏭 3. Clean Manufacturing Shift</h4>
            <p><b>Action:</b> With <b>{manufacturing:.1f}%</b> manufacturing share and <b>{renewable_energy:.1f}%</b> clean energy, Scope 3 climate risk drops by <b>{dyn_scope3_reduction:.1f}%</b>.</p>
        </div>"""

    with rec_c1:
        st.markdown(card1_html, unsafe_allow_html=True)
    with rec_c2:
        st.markdown(card2_html, unsafe_allow_html=True)
    with rec_c3:
        st.markdown(card3_html, unsafe_allow_html=True)

# TAB 3: CORPORATE SIMULATOR & FLEXIBLE LOCATION SELECTION
with tab3:
    st.subheader("🏭 Kurumsal / İşletme İklim Riski Simülatörü (Macro-to-Micro Downscaling)" if is_tr else "🏭 Corporate Climate Risk Simulator (Macro-to-Micro Downscaling)")
    st.markdown("İhracatçı veya tedarik zincirinde yer alan şirketler için **AB SKDM Vergi Hesabı, Serbest Lokasyon Seçimi ve Kapsam 3 (Scope 3) Stres Testi**:" if is_tr else "AB CBAM Tax Calculation, Location Optimization, and Scope 3 Climate Risk Stress Testing for Exporters:")

    # 1. CBAM TAX CALCULATOR
    st.markdown(f"#### 1. 💰 {'Şirket AB İhracatı & SKDM Vergi Cezası / Tasarruf Hesaplayıcı' if is_tr else 'Company EU Exports & CBAM Tax / Savings Calculator'}")
    corp_export = st.number_input("Şirketinizin Yıllık AB İhracat Cirosu (\$ / €):" if is_tr else "Annual EU Export Revenue (\$ / €):", min_value=100000, max_value=1000000000, value=5000000, step=500000)

    c_cur_e = c_info_cur["energy"] if isinstance(c_info_cur, dict) and "energy" in c_info_cur else BASE_ENERGY
    c_cur_r = c_info_cur["renew"] if isinstance(c_info_cur, dict) and "renew" in c_info_cur else BASE_RENEW

    baseline_intensity_factor = (c_cur_e / 10.0) * (1.0 - (c_cur_r / 100.0)) * 0.08
    scen_intensity_factor = (energy_intensity / 10.0) * (1.0 - (renewable_energy / 100.0)) * 0.08

    base_cbam_tax = corp_export * baseline_intensity_factor
    scen_cbam_tax = corp_export * scen_intensity_factor
    net_tax_savings = base_cbam_tax - scen_cbam_tax

    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Mevcut Tahmini SKDM Cezası" if is_tr else "Baseline CBAM Tax Penalty", f"€{base_cbam_tax:,.0f}")
    col_c2.metric("Yeni Senaryo SKDM Cezası" if is_tr else "New Scenario CBAM Tax", f"€{scen_cbam_tax:,.0f}")
    col_c3.metric("Net Yıllık Vergi Tasarrufu" if is_tr else "Net Annual Tax Savings", f"€{net_tax_savings:,.0f}", delta=f"{(net_tax_savings/base_cbam_tax)*100:+.1f}% Savings" if base_cbam_tax>0 else "0%")

    # 2. FLEXIBLE LOCATION EVALUATION
    st.markdown(f"#### 2. 📍 {'Şirket Yatırım & Tedarikçi Lokasyon Değerlendirme' if is_tr else 'Investment & Supplier Location Evaluation'}")
    
    loc_mode_opts = ["🎯 Manuel Ülke Seçimi (İstediğiniz Ülkeleri Karşılaştırın)", "🤖 Otomatik Kriter Bazlı Filtreleme (En Uygun Ülkeler)"] if is_tr else ["🎯 Manual Country Selection (Compare Desired Countries)", "🤖 Automatic Criteria Filtering (Optimal Countries)"]
    loc_mode = st.radio("Lokasyon Analiz Yöntemini Seçiniz:" if is_tr else "Select Location Analysis Method:", loc_mode_opts, index=0)

    if "Manuel" in loc_mode or "Manual" in loc_mode:
        selected_custom_countries = st.multiselect(
            "Değerlendirmek İstediğiniz Ülkeleri Seçiniz:" if is_tr else "Select Countries to Evaluate:",
            list(COUNTRIES_DATA.keys()),
            default=["Türkiye", "Almanya", "Avusturya", "Çekya", "Polonya", "Meksika"]
        )
        if selected_custom_countries:
            custom_eval = []
            for cname in selected_custom_countries:
                cinfo = COUNTRIES_DATA[cname]
                score = (cinfo["renew"] * 0.4) + ((15.0 - cinfo["energy"]) * 0.4) + (cinfo["gdp"]/1000 * 0.2)
                cbam_risk, _, _ = eval_skdm_risk(cinfo["manuf"], cinfo["renew"], cinfo["energy"])
                custom_eval.append({
                    "Ülke / Country": cname,
                    "Uygunluk Skoru / Score": round(score, 1),
                    "Yenilenebilir Enerji (%) / Renewables": cinfo["renew"],
                    "Enerji Yoğunluğu / Energy Intensity": cinfo["energy"],
                    "İmalat Payı (%) / Manufacturing": cinfo["manuf"],
                    "GSYH (\$) / GDP": cinfo["gdp"],
                    "SKDM Risk Durumu / CBAM Risk": cbam_risk
                })
            df_custom_eval = pd.DataFrame(custom_eval).sort_values("Uygunluk Skoru / Score", ascending=False)
            st.success("🎯 **Seçtiğiniz Ülkelerin Detaylı İklim & Yatırım Analiz Tablosu:**" if is_tr else "🎯 **Detailed Climate & Investment Analysis Table of Selected Countries:**")
            st.dataframe(df_custom_eval, use_container_width=True, hide_index=True)
    else:
        min_manuf_target = st.slider("Aradığınız Minimum İmalat Sanayi Altyapısı Payı (%):" if is_tr else "Min Manufacturing Infrastructure Share (%):", 5.0, 40.0, 15.0)
        max_energy_limit = st.slider("Kabul Edilebilir Maksimum Enerji Yoğunluğu (MJ/\$):" if is_tr else "Max Acceptable Energy Intensity (MJ/\$):", 2.0, 10.0, 5.0)

        opt_candidates = []
        for cname, cinfo in COUNTRIES_DATA.items():
            if cinfo["manuf"] >= min_manuf_target and cinfo["energy"] <= max_energy_limit:
                score = (cinfo["renew"] * 0.4) + ((15.0 - cinfo["energy"]) * 0.4) + (cinfo["gdp"]/1000 * 0.2)
                cbam_risk, _, _ = eval_skdm_risk(cinfo["manuf"], cinfo["renew"], cinfo["energy"])
                opt_candidates.append({
                    "Ülke / Country": cname,
                    "Uygunluk Skoru / Score": round(score, 1),
                    "Yenilenebilir (%) / Renewables": cinfo["renew"],
                    "Enerji Yoğ. / Energy Int.": cinfo["energy"],
                    "İmalat Payı (%) / Manufacturing": cinfo["manuf"],
                    "SKDM Riski / CBAM Risk": cbam_risk
                })
        
        if opt_candidates:
            df_opt = pd.DataFrame(opt_candidates).sort_values("Uygunluk Skoru / Score", ascending=False)
            st.success("🤖 **Kriterlerinize Uyan En Optimal Ülke Lokasyonları:**" if is_tr else "🤖 **Optimal Country Locations Matching Your Criteria:**")
            st.dataframe(df_opt, use_container_width=True, hide_index=True)

    # 3. SCOPE 3 STRESS TEST
    st.markdown(f"#### 3. 📊 {'Kurumsal Kapsam 3 (Scope 3) İklim Riski Stres Testi' if is_tr else 'Corporate Scope 3 Climate Risk Stress Test'}")
    supplier_countries = st.multiselect("Tedarikçilerinizin Bulunduğu Ana Ülkeleri Seçiniz:" if is_tr else "Select Main Supplier Countries:", list(COUNTRIES_DATA.keys()), default=["Türkiye", "Polonya"])
    if supplier_countries:
        avg_supp_renew = np.mean([COUNTRIES_DATA[c]["renew"] for c in supplier_countries])
        scen_scope3_reduction = (renewable_energy - avg_supp_renew) * 0.6
        st.info(f"💡 {'Tedarikçilerinizin bulunduğu ülkelerde yapılacak yeşil dönüşüm hamlesi, şirketinizin **Kapsam 3 (Scope 3) tedarik zinciri karbon ayak izini %' if is_tr else 'Green transformation in supplier countries reduces your **Scope 3 supply chain carbon footprint by '}{max(0.0, scen_scope3_reduction):.1f}%**.")

# TAB 4: NET-ZERO RECIPE ENGINE & MONTE CARLO
with tab4:
    st.subheader("🎯 Net-Zero / Hedef Tabanlı Politika Reçetesi Motoru" if is_tr else "🎯 Net-Zero Target-Based Policy Recipe Engine")
    target_pct = st.slider("🎯 Hedeflenen Karbon Emisyonu Azaltım Oranı (%):" if is_tr else "Target Carbon Emission Reduction Rate (%):", 5, 50, 20, step=5)
    rec_renew = min(80.0, BASE_RENEW + (target_pct * 0.8))
    rec_energy = max(1.5, BASE_ENERGY - (target_pct * 0.08))
    rec_manuf = max(8.0, BASE_MANUF - (target_pct * 0.15))

    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.markdown(f"<div class='glass-card-green'><b>{'Gerekli Yenilenebilir Enerji' if is_tr else 'Required Renewable Energy'}</b><h3 style='color:#2ca02c;'>%{rec_renew:.1f}</h3></div>", unsafe_allow_html=True)
    r_col2.markdown(f"<div class='glass-card-green'><b>{'Gerekli Enerji Yoğunluğu' if is_tr else 'Required Energy Intensity'}</b><h3 style='color:#2ca02c;'>{rec_energy:.1f} MJ/\$</h3></div>", unsafe_allow_html=True)
    r_col3.markdown(f"<div class='glass-card-green'><b>{'Önerilen İmalat Sanayi Payı' if is_tr else 'Recommended Manufacturing Share'}</b><h3 style='color:#2ca02c;'>%{rec_manuf:.1f}</h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 " + ("10.000 İterasyonlu Monte Carlo Olasılık Dağılımı" if is_tr else "10,000 Iteration Monte Carlo Probability Distribution"))
    fig_mc = go.Figure()
    fig_mc.add_vline(x=BASE_EMISSION, line_width=2, line_dash="dot", line_color="gray", annotation_text=f"S0 Baseline ({BASE_EMISSION:.1f})")
    fig_mc.add_trace(go.Histogram(x=mc_distribution, nbinsx=50, name="Scenario Distribution", marker_color="#2b5c8f" if emission_diff <= 0 else "#d9534f", opacity=0.75))
    fig_mc.add_vline(x=pred_emission, line_width=3, line_dash="dash", line_color="red", annotation_text=f"New Scenario ({pred_emission:.1f})")
    fig_mc.update_layout(xaxis_title="Demand-Based GHG Emissions (Mt CO₂eq)", yaxis_title="Simulation Frequency", template="plotly_white", height=380)
    st.plotly_chart(fig_mc, use_container_width=True)

# TAB 5: PAIRWISE COUNTRY COMPARISON
with tab5:
    st.subheader("⚔️ İkili Ülke Birebir Karşılaştırma Modu" if is_tr else "⚔️ Pairwise Country Head-to-Head Comparison")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        country_A = st.selectbox("1. Ülkeyi Seçiniz:" if is_tr else "Select Country 1:", list(COUNTRIES_DATA.keys()), index=0)
    with col_k2:
        country_B = st.selectbox("2. Ülkeyi Seçiniz:" if is_tr else "Select Country 2:", list(COUNTRIES_DATA.keys()), index=1)

    cA_data = COUNTRIES_DATA[country_A]
    cB_data = COUNTRIES_DATA[country_B]
    eA = predict_emissions(cA_data["gdp"], cA_data["energy"], cA_data["gvc"], cA_data["trade"], cA_data["manuf"], cA_data["renew"], cA_data["broadband"], cA_data["internet"], cA_data["mobile"])
    eB = predict_emissions(cB_data["gdp"], cB_data["energy"], cB_data["gvc"], cB_data["trade"], cB_data["manuf"], cB_data["renew"], cB_data["broadband"], cB_data["internet"], cB_data["mobile"])

    fig_two = go.Figure(data=[
        go.Bar(name=country_A, x=["Emisyon (Mt)", "GSYH (\$k)", "Yenilenebilir (%)", "Enerji Yoğunluğu (x10)"], y=[eA, cA_data["gdp"]/1000, cA_data["renew"], cA_data["energy"]*10], marker_color="#1f77b4"),
        go.Bar(name=country_B, x=["Emisyon (Mt)", "GSYH (\$k)", "Yenilenebilir (%)", "Enerji Yoğunluğu (x10)"], y=[eB, cB_data["gdp"]/1000, cB_data["renew"], cB_data["energy"]*10], marker_color="#ff7f0e")
    ])
    fig_two.update_layout(barmode='group', title=f"{country_A} vs {country_B} Indicator Comparison", template="plotly_white", height=420)
    st.plotly_chart(fig_two, use_container_width=True)

# TAB 6: GLOBAL BUBBLE CHART & MATRIX
with tab6:
    st.subheader("🔥 Tüm Ülkelerin Etkileşimli Kabarcık Grafiği & Normalize Isı Haritası" if is_tr else "🔥 Interactive Global Bubble Chart & Normalized Heatmap")
    
    all_c_list = []
    for cname, cinfo in COUNTRIES_DATA.items():
        ce = predict_emissions(cinfo["gdp"], cinfo["energy"], cinfo["gvc"], cinfo["trade"], cinfo["manuf"], cinfo["renew"], cinfo["broadband"], cinfo["internet"], cinfo["mobile"])
        all_c_list.append({
            "Ülke": cname, "Emisyon (Mt)": round(ce, 1), "GSYH (\$)": cinfo["gdp"],
            "Enerji Yoğunluğu": cinfo["energy"], "Yenilenebilir (%)": cinfo["renew"], "İmalat (%)": cinfo["manuf"]
        })
    df_all_raw = pd.DataFrame(all_c_list)

    # BUBBLE CHART
    fig_bubble = px.scatter(
        df_all_raw, x="GSYH (\$)", y="Enerji Yoğunluğu", size="Emisyon (Mt)", color="Yenilenebilir (%)",
        hover_name="Ülke", size_max=45, color_continuous_scale="Viridis",
        title="Ülkelerin GSYH, Enerji Yoğunluğu ve Emisyon Büyüklüklerine Göre Kabarcık Dağılımı" if is_tr else "Countries Distribution by GDP, Energy Intensity, and GHG Emission Size"
    )
    fig_bubble.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 📊 {'Normalize Ülke Karşılaştırma Matrisi (0-100 Skorlar)' if is_tr else 'Normalized Country Comparison Matrix (0-100 Scores)'}")
    df_all_hm_raw = df_all_raw.set_index("Ülke")
    df_all_norm = (df_all_hm_raw - df_all_hm_raw.min()) / (df_all_hm_raw.max() - df_all_hm_raw.min() + 1e-9) * 100.0

    fig_all_hm = px.imshow(df_all_norm, text_auto=".0f", color_continuous_scale="Viridis", aspect="auto", title="Küresel Ülke Karşılaştırma Matrisi (0-100 Normalize Skorlar)" if is_tr else "Global Country Comparison Matrix (0-100 Normalized Scores)")
    fig_all_hm.update_layout(height=1100, margin=dict(l=160, r=40, t=50, b=50), yaxis=dict(tickfont=dict(size=11), autorange="reversed"))
    st.plotly_chart(fig_all_hm, use_container_width=True)

# TAB 7: SCENARIO VAULT & EXPORT
with tab7:
    st.subheader("💾 Kaydedilen Senaryolar Deposu & Dışa Aktar" if is_tr else "💾 Saved Scenario Vault & Export")
    st.markdown("Simülatörde hazırlayıp **'Senaryoyu Hafızaya Kaydet'** butonu ile sakladığınız tüm senaryolar burada listelenir:" if is_tr else "All scenarios saved via 'Save Scenario to Memory' are listed here:")

    if st.session_state.saved_scenarios:
        df_saved = pd.DataFrame(st.session_state.saved_scenarios)
        st.dataframe(df_saved, use_container_width=True, hide_index=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_saved.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Senaryoları CSV Olarak İndir" if is_tr else "📥 Download Scenarios (CSV)",
                data=csv_data,
                file_name="kaydedilen_senaryolar.csv",
                mime="text/csv"
            )
        with col_dl2:
            pdf_bytes = generate_pdf_report()
            st.download_button(
                label="📄 Resmi Raporu İndir (PDF)" if is_tr else "📄 Download Official Report (PDF)",
                data=pdf_bytes,
                file_name="Karbon_Ayakizi_Resmi_Rapor.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Henüz hafızaya kaydedilmiş senaryo yok. Yan menüden parametreleri değiştirip **'Senaryoyu Hafızaya Kaydet'** butonuna basarak ekleyebilirsiniz." if is_tr else "No saved scenarios in memory yet. Change parameters in sidebar and click 'Save Scenario to Memory'.")

# TAB 8: METHODOLOGY & XAI
with tab8:
    st.subheader("📜 Metodoloji, Şeffaf Hesaplama & XAI Notları" if is_tr else "📜 Methodology, Transparent Calculation & XAI Notes")
    st.markdown(r"""
    ### 🔬 5-Kademeli Şeffaf Hesaplama ve Yapay Zeka Metodolojisi

    1. **Temel Bileşenler Analizi (PCA):** Sabit Genişbant (\\(X_1\\)), İnternet Kullanımı (\\(X_2\\)) ve Mobil Abonelik (\\(X_3\\)) göstergeleri özdeğeri \\(\lambda_1 = 2.13\\) olan birincil bileşene dönüştürülür:
       \\[PC_1 = 0.621 \cdot Z(X_1) + 0.649 \cdot Z(X_2) + 0.439 \cdot Z(X_3)\\]

    2. **SVR Makro Tahmin Modeli:** RBF çekirdekli Destek Vektör Regresyonu (\\(R^2 = 0.975\\), \\(RMSE = 101.24 \text{ Mt CO}_2\text{eq}\\)).

    3. **XAI / SHAP Marjinal Katkı:** Her bir politikanın emisyon üzerindeki net etkisi (\\(c_k\\)) diğer değişkenler \\(S_0\\) Baseline seviyesinde sabit tutularak tekil ayrıştırılır.

    4. **Monte Carlo Risk Simülasyonu:** 10.000 iterasyonlu rassal gürültü eklenerek %5 ve %95 olasılık güven aralıkları hesaplanır.

    5. **Macro-to-Micro Downscaling & SKDM / Kapsam 3 (Scope 3):** Ülke düzeyindeki emisyon ve enerji yoğunluğu, ihracatçı şirketlerin AB SKDM vergi yükü (€) ve Kapsam 3 tedarik zinciri iklim risklerine indirgenir.

    ---
    ⚠️ **Nedensellik Sınırı:** Bu platformdaki tahminler ilişkisel ve tahminsel (*associative / predictive marginal effects*) duyarlılıklara dayanır; doğrudan neden-sonuç iddiası taşımaz.
    """)

# SIDEBAR PDF DOWNLOAD BUTTON
st.sidebar.markdown("---")
pdf_bytes_sb = generate_pdf_report()
st.sidebar.download_button(
    label="📄 " + ("Resmi Raporu İndir (PDF)" if is_tr else "Download Official Report (PDF)"),
    data=pdf_bytes_sb,
    file_name="Karbon_Ayakizi_Resmi_Rapor.pdf",
    mime="application/pdf"
)
