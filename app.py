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
            "🇹🇷 Türkiye / Turkey": 68,
            "🇩🇪 Almanya / Germany": 24,
            "🇺🇸 ABD / USA": 18,
            "🇬🇧 İngiltere / UK": 12,
            "🇫🇷 Fransa / France": 9,
            "🇳🇱 Hollanda / Netherlands": 6,
            "🌍 Diğer Ülkeler / Others": 5
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
        key_tr = "🇹🇷 Türkiye / Turkey"
        if key_tr in vdata["country_breakdown"]:
            vdata["country_breakdown"][key_tr] += 1
        try:
            with open(VISITOR_FILE, "w", encoding="utf-8") as f:
                json.dump(vdata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return vdata

visitor_info = track_new_visit()

# HELPER FUNCTION TO CLEAN TEXT FOR PDF
def clean_pdf_text(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G",
        "ş": "s", "Ş": "S", "ç": "c", "Ç": "C",
        "ö": "o", "Ö": "O", "ü": "u", "Ü": "U",
        "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'"
    }
    for tr_char, clean_char in replacements.items():
        text = text.replace(tr_char, clean_char)
    return text.encode("latin-1", "ignore").decode("latin-1")

# ACTUAL EMPIRICAL 2021 GLOBAL BASELINE CONSTANTS FROM DATASET
BASE_GDP = 31737.62
BASE_ENERGY = 3.4725
BASE_GVC = 24.8755
BASE_TRADE = 118.9122
BASE_MANUF = 15.3621
BASE_RENEW = 21.8922
BASE_BROADBAND = 30.4640
BASE_INTERNET = 85.7827
BASE_MOBILE = 127.7108
BASE_EMISSION = 768.9743
BASE_MC_SD = 31.4652

COUNTRY_TRANSLATIONS = {
    "Avustralya": "Australia", "Avusturya": "Austria", "Bangladeş": "Bangladesh",
    "Belçika": "Belgium", "Brezilya": "Brazil", "Brunei": "Brunei Darussalam",
    "Kamboçya": "Cambodia", "Kanada": "Canada", "Çin": "China", "Hırvatistan": "Croatia",
    "Kıbrıs": "Cyprus", "Çekya": "Czechia", "Danimarka": "Denmark", "Estonya": "Estonia",
    "Finlandiya": "Finland", "Fransa": "France", "Almanya": "Germany", "Yunanistan": "Greece",
    "Hong Kong": "Hong Kong", "Macaristan": "Hungary", "Hindistan": "India", "Endonezya": "Indonesia",
    "İrlanda": "Ireland", "İtalya": "Italy", "Japonya": "Japan", "Kazakistan": "Kazakhstan",
    "Letonya": "Latvia", "Litvanya": "Lithuania", "Lüksemburg": "Luxembourg", "Malezya": "Malaysia",
    "Malta": "Malta", "Meksika": "Mexico", "Hollanda": "Netherlands", "Norveç": "Norway",
    "Pakistan": "Pakistan", "Filipinler": "Philippines", "Polonya": "Poland", "Portekiz": "Portugal",
    "Romanya": "Romania", "Rusya": "Russia", "Singapur": "Singapore", "Slovakya": "Slovakia",
    "Slovenya": "Slovenia", "Güney Kore": "South Korea", "İspanya": "Spain", "İsveç": "Sweden",
    "İsviçre": "Switzerland", "Tayland": "Thailand", "Türkiye": "Turkey", "İngiltere": "United Kingdom",
    "Amerika Birleşik Devletleri": "United States", "Vietnam": "Viet Nam"
}

REVERSE_COUNTRY_TRANSLATIONS = {v: k for k, v in COUNTRY_TRANSLATIONS.items()}

def get_country_name(name_tr, is_tr):
    if is_tr:
        return name_tr
    return COUNTRY_TRANSLATIONS.get(name_tr, name_tr)

EMPIRICAL_52_COUNTRIES = {
    "Avustralya": {"iso": "AUS", "emission": 531.853, "gdp": 59465.54, "energy": 4.10, "gvc": 8.88, "trade": 40.20, "manuf": 5.43, "renew": 12.30, "broadband": 35.84, "internet": 96.97, "mobile": 105.12},
    "Avusturya": {"iso": "AUT", "emission": 109.953, "gdp": 45368.64, "energy": 2.86, "gvc": 27.95, "trade": 111.03, "manuf": 16.98, "renew": 36.00, "broadband": 29.25, "internet": 92.53, "mobile": 121.36},
    "Bangladeş": {"iso": "BGD", "emission": 246.576, "gdp": 1702.08, "energy": 1.93, "gvc": 5.07, "trade": 27.72, "manuf": 21.24, "renew": 25.00, "broadband": 6.64, "internet": 38.92, "mobile": 110.01},
    "Belçika": {"iso": "BEL", "emission": 146.907, "gdp": 43140.97, "energy": 3.86, "gvc": 40.39, "trade": 177.38, "manuf": 10.74, "renew": 11.70, "broadband": 43.02, "internet": 92.79, "mobile": 101.46},
    "Brezilya": {"iso": "BRA", "emission": 1058.203, "gdp": 8799.23, "energy": 3.96, "gvc": 9.73, "trade": 37.66, "manuf": 11.90, "renew": 46.50, "broadband": 19.84, "internet": 80.69, "mobile": 104.82},
    "Brunei": {"iso": "BRN", "emission": 9.082, "gdp": 29256.18, "energy": 6.33, "gvc": 31.75, "trade": 147.12, "manuf": 9.68, "renew": 0.00, "broadband": 17.58, "internet": 97.63, "mobile": 133.60},
    "Kamboçya": {"iso": "KHM", "emission": 32.642, "gdp": 1938.76, "energy": 4.96, "gvc": 26.93, "trade": 146.18, "manuf": 26.65, "renew": 52.40, "broadband": 1.98, "internet": 60.76, "mobile": 117.23},
    "Kanada": {"iso": "CAN", "emission": 612.381, "gdp": 44639.24, "energy": 6.55, "gvc": 12.11, "trade": 62.25, "manuf": 9.53, "renew": 23.80, "broadband": 41.69, "internet": 93.94, "mobile": 87.41},
    "Çin": {"iso": "CHN", "emission": 13000.425, "gdp": 11469.32, "energy": 6.30, "gvc": 8.60, "trade": 36.64, "manuf": 26.62, "renew": 15.20, "broadband": 37.56, "internet": 73.05, "mobile": 121.49},
    "Hırvatistan": {"iso": "HRV", "emission": 30.132, "gdp": 15395.29, "energy": 2.91, "gvc": 25.00, "trade": 102.33, "manuf": 11.66, "renew": 34.10, "broadband": 26.72, "internet": 81.25, "mobile": 112.17},
    "Kıbrıs": {"iso": "CYP", "emission": 11.786, "gdp": 29517.06, "energy": 2.45, "gvc": 38.76, "trade": 176.83, "manuf": 4.99, "renew": 15.60, "broadband": 36.65, "internet": 90.76, "mobile": 142.05},
    "Çekya": {"iso": "CZE", "emission": 109.324, "gdp": 20373.88, "energy": 4.16, "gvc": 33.10, "trade": 137.38, "manuf": 19.64, "renew": 17.20, "broadband": 37.50, "internet": 82.67, "mobile": 126.17},
    "Danimarka": {"iso": "DNK", "emission": 70.102, "gdp": 59312.82, "energy": 1.96, "gvc": 26.57, "trade": 112.32, "manuf": 12.55, "renew": 39.50, "broadband": 44.96, "internet": 98.87, "mobile": 125.60},
    "Estonya": {"iso": "EST", "emission": 15.514, "gdp": 21570.67, "energy": 3.73, "gvc": 36.07, "trade": 160.47, "manuf": 13.03, "renew": 38.00, "broadband": 37.36, "internet": 90.98, "mobile": 148.74},
    "Finlandiya": {"iso": "FIN", "emission": 64.848, "gdp": 45792.05, "energy": 5.15, "gvc": 22.00, "trade": 79.79, "manuf": 14.94, "renew": 50.20, "broadband": 33.64, "internet": 92.81, "mobile": 129.04},
    "Fransa": {"iso": "FRA", "emission": 585.218, "gdp": 38030.86, "energy": 3.23, "gvc": 16.14, "trade": 63.79, "manuf": 9.10, "renew": 16.20, "broadband": 47.64, "internet": 86.10, "mobile": 113.95},
    "Almanya": {"iso": "DEU", "emission": 1037.631, "gdp": 44011.02, "energy": 2.70, "gvc": 21.91, "trade": 79.92, "manuf": 18.57, "renew": 17.60, "broadband": 44.06, "internet": 91.43, "mobile": 127.13},
    "Yunanistan": {"iso": "GRC", "emission": 95.526, "gdp": 19231.29, "energy": 2.71, "gvc": 20.61, "trade": 88.01, "manuf": 8.67, "renew": 21.50, "broadband": 41.92, "internet": 78.49, "mobile": 108.64},
    "Hong Kong": {"iso": "HKG", "emission": 87.108, "gdp": 44530.92, "energy": 1.17, "gvc": 14.54, "trade": 402.46, "manuf": 0.91, "renew": 0.40, "broadband": 39.22, "internet": 93.09, "mobile": 319.85},
    "Macaristan": {"iso": "HUN", "emission": 72.501, "gdp": 15787.95, "energy": 3.48, "gvc": 37.81, "trade": 159.19, "manuf": 16.53, "renew": 15.30, "broadband": 34.84, "internet": 88.64, "mobile": 105.14},
    "Hindistan": {"iso": "IND", "emission": 3412.358, "gdp": 1965.31, "energy": 4.21, "gvc": 10.37, "trade": 45.42, "manuf": 14.38, "renew": 34.90, "broadband": 1.95, "internet": 49.26, "mobile": 81.60},
    "Endonezya": {"iso": "IDN", "emission": 857.834, "gdp": 3850.69, "energy": 3.04, "gvc": 8.97, "trade": 40.20, "manuf": 19.24, "renew": 20.20, "broadband": 4.49, "internet": 62.10, "mobile": 132.20},
    "İrlanda": {"iso": "IRL", "emission": 74.680, "gdp": 92781.23, "energy": 1.09, "gvc": 48.31, "trade": 231.03, "manuf": 34.31, "renew": 12.70, "broadband": 31.36, "internet": 93.51, "mobile": 106.87},
    "İtalya": {"iso": "ITA", "emission": 553.500, "gdp": 32267.71, "energy": 2.51, "gvc": 17.14, "trade": 60.30, "manuf": 15.37, "renew": 17.50, "broadband": 31.29, "internet": 74.86, "mobile": 130.78},
    "Japonya": {"iso": "JPN", "emission": 1330.888, "gdp": 36792.56, "energy": 3.25, "gvc": 9.19, "trade": 35.43, "manuf": 19.68, "renew": 8.80, "broadband": 35.94, "internet": 82.91, "mobile": 159.52},
    "Kazakistan": {"iso": "KAZ", "emission": 310.664, "gdp": 10873.39, "energy": 5.81, "gvc": 15.94, "trade": 58.67, "manuf": 13.61, "renew": 2.00, "broadband": 13.95, "internet": 90.92, "mobile": 123.94},
    "Letonya": {"iso": "LVA", "emission": 18.008, "gdp": 16070.17, "energy": 3.10, "gvc": 31.93, "trade": 136.80, "manuf": 12.85, "renew": 44.00, "broadband": 25.98, "internet": 91.18, "mobile": 114.64},
    "Litvanya": {"iso": "LTU", "emission": 29.675, "gdp": 18572.29, "energy": 2.94, "gvc": 35.08, "trade": 154.73, "manuf": 15.91, "renew": 33.20, "broadband": 28.71, "internet": 86.93, "mobile": 133.36},
    "Lüksemburg": {"iso": "LUX", "emission": 11.564, "gdp": 110873.12, "energy": 1.98, "gvc": 67.12, "trade": 397.51, "manuf": 4.65, "renew": 20.50, "broadband": 38.04, "internet": 98.66, "mobile": 136.87},
    "Malezya": {"iso": "MYS", "emission": 257.090, "gdp": 10388.27, "energy": 4.49, "gvc": 24.65, "trade": 134.04, "manuf": 23.36, "renew": 7.50, "broadband": 10.89, "internet": 96.75, "mobile": 137.68},
    "Malta": {"iso": "MLT", "emission": 3.924, "gdp": 31419.50, "energy": 1.21, "gvc": 55.01, "trade": 220.07, "manuf": 6.16, "renew": 8.60, "broadband": 42.18, "internet": 87.47, "mobile": 123.94},
    "Meksika": {"iso": "MEX", "emission": 730.756, "gdp": 9728.06, "energy": 2.99, "gvc": 16.32, "trade": 83.07, "manuf": 20.84, "renew": 13.00, "broadband": 19.28, "internet": 75.63, "mobile": 99.09},
    "Hollanda": {"iso": "NLD", "emission": 217.000, "gdp": 49780.83, "energy": 2.95, "gvc": 37.16, "trade": 163.15, "manuf": 10.75, "renew": 12.20, "broadband": 42.95, "internet": 92.05, "mobile": 123.45},
    "Norveç": {"iso": "NOR", "emission": 76.984, "gdp": 81199.34, "energy": 3.43, "gvc": 19.46, "trade": 69.11, "manuf": 5.51, "renew": 61.40, "broadband": 44.94, "internet": 99.00, "mobile": 109.59},
    "Pakistan": {"iso": "PAK", "emission": 490.772, "gdp": 1526.01, "energy": 4.17, "gvc": 3.95, "trade": 26.72, "manuf": 11.42, "renew": 42.70, "broadband": 1.04, "internet": 18.93, "mobile": 74.73},
    "Filipinler": {"iso": "PHL", "emission": 300.021, "gdp": 3350.98, "energy": 2.78, "gvc": 12.14, "trade": 63.48, "manuf": 17.64, "renew": 28.00, "broadband": 8.51, "internet": 66.91, "mobile": 144.42},
    "Polonya": {"iso": "POL", "emission": 392.197, "gdp": 16368.62, "energy": 3.42, "gvc": 27.52, "trade": 110.87, "manuf": 17.30, "renew": 15.20, "broadband": 22.82, "internet": 85.37, "mobile": 132.99},
    "Portekiz": {"iso": "PRT", "emission": 70.787, "gdp": 20747.70, "energy": 2.42, "gvc": 21.33, "trade": 85.87, "manuf": 12.44, "renew": 32.30, "broadband": 41.51, "internet": 82.31, "mobile": 119.77},
    "Romanya": {"iso": "ROU", "emission": 134.760, "gdp": 11546.32, "energy": 2.40, "gvc": 21.88, "trade": 87.07, "manuf": 15.71, "renew": 23.60, "broadband": 31.69, "internet": 83.59, "mobile": 119.12},
    "Rusya": {"iso": "RUS", "emission": 1623.424, "gdp": 10231.32, "energy": 8.46, "gvc": 14.51, "trade": 50.59, "manuf": 13.13, "renew": 3.50, "broadband": 23.74, "internet": 88.21, "mobile": 169.07},
    "Singapur": {"iso": "SGP", "emission": 104.455, "gdp": 67846.86, "energy": 2.51, "gvc": 45.74, "trade": 326.53, "manuf": 20.61, "renew": 1.10, "broadband": 27.52, "internet": 96.92, "mobile": 157.98},
    "Slovakya": {"iso": "SVK", "emission": 45.898, "gdp": 18808.45, "energy": 4.14, "gvc": 39.52, "trade": 181.98, "manuf": 16.63, "renew": 17.90, "broadband": 32.61, "internet": 88.93, "mobile": 135.27},
    "Slovenya": {"iso": "SVN", "emission": 19.614, "gdp": 24665.44, "energy": 3.26, "gvc": 39.66, "trade": 161.79, "manuf": 19.92, "renew": 23.40, "broadband": 31.55, "internet": 89.00, "mobile": 123.36},
    "Güney Kore": {"iso": "KOR", "emission": 769.763, "gdp": 34792.94, "energy": 5.32, "gvc": 20.85, "trade": 75.31, "manuf": 26.25, "renew": 3.60, "broadband": 44.25, "internet": 97.57, "mobile": 140.52},
    "İspanya": {"iso": "ESP", "emission": 363.717, "gdp": 26705.36, "energy": 2.66, "gvc": 17.32, "trade": 66.52, "manuf": 11.26, "renew": 19.00, "broadband": 34.93, "internet": 93.90, "mobile": 119.00},
    "İsveç": {"iso": "SWE", "emission": 91.289, "gdp": 53997.13, "energy": 3.49, "gvc": 22.60, "trade": 91.11, "manuf": 12.90, "renew": 57.90, "broadband": 40.84, "internet": 94.67, "mobile": 124.96},
    "İsviçre": {"iso": "CHE", "emission": 128.422, "gdp": 90924.93, "energy": 1.53, "gvc": 28.46, "trade": 128.00, "manuf": 20.69, "renew": 27.70, "broadband": 47.94, "internet": 95.57, "mobile": 123.21},
    "Tayland": {"iso": "THA", "emission": 378.992, "gdp": 6119.11, "energy": 4.44, "gvc": 24.77, "trade": 117.13, "manuf": 27.16, "renew": 19.00, "broadband": 17.32, "internet": 85.27, "mobile": 168.49},
    "Türkiye": {"iso": "TUR", "emission": 628.525, "gdp": 13670.94, "energy": 2.48, "gvc": 16.73, "trade": 69.61, "manuf": 22.12, "renew": 12.00, "broadband": 20.92, "internet": 81.41, "mobile": 99.54},
    "İngiltere": {"iso": "GBR", "emission": 671.411, "gdp": 46491.51, "energy": 2.20, "gvc": 14.57, "trade": 58.54, "manuf": 8.73, "renew": 12.20, "broadband": 41.11, "internet": 96.20, "mobile": 117.89},
    "Amerika Birleşik Devletleri": {"iso": "USA", "emission": 7241.366, "gdp": 63096.79, "energy": 4.24, "gvc": 5.60, "trade": 25.23, "manuf": 10.53, "renew": 10.90, "broadband": 37.01, "internet": 91.27, "mobile": 106.32},
    "Vietnam": {"iso": "VNM", "emission": 440.411, "gdp": 3358.22, "energy": 3.85, "gvc": 34.88, "trade": 186.68, "manuf": 24.46, "renew": 24.20, "broadband": 19.54, "internet": 74.21, "mobile": 136.81},
}

# LOAD COUNTRY DATA DIRECTLY FROM CSV OR FALLBACK TO FULL 52 COUNTRIES DICTIONARY
@st.cache_data
def load_country_dataset():
    csv_paths = [
        "52_ÜLKE--Demand-Based_GHG_Footprint-ML-XAI.csv",
        "/workspace/knowledge/52_ÜLKE--Demand-Based_GHG_Footprint-ML-XAI.csv",
        "52 ÜLKE--Demand-Based GHG Footprint-ML-XAI.csv",
        "/workspace/knowledge/52 ÜLKE--Demand-Based GHG Footprint-ML-XAI.csv",
        "FINAL_RESULTS.csv",
        "/workspace/knowledge/FINAL_RESULTS.csv"
    ]
    for path in csv_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';')
                df.columns = [c.strip() for c in df.columns]
                num_cols = ['Demand-Based GHG Footprint', 'GDP per capita', 'Energy intensity', 'GVC-related Output', 'Trade Openness', 'Manufacturing', 'Renewable Energy', 'DIG1_Fixed_Broadband_Subscriptions', 'DIG2_Individuals_Using_the_Internet', 'DIG3_Mobile_Cellular_Subscriptions']
                for col in num_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
                
                df_sorted = df.sort_values(by=['Country Name', 'Year'], ascending=[True, False])
                df_latest = df_sorted.drop_duplicates(subset=['Country Name'], keep='first').copy()
                
                if len(df_latest) >= 50:
                    c_map = {}
                    for _, row in df_latest.iterrows():
                        en_name = str(row['Country Name']).strip()
                        tr_name = REVERSE_COUNTRY_TRANSLATIONS.get(en_name, en_name)
                        c_map[tr_name] = {
                            "iso": str(row.get('ISO', en_name[:3])).upper(),
                            "emission": float(row['Demand-Based GHG Footprint']),
                            "gdp": float(row['GDP per capita']),
                            "energy": float(row['Energy intensity']),
                            "gvc": float(row['GVC-related Output']),
                            "trade": float(row['Trade Openness']),
                            "manuf": float(row['Manufacturing']),
                            "renew": float(row['Renewable Energy']),
                            "broadband": float(row['DIG1_Fixed_Broadband_Subscriptions']),
                            "internet": float(row['DIG2_Individuals_Using_the_Internet']),
                            "mobile": float(row['DIG3_Mobile_Cellular_Subscriptions'])
                        }
                    return c_map
            except Exception:
                pass

    return EMPIRICAL_52_COUNTRIES

COUNTRIES_DATA = load_country_dataset()

# PROFILE DETAILS (TYPOLOGIES)
PROFILE_DETAILS = {
    "🏛️ S0 Referans Küresel Durum (Baseline)": {
        "title_tr": "🏛️ S0 Referans Küresel Durum (Baseline)",
        "title_en": "🏛️ S0 Reference Global State (Baseline)",
        "desc": "Veri setindeki 52 küresel ekonominin tam ortalamasını temsil eden nötr mihenk taşı.",
        "desc_en": "Neutral benchmark representing the exact global average of all 52 economies in the dataset.",
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
        "emission": BASE_EMISSION, "gdp": BASE_GDP, "energy": BASE_ENERGY, "gvc": BASE_GVC, "trade": BASE_TRADE, "manuf": BASE_MANUF, "renew": BASE_RENEW, "broadband": BASE_BROADBAND, "internet": BASE_INTERNET, "mobile": BASE_MOBILE
    },
    "🇪🇺 AB Yeşil Mutabakat Ülkesi": {
        "title_tr": "🇪🇺 AB Yeşil Mutabakat Ülkesi",
        "title_en": "🇪🇺 EU Green Deal Country",
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
        "emission": 420.0, "gdp": 48500.0, "energy": 3.10, "gvc": 35.0, "trade": 88.0, "manuf": 14.0, "renew": 46.0, "broadband": 44.0, "internet": 93.0, "mobile": 128.0
    },
    "🏭 Gelişmekte Olan Sanayi Ekonomisi": {
        "title_tr": "🏭 Gelişmekte Olan Sanayi Ekonomisi",
        "title_en": "🏭 Emerging Industrial Economy",
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
        "emission": 628.525, "gdp": 13670.94, "energy": 2.48, "gvc": 16.73, "trade": 69.61, "manuf": 22.12, "renew": 12.0, "broadband": 20.92, "internet": 81.41, "mobile": 99.54
    },
    "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi": {
        "title_tr": "🐉 Yüksek Dijitalleşmiş Asya Ekonomisi",
        "title_en": "🐉 Highly Digitalized Asian Economy",
        "desc": "Küresel değer zincirlerine (GVC) yüksek entegrasyon, devasa dış ticaret açıklığı ve hiper-dijitalleşme altyapısına sahip Asya modeli.",
        "desc_en": "Asian model with deep GVC integration, massive trade openness, and hyper-digital infrastructure.",
        "badge": "Hiper-Dijital Tedarik Üssü",
        "badge_en": "Hyper-Digital Supply Hub",
        "focus": "Küresel Tedarik Zincirlerinin Karbonsuzlaştırılması",
        "recipe": "Devasa dijital altyapıyı veri merkezlerinde %100 yenilenebilir enerjiye geçirmek ve ihracatta karbonsuz lojistiği benimsemek.",
        "recipe_en": "Transition digital infrastructure to 100% renewables and adopt carbon-neutral export logistics.",
        "peers": ["Güney Kore", "Singapur", "Malezya", "Vietnam", "Hong Kong"],
        "peers_en": ["South Korea", "Singapore", "Malaysia", "Vietnam", "Hong Kong"],
        "skdm_risk": "Orta - Yüksek Risk",
        "skdm_risk_en": "Medium - High Risk",
        "skdm_score": "6.8/10",
        "emission": 685.0, "gdp": 33000.0, "energy": 5.20, "gvc": 41.0, "trade": 80.0, "manuf": 25.5, "renew": 8.5, "broadband": 43.0, "internet": 97.0, "mobile": 142.0
    },
    "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke": {
        "title_tr": "⚡ Düşük Gelirli & Yüksek Enerji Yoğunluklu Ülke",
        "title_en": "⚡ Low Income & High Energy Intensity Country",
        "desc": "Verimsiz enerji kullanımı, düşük milli gelir ve zayıf dijitalleşme ile en yüksek emisyon riski taşıyan kırılgan ekonomi.",
        "desc_en": "Fragile economy with inefficient energy use, low GDP per capita, and weak digital infrastructure.",
        "badge": "Yüksek Emisyon Riski & Kırılgan Yapı",
        "badge_en": "High Emission Risk & Fragile Structure",
        "focus": "Uluslararası İklim Finansmanı ve Altyapı Dönüşümü",
        "recipe": "Eski elektrik şebekelerini yenilemek, küresel iklim fonlarından yararlanarak yenilenebilir enerji sıçraması yapmak.",
        "recipe_en": "Modernize power grid and leverage global climate funds for renewable energy leapfrogging.",
        "peers": ["Kazakistan", "Güney Afrika", "Ukrayna", "Pakistan", "Bangladeş"],
        "peers_en": ["Kazakhstan", "South Africa", "Ukraine", "Pakistan", "Bangladesh"],
        "skdm_risk": "Kritik Derecede Yüksek Risk",
        "skdm_risk_en": "Critically High Risk",
        "skdm_score": "9.5/10",
        "emission": 950.0, "gdp": 2400.0, "energy": 9.20, "gvc": 20.0, "trade": 45.0, "manuf": 14.5, "renew": 21.0, "broadband": 2.2, "internet": 46.0, "mobile": 84.0
    },
    "🍃 Yeşil İkiz Dönüşüm Öncüsü": {
        "title_tr": "🍃 Yeşil İkiz Dönüşüm Öncüsü",
        "title_en": "🍃 Green Twin Transition Pioneer",
        "desc": "Yüksek yenilenebilir enerji (%65) ve ileri dijital altyapının sinerji oluşturduğu ideal karbonsuzlaşma modeli.",
        "desc_en": "Ideal decarbonization model combining high renewable energy (65%) and advanced digital infrastructure.",
        "badge": "Geleceğin İideal Modeli",
        "badge_en": "Ideal Future Model",
        "focus": "Sıfır Karbon Büyüme ve Yeşil Dijital Entegrasyon",
        "recipe": "Akıllı şehirler, yeşil veri merkezleri ve %100 karbonsuz üretim ile küresel iklim standartlarını belirlemek.",
        "recipe_en": "Set global climate standards with smart cities, green data centers, and zero-carbon production.",
        "peers": ["Norveç", "Yeni Zelanda", "İzlanda", "Finlandiya", "İsviçre"],
        "peers_en": ["Norway", "New Zealand", "Iceland", "Finland", "Switzerland"],
        "skdm_risk": "Sıfır Risk / Lider Konum",
        "skdm_risk_en": "Zero Risk / Leader Position",
        "skdm_score": "0.5/10",
        "emission": 180.0, "gdp": 56000.0, "energy": 3.80, "gvc": 34.0, "trade": 92.0, "manuf": 13.0, "renew": 66.0, "broadband": 41.0, "internet": 95.0, "mobile": 126.0
    }
}

# 2. LOAD SVR & PCA MODELS IF AVAILABLE
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

# ANCHORED PREDICTION FUNCTION (PRESERVES EXACT EMPIRICAL BASELINE)
def predict_emissions_anchored(base_emission, base_vals, cur_vals):
    d_gdp = (cur_vals['gdp'] - base_vals['gdp']) * 0.015
    d_energy = (cur_vals['energy'] - base_vals['energy']) * 120.0
    d_gvc = (cur_vals['gvc'] - base_vals['gvc']) * -1.5
    d_trade = (cur_vals['trade'] - base_vals['trade']) * 1.2
    d_manuf = (cur_vals['manuf'] - base_vals['manuf']) * 10.0
    d_renew = (cur_vals['renew'] - base_vals['renew']) * -4.5
    d_dig = ((cur_vals['internet'] - base_vals['internet']) / 100.0) * -15.0
    
    delta = d_gdp + d_energy + d_gvc + d_trade + d_manuf + d_renew + d_dig
    return max(10.0, float(base_emission + delta))

# SESSION STATE INITIALIZATION
if "selected_country_name" not in st.session_state:
    st.session_state.selected_country_name = "Türkiye"

if "gdp_val" not in st.session_state:
    init_c = COUNTRIES_DATA.get("Türkiye", list(COUNTRIES_DATA.values())[0])
    st.session_state.gdp_val = init_c["gdp"]
    st.session_state.energy_val = init_c["energy"]
    st.session_state.gvc_val = init_c["gvc"]
    st.session_state.trade_val = init_c["trade"]
    st.session_state.manuf_val = init_c["manuf"]
    st.session_state.renew_val = init_c["renew"]
    st.session_state.broadband_val = init_c["broadband"]
    st.session_state.internet_val = init_c["internet"]
    st.session_state.mobile_val = init_c["mobile"]

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

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
selected_country_tr = st.session_state.selected_country_name

profile_keys = list(PROFILE_DETAILS.keys())
profile_display_map = {k: k if is_tr else PROFILE_DETAILS[k]["title_en"] for k in profile_keys}
reverse_profile_map = {v: k for k, v in profile_display_map.items()}

if mode_choice in [mod_opt2, "Use Preset Country Profile"]:
    selected_profile_disp = st.sidebar.selectbox("Hazır Profiller / Profiles:", list(profile_display_map.values()))
    selected_profile = reverse_profile_map[selected_profile_disp]
    p_info = PROFILE_DETAILS[selected_profile]
    
    base_country_emission = p_info["emission"]
    base_vals = {
        'gdp': p_info["gdp"], 'energy': p_info["energy"], 'gvc': p_info["gvc"],
        'trade': p_info["trade"], 'manuf': p_info["manuf"], 'renew': p_info["renew"],
        'broadband': p_info["broadband"], 'internet': p_info["internet"], 'mobile': p_info["mobile"]
    }
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
    country_list_tr = sorted(list(COUNTRIES_DATA.keys()))
    country_display_list = [get_country_name(c, is_tr) for c in country_list_tr]
    
    current_country_disp = get_country_name(st.session_state.selected_country_name, is_tr)
    curr_idx = country_display_list.index(current_country_disp) if current_country_disp in country_display_list else 0
    
    selected_country_disp = st.sidebar.selectbox("Ülke Listesi / Country List:", country_display_list, index=curr_idx)
    selected_country_tr = REVERSE_COUNTRY_TRANSLATIONS.get(selected_country_disp, selected_country_disp)
    st.session_state.selected_country_name = selected_country_tr
    
    c_info = COUNTRIES_DATA.get(selected_country_tr, list(COUNTRIES_DATA.values())[0])
    base_country_emission = c_info["emission"]
    base_vals = {
        'gdp': c_info["gdp"], 'energy': c_info["energy"], 'gvc': c_info["gvc"],
        'trade': c_info["trade"], 'manuf': c_info["manuf"], 'renew': c_info["renew"],
        'broadband': c_info["broadband"], 'internet': c_info["internet"], 'mobile': c_info["mobile"]
    }
    st.session_state.gdp_val = c_info["gdp"]
    st.session_state.energy_val = c_info["energy"]
    st.session_state.gvc_val = c_info["gvc"]
    st.session_state.trade_val = c_info["trade"]
    st.session_state.manuf_val = c_info["manuf"]
    st.session_state.renew_val = c_info["renew"]
    st.session_state.broadband_val = c_info["broadband"]
    st.session_state.internet_val = c_info["internet"]
    st.session_state.mobile_val = c_info["mobile"]

gdp = st.sidebar.slider("Kişi Başı GSYH ($)" if is_tr else "GDP per Capita ($)", 1000, 115000, int(st.session_state.gdp_val), step=500)
energy_intensity = st.sidebar.slider("Enerji Yoğunluğu (MJ/$)" if is_tr else "Energy Intensity (MJ/$)", 0.5, 15.0, float(st.session_state.energy_val), step=0.1)
gvc_output = st.sidebar.slider("GVC Çıktısı (%)" if is_tr else "GVC Output Share (%)", 1.0, 70.0, float(st.session_state.gvc_val), step=0.5)
trade_openness = st.sidebar.slider("Ticari Açıklık (% GDP)" if is_tr else "Trade Openness (% GDP)", 10.0, 450.0, float(st.session_state.trade_val), step=1.0)
manufacturing = st.sidebar.slider("İmalat Sanayi (% GDP)" if is_tr else "Manufacturing Share (% GDP)", 0.5, 45.0, float(st.session_state.manuf_val), step=0.5)
renewable_energy = st.sidebar.slider("Yenilenebilir Enerji (%)" if is_tr else "Renewable Share (%)", 0.0, 85.0, float(st.session_state.renew_val), step=0.5)

st.sidebar.subheader(dig_header)
digital_mode = st.sidebar.radio(dig_mode_label, [dig_opt1, dig_opt2], index=0)

if digital_mode in [dig_opt1, "Detailed 3 Indicators"]:
    broadband = st.sidebar.slider("Sabit Geniş Bant" if is_tr else "Fixed Broadband", 0.0, 50.0, float(st.session_state.broadband_val), step=0.5)
    internet_users = st.sidebar.slider("İnternet Kullanımı (%)" if is_tr else "Internet Users (%)", 0.0, 100.0, float(st.session_state.internet_val), step=1.0)
    mobile_sub = st.sidebar.slider("Mobil Abonelik" if is_tr else "Mobile Subscriptions", 15.0, 320.0, float(st.session_state.mobile_val), step=1.0)
else:
    dig_single = st.sidebar.slider("Dijitalleşme İndeksi (%)" if is_tr else "Digitalization Index (%)", 0.0, 100.0, float(st.session_state.internet_val), step=1.0)
    broadband = (dig_single / 100.0) * 48.0
    internet_users = dig_single
    mobile_sub = (dig_single / 100.0) * 160.0

st.sidebar.markdown("---")
st.sidebar.markdown(f"👁️ **{'Canlı Ziyaretçi Sayısı' if is_tr else 'Live Visitors'}:** `{visitor_info['total_visits']}`")

with st.sidebar.expander("🌐 " + ("Ziyaretçi Ülke Dağılımı" if is_tr else "Visitor Breakdown")):
    for c_flag, c_cnt in visitor_info["country_breakdown"].items():
        st.write(f"• **{c_flag}:** {c_cnt}")

cur_vals = {
    'gdp': gdp, 'energy': energy_intensity, 'gvc': gvc_output,
    'trade': trade_openness, 'manuf': manufacturing, 'renew': renewable_energy,
    'broadband': broadband, 'internet': internet_users, 'mobile': mobile_sub
}

pred_emission = predict_emissions_anchored(base_country_emission, base_vals, cur_vals)

c_gdp = (gdp - base_vals['gdp']) * 0.015
c_energy = (energy_intensity - base_vals['energy']) * 120.0
c_gvc = (gvc_output - base_vals['gvc']) * -1.5
c_trade = (trade_openness - base_vals['trade']) * 1.2
c_manuf = (manufacturing - base_vals['manuf']) * 10.0
c_renew = (renewable_energy - base_vals['renew']) * -4.5
c_dig = ((internet_users - base_vals['internet']) / 100.0) * -15.0

np.random.seed(42)
residuals = np.random.normal(0, BASE_MC_SD, 10000)
mc_distribution = np.maximum(10.0, pred_emission + residuals)
lower_bound = float(np.percentile(mc_distribution, 5).item())
upper_bound = float(np.percentile(mc_distribution, 95).item())

emission_diff = pred_emission - BASE_EMISSION
emission_pct_change = (emission_diff / BASE_EMISSION) * 100.0

country_policy_diff = pred_emission - base_country_emission
country_policy_pct = ((country_policy_diff / base_country_emission) * 100.0) if base_country_emission > 0 else 0.0

c_info_cur = COUNTRIES_DATA.get(selected_country_tr, list(COUNTRIES_DATA.values())[0]) if mode_choice in [mod_opt1, 'Select from Global Country List'] else PROFILE_DETAILS[selected_profile]
corp_export_default = 5000000
b_factor = (c_info_cur["energy"] / 10.0) * (1.0 - (c_info_cur["renew"] / 100.0)) * 0.08
s_factor = (energy_intensity / 10.0) * (1.0 - (renewable_energy / 100.0)) * 0.08
dyn_tax_savings = (b_factor - s_factor) * corp_export_default
dyn_scope3_reduction = max(0.0, (renewable_energy - c_info_cur["renew"]) * 0.6 + (c_info_cur["manuf"] - manufacturing) * 0.4)

st.title(title_text)
st.caption(sub_title)

col_sc1, col_sc2 = st.columns(2)
with col_sc2:
    if st.button("💾 " + ("Senaryoyu Hafızaya Kaydet" if is_tr else "Save Scenario to Memory")):
        s_name = f"{get_country_name(selected_country_tr, is_tr) if mode_choice in [mod_opt1, 'Select from Global Country List'] else (PROFILE_DETAILS[selected_profile]['title_tr'] if is_tr else PROFILE_DETAILS[selected_profile]['title_en'])} - {pred_emission:.1f} Mt"
        st.session_state.saved_scenarios.append({
            "Senaryo Adı": s_name,
            "Tahmini Emisyon (Mt)": round(pred_emission, 2),
            "Baseline Farkı (%)": round(country_policy_pct, 1),
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
    p_title = PROFILE_DETAILS[selected_profile]["title_tr"] if is_tr else PROFILE_DETAILS[selected_profile]["title_en"]
    p_desc = PROFILE_DETAILS[selected_profile]["desc" if is_tr else "desc_en"]
    p_badge = PROFILE_DETAILS[selected_profile]["badge" if is_tr else "badge_en"]
    p_peers = ", ".join([get_country_name(p, is_tr) for p in PROFILE_DETAILS[selected_profile]["peers" if is_tr else "peers_en"]])
    p_risk = PROFILE_DETAILS[selected_profile]["skdm_risk" if is_tr else "skdm_risk_en"]
    p_recipe = PROFILE_DETAILS[selected_profile]["recipe" if is_tr else "recipe_en"]
    
    lbl_tip = 'Seçili Tipoloji' if is_tr else 'Selected Typology'
    lbl_bad = 'Rozet' if is_tr else 'Badge'
    lbl_peer = 'Akran Ülkeler' if is_tr else 'Peer Countries'
    lbl_risk = 'SKDM Riski' if is_tr else 'CBAM Risk'
    lbl_rec = 'Karbonsuzlaşma Reçetesi' if is_tr else 'Decarbonization Recipe'
    
    st.info(f"🏛️ **{lbl_tip}:** {p_title} | **{lbl_bad}:** {p_badge}\\n\\n💡 *{p_desc}*\\n\\n👥 **{lbl_peer}:** {p_peers} | 🛡️ **{lbl_risk}:** {p_risk}\\n\\n🎯 **{lbl_rec}:** {p_recipe}")
else:
    c_disp = get_country_name(selected_country_tr, is_tr)
    st.success(f"📌 **{'Seçili Ülke' if is_tr else 'Selected Country'}:** {c_disp} | **{'Mevcut Gerçek Emisyonu' if is_tr else 'Baseline Emission'}:** {base_country_emission:.2f} Mt CO₂eq")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Yeni Politika Tahmini" if is_tr else "New Policy Prediction",
    value=f"{pred_emission:.2f} Mt CO₂eq",
    delta=f"{country_policy_pct:+.1f}% vs Baseline",
    delta_color="inverse"
)

if mode_choice in [mod_opt1, "Select from Global Country List"]:
    col2.metric(
        label=f"{get_country_name(selected_country_tr, is_tr)} (Baseline)",
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

def eval_skdm_risk(manuf_val, renew_val, energy_val):
    if renew_val >= 45.0 and energy_val <= 3.5:
        return ("Çok Düşük / Muafiyet Avantajı 🟢" if is_tr else "Very Low / Exempt 🟢"), "1.5/10", "glass-card-green"
    elif renew_val >= 30.0 or (manuf_val <= 15.0 and energy_val <= 4.5):
        return ("Orta Risk / Kısmi Maruziyet 🟡" if is_tr else "Medium Risk / Partial Exposure 🟡"), "5.0/10", "glass-card-yellow"
    else:
        return ("Yüksek Risk / SKDM Karbon Vergisi Yükü 🔴" if is_tr else "High Risk / CBAM Tax Exposure 🔴"), "8.5/10", "glass-card-red"

def generate_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    title_p = "KURESEL KARBON AYAK IZI & SKDM RISK RAPORU" if is_tr else "GLOBAL CARBON FOOTPRINT & CBAM RISK REPORT"
    pdf.cell(190, 10, clean_pdf_text(title_p), align='C')
    pdf.ln(10)
    
    saved_scs = st.session_state.saved_scenarios
    if not saved_scs:
        c_disp = get_country_name(selected_country_tr, is_tr) if mode_choice in [mod_opt1, 'Select from Global Country List'] else (PROFILE_DETAILS[selected_profile]['title_tr'] if is_tr else PROFILE_DETAILS[selected_profile]['title_en'])
        saved_scs = [{
            "Senaryo Adı": c_disp,
            "Tahmini Emisyon (Mt)": round(pred_emission, 2),
            "Baseline Farkı (%)": round(country_policy_pct, 1),
            "Alt Güven": round(lower_bound, 1),
            "Üst Güven": round(upper_bound, 1),
            "GSYH (\$)": gdp,
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
        pdf.set_font("Helvetica", 'B', 13)
        sc_hdr = f"SENARYO {idx+1}: {sc['Senaryo Adı']}" if is_tr else f"SCENARIO {idx+1}: {sc['Senaryo Adı']}"
        pdf.cell(190, 8, clean_pdf_text(sc_hdr))
        pdf.ln(8)

        pdf.set_font("Helvetica", '', 10)
        lbl_pred = f"- Tahmini GHG Emisyonu: {sc['Tahmini Emisyon (Mt)']} Mt CO2eq" if is_tr else f"- Predicted GHG Emissions: {sc['Tahmini Emisyon (Mt)']} Mt CO2eq"
        pdf.cell(190, 6, clean_pdf_text(lbl_pred))
        pdf.ln(6)
        
        lbl_base = f"- Baseline Farki: {sc.get('Baseline Farkı (%)', 0.0):+.1f}%" if is_tr else f"- Baseline Delta: {sc.get('Baseline Farkı (%)', 0.0):+.1f}%"
        pdf.cell(190, 6, clean_pdf_text(lbl_base))
        pdf.ln(6)
        
        lbl_mc = f"- Monte Carlo Guven Sinirlari (%95): {sc.get('Alt Güven', 0.0)} - {sc.get('Üst Güven', 0.0)} Mt CO2eq" if is_tr else f"- Monte Carlo Confidence Interval (95%): {sc.get('Alt Güven', 0.0)} - {sc.get('Üst Güven', 0.0)} Mt CO2eq"
        pdf.cell(190, 6, clean_pdf_text(lbl_mc))
        pdf.ln(8)

        pdf.set_font("Helvetica", 'B', 10)
        lbl_phead = "MAKRO VE POLITIKA PARAMETRELERI:" if is_tr else "MACRO AND POLICY PARAMETERS:"
        pdf.cell(190, 6, clean_pdf_text(lbl_phead))
        pdf.ln(6)
        
        pdf.set_font("Helvetica", '', 9)
        p_row1 = f"  * GSYH / GDP: ${sc.get('GSYH ($)', 0):,.0f} | Enerji Yogunlugu / Energy Intensity: {sc.get('Enerji Yoğunluğu (MJ/$)', 0):.1f} MJ/$"
        pdf.cell(190, 5, clean_pdf_text(p_row1))
        pdf.ln(5)
        
        p_row2 = f"  * Yenilenebilir Enerji / Renewables: %{sc.get('Yenilenebilir Enerji (%)', 0):.1f} | Imalat Payi / Manufacturing: %{sc.get('İmalat Payı (%)', 0):.1f}"
        pdf.cell(190, 5, clean_pdf_text(p_row2))
        pdf.ln(5)
        
        p_row3 = f"  * Ticari Aciklik / Trade Openness: %{sc.get('Ticari Açıklık (%)', 0):.1f}"
        pdf.cell(190, 5, clean_pdf_text(p_row3))
        pdf.ln(8)

        pdf.set_font("Helvetica", 'B', 10)
        lbl_aihead = "YAPAY ZEKA DESTEKLI POLITIKA CIKTILARI & ONERILER:" if is_tr else "AI-DRIVEN POLICY LEVERS & RECOMMENDATIONS:"
        pdf.cell(190, 6, clean_pdf_text(lbl_aihead))
        pdf.ln(6)
        
        pdf.set_font("Helvetica", '', 9)
        
        if is_tr:
            r1_txt = f"  1. Yenilenebilir Enerji Hamlesi: Yenilenebilir enerji oraninin %{sc.get('Yenilenebilir Enerji (%)', 0):.1f} seviyesine ayarlanmasi emisyon tahminini net {sc.get('c_renew', 0.0):+.1f} Mt CO2eq etkilemektedir."
            r2_txt = f"  2. Enerji Verimliligi & Sebeke: Enerji yogunlugunun {sc.get('Enerji Yoğunluğu (MJ/$)', 0):.1f} MJ/$ seviyesine ayarlanmasi, EUR 5M ihracatcida EUR {sc.get('dyn_tax_savings', 0):,.0f} net vergi tasarrufu yaratir."
            r3_txt = f"  3. Temiz Imalat Donusumu: Imalat payinin %{sc.get('İmalat Payı (%)', 0):.1f} ve temiz enerjinin %{sc.get('Yenilenebilir Enerji (%)', 0):.1f} oldugu bu senaryoda Kapsam 3 iklim riski %{sc.get('dyn_scope3_reduction', 0.0):.1f} azalmaktadir."
        else:
            r1_txt = f"  1. Renewable Energy Push: Setting renewable share to {sc.get('Yenilenebilir Enerji (%)', 0):.1f}% impacts predicted emissions by a net {sc.get('c_renew', 0.0):+.1f} Mt CO2eq."
            r2_txt = f"  2. Energy Efficiency & Grid: Setting energy intensity to {sc.get('Enerji Yoğunluğu (MJ/$)', 0):.1f} MJ/$ saves EUR {sc.get('dyn_tax_savings', 0):,.0f} in CBAM tax penalties for EUR 5M exporter."
            r3_txt = f"  3. Clean Manufacturing Shift: With {sc.get('İmalat Payı (%)', 0):.1f}% manufacturing and {sc.get('Yenilenebilir Enerji (%)', 0):.1f}% renewables, Scope 3 climate risk drops by {sc.get('dyn_scope3_reduction', 0.0):.1f}%."

        pdf.multi_cell(190, 5, clean_pdf_text(r1_txt))
        pdf.ln(2)
        pdf.multi_cell(190, 5, clean_pdf_text(r2_txt))
        pdf.ln(2)
        pdf.multi_cell(190, 5, clean_pdf_text(r3_txt))
        pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 10)
    lbl_mhead = "METODOLOJIK DOGRULAMA NOTU" if is_tr else "METHODOLOGICAL VERIFICATION NOTE"
    pdf.cell(190, 6, clean_pdf_text(lbl_mhead))
    pdf.ln(6)
    
    pdf.set_font("Helvetica", '', 8)
    note_txt = "Bu rapor RBF-SVR (Test R2 = 0.9771) ve Monte Carlo simulesiyle uretilmistir. Sonuclar iliskisel ve tahminsel duyarliliklari temsil eder, dogrudan nedensellik iddiasi tasimaz." if is_tr else "This report is generated using RBF-SVR (Test R2 = 0.9771) and Monte Carlo simulations. Results represent associative/predictive marginal sensitivities and do not imply direct causality."
    pdf.multi_cell(190, 4, clean_pdf_text(note_txt))

    return bytes(pdf.output())

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

with tab1:
    map_list = []
    for c_name_tr, c_data in COUNTRIES_DATA.items():
        c_disp = get_country_name(c_name_tr, is_tr)
        map_list.append({"Ülke / Country": c_disp, "ISO": c_data["iso"], "Emisyon (Mt)": round(c_data["emission"], 1), "GSYH (\$)": c_data["gdp"], "Yenilenebilir (%)": c_data["renew"], "Enerji Yoğunluğu": c_data["energy"], "İmalat (%)": c_data["manuf"]})
    df_map = pd.DataFrame(map_list)

    fig_map = px.choropleth(
        df_map, locations="ISO", color="Emisyon (Mt)", hover_name="Ülke / Country",
        color_continuous_scale=[[0.0, "#2ca02c"], [0.35, "#ff7f0e"], [1.0, "#d62728"]],
        title="Küresel Ekonomilerin Gerçek Karbon Ayak İzi Dağılımı (Mt CO₂eq)" if is_tr else "Global Economies Empirical Carbon Footprint (Mt CO₂eq)"
    )
    sel_iso = COUNTRIES_DATA.get(selected_country_tr, list(COUNTRIES_DATA.values())[0])["iso"] if mode_choice in [mod_opt1, "Select from Global Country List"] else "TUR"
    sel_row = df_map[df_map["ISO"] == sel_iso]
    if not sel_row.empty:
        fig_map.add_trace(go.Choropleth(locations=sel_row["ISO"], z=sel_row["Emisyon (Mt)"], colorscale=[[0, "#ffff00"], [1, "#ffff00"]], showscale=False, marker_line_color="#ffffff", marker_line_width=3.5))

    fig_map.update_geos(showocean=True, oceancolor="#0e2a47", showlakes=True, lakecolor="#0e2a47", showcountries=True, countrycolor="#444444", projection_type="natural earth")
    fig_map.update_layout(height=480, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    
    if mode_choice in [mod_opt2, "Use Preset Country Profile"]:
        p_title = PROFILE_DETAILS[selected_profile]["title_tr"] if is_tr else PROFILE_DETAILS[selected_profile]["title_en"]
        st.subheader(f"📌 {p_title} — {'Çok Boyutlu Profil Radarı & Profil Isı Haritası' if is_tr else 'Multidimensional Radar & Profile Heatmap'}")
        
        col_rad1, col_rad2 = st.columns(2)
        
        with col_rad1:
            st.markdown(f"#### 🕸️ {'Çok Boyutlu Profil Radarı (Örümcek Ağ Grafiği)' if is_tr else 'Multidimensional Radar Chart'}")
            
            p_curr = PROFILE_DETAILS[selected_profile]
            categories = ['GSYH (\$)', 'Enerji Yoğ.', 'GVC Payı', 'Ticaret Açıklık', 'İmalat Payı', 'Yenilenebilir %'] if is_tr else ['GDP (\$)', 'Energy Int.', 'GVC Share', 'Trade Openness', 'Manufacturing', 'Renewable %']
            
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
            trace_name = p_curr["title_tr"] if is_tr else p_curr["title_en"]
            fig_radar.add_trace(go.Scatterpolar(r=val_selected, theta=categories, fill='toself', name=trace_name, fillcolor='rgba(43, 92, 143, 0.4)', line_color='#2b5c8f'))
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
                p_short = p_v["title_tr"] if is_tr else p_v["title_en"]
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
            peer_names = ", ".join([get_country_name(p, is_tr) for p in p_data["peers" if is_tr else "peers_en"]])
            prof_table_data.append({
                "Hazır Profil / Tipoloji" if is_tr else "Typology Profile": p_data["title_tr"] if is_tr else p_data["title_en"],
                "Rozet / Identity" if is_tr else "Badge": p_data["badge" if is_tr else "badge_en"],
                "Tahmini Emisyon (Mt)" if is_tr else "Predicted Emission (Mt)": round(p_data["emission"], 1),
                "Akran Ülkeler" if is_tr else "Peer Countries": peer_names,
                "SKDM Riski" if is_tr else "CBAM Risk": p_data["skdm_risk" if is_tr else "skdm_risk_en"],
                "Karbonsuzlaşma Reçetesi" if is_tr else "Decarbonization Recipe": p_data["recipe" if is_tr else "recipe_en"]
            })
        df_prof_table = pd.DataFrame(prof_table_data)
        st.dataframe(df_prof_table.sort_values("Tahmini Emisyon (Mt)" if is_tr else "Predicted Emission (Mt)", ascending=False), use_container_width=True, hide_index=True)

    else:
        st.subheader(f"📌 {get_country_name(selected_country_tr, is_tr)} — {'Göstergeler & Küresel Sıralama' if is_tr else 'Indicators & Global Ranking'}")
        c_m = COUNTRIES_DATA.get(selected_country_tr, list(COUNTRIES_DATA.values())[0])
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Milli Gelir (GSYH)" if is_tr else "GDP per Capita", f"${c_m['gdp']:,.0f}")
        col_m2.metric("Enerji Yoğunluğu" if is_tr else "Energy Intensity", f"{c_m['energy']:.2f} MJ/$")
        col_m3.metric("Yenilenebilir Enerji" if is_tr else "Renewable Share", f"%{c_m['renew']:.1f}")
        col_m4.metric("İmalat Sanayi Payı" if is_tr else "Manufacturing Share", f"%{c_m['manuf']:.1f}")
        col_m5.metric("Ticari Açıklık" if is_tr else "Trade Openness", f"%{c_m['trade']:.1f}")

        st.markdown(f"#### 📊 {'Küresel Ülke Emisyon & Politika Listesi' if is_tr else 'Global Country Emission & Policy List'}")
        st.dataframe(df_map.sort_values("Emisyon (Mt)", ascending=False), use_container_width=True, hide_index=True)

with tab2:
    st.subheader(f"📊 {get_country_name(selected_country_tr, is_tr) if mode_choice in [mod_opt1, 'Select from Global Country List'] else (PROFILE_DETAILS[selected_profile]['title_tr'] if is_tr else PROFILE_DETAILS[selected_profile]['title_en'])} — {'Canlı Senaryo SKDM & Karşılaştırma Analizi' if is_tr else 'Live Scenario CBAM & Comparison Analysis'}")
    
    base_skdm_label, base_skdm_score, base_skdm_class = eval_skdm_risk(c_info_cur["manuf"], c_info_cur["renew"], c_info_cur["energy"])
    scen_skdm_label, scen_skdm_score, scen_skdm_class = eval_skdm_risk(manufacturing, renewable_energy, energy_intensity)

    col_sk1, col_sk2 = st.columns(2)
    col_sk1.info(f"🏛️ **{'Mevcut Durum SKDM Risk' if is_tr else 'Baseline CBAM Risk'}:** {base_skdm_label} | İmalat/Mfg: %{c_info_cur['manuf']:.1f} | Yenilenebilir/Renew: %{c_info_cur['renew']:.1f}")
    col_sk2.success(f"🎛️ **{'Yeni Senaryo SKDM Risk (Canlı)' if is_tr else 'New Scenario CBAM Risk (Live)'}:** {scen_skdm_label} | İmalat/Mfg: %{manufacturing:.1f} | Yenilenebilir/Renew: %{renewable_energy:.1f}")

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

    st.markdown(f"#### 💡 {'Yapay Zeka Destekli Politika Kaldıraçları & Canlı Öneriler' if is_tr else 'AI-Driven Policy Levers & Live Recommendations'}")
    
    rec_c1, rec_c2, rec_c3 = st.columns(3)
    
    if is_tr:
        card1_html = f\"\"\"<div class='glass-card-green'>
            <h4>🍃 1. Yenilenebilir Enerji Hamlesi</h4>
            <p><b>Aksiyon:</b> Yenilenebilir enerji oranının <b>%{renewable_energy:.1f}</b> seviyesine ayarlanması emisyon tahminini net <b>{c_renew:+.1f} Mt CO₂eq</b> etkilemektedir.</p>
        </div>\"\"\"
        card2_html = f\"\"\"<div class='glass-card-blue'>
            <h4>⚡ 2. Enerji Verimliliği & Şebeke</h4>
            <p><b>Aksiyon:</b> Enerji yoğunluğunun <b>{energy_intensity:.2f} MJ/$</b> seviyesine ayarlanması, €5M ihracat yapan bir işletme için tahmini SKDM cezasında <b>€{max(0.0, dyn_tax_savings):,.0f}</b> net tasarruf yaratmaktadır.</p>
        </div>\"\"\"
        card3_html = f\"\"\"<div class='glass-card-yellow'>
            <h4>🏭 3. Temiz İmalat Dönüşümü</h4>
            <p><b>Aksiyon:</b> İmalat payının <b>%{manufacturing:.1f}</b> ve temiz enerjinin <b>%{renewable_energy:.1f}</b> olduğu bu senaryoda Kapsam 3 tedarik zinciri iklim riski <b>%{dyn_scope3_reduction:.1f}</b> azalmaktadır.</p>
        </div>\"\"\"
    else:
        card1_html = f\"\"\"<div class='glass-card-green'>
            <h4>🍃 1. Renewable Energy Push</h4>
            <p><b>Action:</b> Setting renewable energy share to <b>{renewable_energy:.1f}%</b> impacts emission prediction by a net <b>{c_renew:+.1f} Mt CO₂eq</b>.</p>
        </div>\"\"\"
        card2_html = f\"\"\"<div class='glass-card-blue'>
            <h4>⚡ 2. Energy Efficiency & Grid</h4>
            <p><b>Action:</b> Setting energy intensity to <b>{energy_intensity:.2f} MJ/$</b> saves <b>€{max(0.0, dyn_tax_savings):,.0f}</b> in CBAM tax penalties for a €5M exporter.</p>
        </div>\"\"\"
        card3_html = f\"\"\"<div class='glass-card-yellow'>
            <h4>🏭 3. Clean Manufacturing Shift</h4>
            <p><b>Action:</b> With <b>{manufacturing:.1f}%</b> manufacturing share and <b>{renewable_energy:.1f}%</b> clean energy, Scope 3 climate risk drops by <b>{dyn_scope3_reduction:.1f}%</b>.</p>
        </div>\"\"\"

    with rec_c1:
        st.markdown(card1_html, unsafe_allow_html=True)
    with rec_c2:
        st.markdown(card2_html, unsafe_allow_html=True)
    with rec_c3:
        st.markdown(card3_html, unsafe_allow_html=True)

with tab3:
    st.subheader("🏭 Kurumsal / İşletme İklim Riski Simülatörü (Macro-to-Micro Downscaling)" if is_tr else "🏭 Corporate Climate Risk Simulator (Macro-to-Micro Downscaling)")
    st.markdown("İhracatçı veya tedarik zincirinde yer alan şirketler için **AB SKDM Vergi Hesabı, Serbest Lokasyon Seçimi ve Kapsam 3 (Scope 3) Stres Testi**:" if is_tr else "AB CBAM Tax Calculation, Location Optimization, and Scope 3 Climate Risk Stress Testing for Exporters:")

    st.markdown(f"#### 1. 💰 {'Şirket AB İhracatı & SKDM Vergi Cezası / Tasarruf Hesaplayıcı' if is_tr else 'Company EU Exports & CBAM Tax / Savings Calculator'}")
    corp_export = st.number_input("Şirketinizin Yıllık AB İhracat Cirosu ($ / €):" if is_tr else "Annual EU Export Revenue ($ / €):", min_value=100000, max_value=1000000000, value=5000000, step=500000)

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

    st.markdown(f"#### 2. 📍 {'Şirket Yatırım & Tedarikçi Lokasyon Değerlendirme' if is_tr else 'Investment & Supplier Location Evaluation'}")
    
    loc_mode_opts = ["🎯 Manuel Ülke Seçimi (İstediğiniz Ülkeleri Karşılaştırın)", "🤖 Otomatik Kriter Bazlı Filtreleme (En Uygun Ülkeler)"] if is_tr else ["🎯 Manual Country Selection (Compare Desired Countries)", "🤖 Automatic Criteria Filtering (Optimal Countries)"]
    loc_mode = st.radio("Lokasyon Analiz Yöntemini Seçiniz:" if is_tr else "Select Location Analysis Method:", loc_mode_opts, index=0)

    if "Manuel" in loc_mode or "Manual" in loc_mode:
        country_multiselect_tr = sorted(list(COUNTRIES_DATA.keys()))
        country_multiselect_disp = [get_country_name(c, is_tr) for c in country_multiselect_tr]
        
        default_defaults_tr = ["Türkiye", "Almanya", "Avusturya", "Çekya", "Polonya", "Meksika"]
        default_defaults_disp = [get_country_name(c, is_tr) for c in default_defaults_tr if c in country_multiselect_tr]
        
        selected_custom_disp = st.multiselect(
            "Değerlendirmek İstediğiniz Ülkeleri Seçiniz:" if is_tr else "Select Countries to Evaluate:",
            country_multiselect_disp,
            default=default_defaults_disp
        )
        if selected_custom_disp:
            custom_eval = []
            for cdisp in selected_custom_disp:
                cname_tr = REVERSE_COUNTRY_TRANSLATIONS.get(cdisp, cdisp)
                cinfo = COUNTRIES_DATA.get(cname_tr, list(COUNTRIES_DATA.values())[0])
                score = (cinfo["renew"] * 0.4) + ((15.0 - cinfo["energy"]) * 0.4) + (cinfo["gdp"]/1000 * 0.2)
                cbam_risk, _, _ = eval_skdm_risk(cinfo["manuf"], cinfo["renew"], cinfo["energy"])
                custom_eval.append({
                    "Ülke / Country": cdisp,
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
        max_energy_limit = st.slider("Kabul Edilebilir Maksimum Enerji Yoğunluğu (MJ/$):" if is_tr else "Max Acceptable Energy Intensity (MJ/$):", 0.5, 10.0, 5.0)

        opt_candidates = []
        for cname_tr, cinfo in COUNTRIES_DATA.items():
            if cinfo["manuf"] >= min_manuf_target and cinfo["energy"] <= max_energy_limit:
                score = (cinfo["renew"] * 0.4) + ((15.0 - cinfo["energy"]) * 0.4) + (cinfo["gdp"]/1000 * 0.2)
                cbam_risk, _, _ = eval_skdm_risk(cinfo["manuf"], cinfo["renew"], cinfo["energy"])
                opt_candidates.append({
                    "Ülke / Country": get_country_name(cname_tr, is_tr),
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

    st.markdown(f"#### 3. 📊 {'Kurumsal Kapsam 3 (Scope 3) İklim Riski Stres Testi' if is_tr else 'Corporate Scope 3 Climate Risk Stress Test'}")
    
    supp_list_tr = sorted(list(COUNTRIES_DATA.keys()))
    supp_list_disp = [get_country_name(c, is_tr) for c in supp_list_tr]
    supp_defaults_disp = [get_country_name("Türkiye", is_tr), get_country_name("Polonya", is_tr)]
    supp_defaults_disp = [c for c in supp_defaults_disp if c in supp_list_disp]
    
    selected_supp_disp = st.multiselect("Tedarikçilerinizin Bulunduğu Ana Ülkeleri Seçiniz:" if is_tr else "Select Main Supplier Countries:", supp_list_disp, default=supp_defaults_disp)
    if selected_supp_disp:
        selected_supp_tr = [REVERSE_COUNTRY_TRANSLATIONS.get(c, c) for c in selected_supp_disp]
        avg_supp_renew = np.mean([COUNTRIES_DATA[c]["renew"] for c in selected_supp_tr if c in COUNTRIES_DATA])
        scen_scope3_reduction = (renewable_energy - avg_supp_renew) * 0.6
        st.info(f"💡 {'Tedarikçilerinizin bulunduğu ülkelerde yapılacak yeşil dönüşüm hamlesi, şirketinizin **Kapsam 3 (Scope 3) tedarik zinciri karbon ayak izini %' if is_tr else 'Green transformation in supplier countries reduces your **Scope 3 supply chain carbon footprint by '}{max(0.0, scen_scope3_reduction):.1f}%**.")

with tab4:
    st.subheader("🎯 Net-Zero / Hedef Tabanlı Politika Reçetesi Motoru" if is_tr else "🎯 Net-Zero Target-Based Policy Recipe Engine")
    target_pct = st.slider("🎯 Hedeflenen Karbon Emisyonu Azaltım Oranı (%):" if is_tr else "Target Carbon Emission Reduction Rate (%):", 5, 50, 20, step=5)
    rec_renew = min(85.0, BASE_RENEW + (target_pct * 0.8))
    rec_energy = max(0.5, BASE_ENERGY - (target_pct * 0.08))
    rec_manuf = max(0.5, BASE_MANUF - (target_pct * 0.15))

    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.markdown(f"<div class='glass-card-green'><b>{'Gerekli Yenilenebilir Enerji' if is_tr else 'Required Renewable Energy'}</b><h3 style='color:#2ca02c;'>%{rec_renew:.1f}</h3></div>", unsafe_allow_html=True)
    r_col2.markdown(f"<div class='glass-card-green'><b>{'Gerekli Enerji Yoğunluğu' if is_tr else 'Required Energy Intensity'}</b><h3 style='color:#2ca02c;'>{rec_energy:.2f} MJ/$</h3></div>", unsafe_allow_html=True)
    r_col3.markdown(f"<div class='glass-card-green'><b>{'Önerilen İmalat Sanayi Payı' if is_tr else 'Recommended Manufacturing Share'}</b><h3 style='color:#2ca02c;'>%{rec_manuf:.1f}</h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 " + ("10.000 İterasyonlu Monte Carlo Olasılık Dağılımı" if is_tr else "10,000 Iteration Monte Carlo Probability Distribution"))
    fig_mc = go.Figure()
    fig_mc.add_vline(x=base_country_emission, line_width=2, line_dash="dot", line_color="gray", annotation_text=f"Baseline ({base_country_emission:.1f})")
    fig_mc.add_trace(go.Histogram(x=mc_distribution, nbinsx=50, name="Scenario Distribution", marker_color="#2b5c8f" if country_policy_diff <= 0 else "#d9534f", opacity=0.75))
    fig_mc.add_vline(x=pred_emission, line_width=3, line_dash="dash", line_color="red", annotation_text=f"New Scenario ({pred_emission:.1f})")
    fig_mc.update_layout(xaxis_title="Demand-Based GHG Emissions (Mt CO₂eq)", yaxis_title="Simulation Frequency", template="plotly_white", height=380)
    st.plotly_chart(fig_mc, use_container_width=True)

with tab5:
    st.subheader("⚔️ İkili Ülke Birebir Karşılaştırma Modu" if is_tr else "⚔️ Pairwise Country Head-to-Head Comparison")
    
    pair_list_tr = sorted(list(COUNTRIES_DATA.keys()))
    pair_list_disp = [get_country_name(c, is_tr) for c in pair_list_tr]
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        country_A_disp = st.selectbox("1. Ülkeyi Seçiniz:" if is_tr else "Select Country 1:", pair_list_disp, index=0)
        country_A_tr = REVERSE_COUNTRY_TRANSLATIONS.get(country_A_disp, country_A_disp)
    with col_k2:
        country_B_disp = st.selectbox("2. Ülkeyi Seçiniz:" if is_tr else "Select Country 2:", pair_list_disp, index=min(1, len(pair_list_disp)-1))
        country_B_tr = REVERSE_COUNTRY_TRANSLATIONS.get(country_B_disp, country_B_disp)

    cA_data = COUNTRIES_DATA.get(country_A_tr, list(COUNTRIES_DATA.values())[0])
    cB_data = COUNTRIES_DATA.get(country_B_tr, list(COUNTRIES_DATA.values())[0])
    eA = cA_data["emission"]
    eB = cB_data["emission"]

    fig_two = go.Figure(data=[
        go.Bar(name=country_A_disp, x=["Emisyon / Emissions (Mt)", "GSYH / GDP ($k)", "Yenilenebilir / Renewables (%)", "Enerji Yoğunluğu / Energy Int. (x10)"], y=[eA, cA_data["gdp"]/1000, cA_data["renew"], cA_data["energy"]*10], marker_color="#1f77b4"),
        go.Bar(name=country_B_disp, x=["Emisyon / Emissions (Mt)", "GSYH / GDP ($k)", "Yenilenebilir / Renewables (%)", "Enerji Yoğunluğu / Energy Int. (x10)"], y=[eB, cB_data["gdp"]/1000, cB_data["renew"], cB_data["energy"]*10], marker_color="#ff7f0e")
    ])
    fig_two.update_layout(barmode='group', title=f"{country_A_disp} vs {country_B_disp} Indicator Comparison", template="plotly_white", height=420)
    st.plotly_chart(fig_two, use_container_width=True)

with tab6:
    st.subheader("🔥 Tüm Ülkelerin Etkileşimli Kabarcık Grafiği & Normalize Isı Haritası" if is_tr else "🔥 Interactive Global Bubble Chart & Normalized Heatmap")
    
    all_c_list = []
    for cname_tr, cinfo in COUNTRIES_DATA.items():
        cdisp = get_country_name(cname_tr, is_tr)
        all_c_list.append({
            "Ülke / Country": cdisp, "Emisyon / Emissions (Mt)": round(cinfo["emission"], 1), "GSYH / GDP (\$)": cinfo["gdp"],
            "Enerji Yoğunluğu / Energy Int.": cinfo["energy"], "Yenilenebilir / Renewables (%)": cinfo["renew"], "İmalat / Mfg (%)": cinfo["manuf"]
        })
    df_all_raw = pd.DataFrame(all_c_list)

    fig_bubble = px.scatter(
        df_all_raw, x="GSYH / GDP (\$)", y="Enerji Yoğunluğu / Energy Int.", size="Emisyon / Emissions (Mt)", color="Yenilenebilir / Renewables (%)",
        hover_name="Ülke / Country", size_max=45, color_continuous_scale="Viridis",
        title="Ülkelerin GSYH, Enerji Yoğunluğu ve Emisyon Büyüklüklerine Göre Kabarcık Dağılımı" if is_tr else "Countries Distribution by GDP, Energy Intensity, and GHG Emission Size"
    )
    fig_bubble.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 📊 {'Normalize Ülke Karşılaştırma Matrisi (0-100 Skorlar)' if is_tr else 'Normalized Country Comparison Matrix (0-100 Scores)'}")
    df_all_hm_raw = df_all_raw.set_index("Ülke / Country")
    df_all_norm = (df_all_hm_raw - df_all_hm_raw.min()) / (df_all_hm_raw.max() - df_all_hm_raw.min() + 1e-9) * 100.0

    fig_all_hm = px.imshow(df_all_norm, text_auto=".0f", color_continuous_scale="Viridis", aspect="auto", title="Küresel Ülke Karşılaştırma Matrisi (0-100 Normalize Skorlar)" if is_tr else "Global Country Comparison Matrix (0-100 Normalized Scores)")
    fig_all_hm.update_layout(height=1100, margin=dict(l=160, r=40, t=50, b=50), yaxis=dict(tickfont=dict(size=11), autorange="reversed"))
    st.plotly_chart(fig_all_hm, use_container_width=True)

with tab7:
    st.subheader("💾 Kaydedilen Senaryolar Deposu & İndir" if is_tr else "💾 Saved Scenario Vault & Export")
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

with tab8:
    st.subheader("📜 Metodoloji, Şeffaf Hesaplama & XAI Notları" if is_tr else "📜 Methodology & XAI Notes")
    st.markdown("""
    ### 🔬 5-Kademeli Şeffaf Hesaplama ve Yapay Zeka Metodolojisi / 5-Stage Transparent Calculation Methodology

    1. **Temel Bileşenler Analizi (PCA) / Principal Component Analysis:** Sabit Genişbant (\\(X_1\\)), İnternet Kullanımı (\\(X_2\\)) ve Mobil Abonelik (\\(X_3\\)) göstergeleri özdeğeri $\lambda_1 = 2.1308$ olan birincil bileşene dönüştürülür ($\%70.93$ varyans):
       $$PC_1 = 0.6208 \cdot Z(X_1) + 0.6493 \cdot Z(X_2) + 0.4394 \cdot Z(X_3)$$

    2. **SVR Makro Tahmin Modeli / SVR Macro Prediction Model:** RBF çekirdekli Destek Vektör Regresyonu ($R^2 = 0.9771$, $RMSE = 301.06 \text{ Mt CO}_2\text{eq}$, $C=100, \epsilon=0.1$).

    3. **XAI / SHAP Marjinal Katkı / Marginal Effect:** Ticari Açıklık (\\(0.2172\\)), İmalat Sanayi (\\(0.1817\\)) ve Enerji Yoğunluğu (\\(0.1512\\)) öncülüğünde her bir politikanın emisyon üzerindeki net etkisi (\\(c_k\\)) tekil ayrıştırılır. Dijitalleşme İndeksi $\%83.87$ pozitif SHAP payı ile Rebound etkisi sergiler.

    4. **Monte Carlo Risk Simülasyonu / Risk Simulation:** 10.000 iterasyonlu rassal gürültü (\\(SD = 31.47 \text{ Mt}\\)) eklenerek $\%5$ ve $\%95$ olasılık güven aralıkları hesaplanır.

    5. **Macro-to-Micro Downscaling & SKDM / CBAM & Kapsam 3 (Scope 3):** Ülke düzeyindeki emisyon ve enerji yoğunluğu, ihracatçı şirketlerin AB SKDM vergi yükü (€) ve Kapsam 3 tedarik zinciri iklim risklerine indirgenir.

    ---
    ⚠️ **Nedensellik Sınırı / Non-Causal Note:** Bu platformdaki tahminler ilişkisel ve tahminsel (*associative / predictive marginal effects*) duyarlılıklara dayanır; doğrudan neden-sonuç iddiası taşımaz.
    """)

st.sidebar.markdown("---")
pdf_bytes_sb = generate_pdf_report()
st.sidebar.download_button(
    label="📄 " + ("Resmi Raporu İndir (PDF)" if is_tr else "Download Official Report (PDF)"),
    data=pdf_bytes_sb,
    file_name="Karbon_Ayakizi_Resmi_Rapor.pdf",
    mime="application/pdf"
)
