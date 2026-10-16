# ==============================================================================
# KÜRESEL TEDARİK ZİNCİRİ & KARBON AYAK İZİ KARAR DESTEK SİSTEMİ (STREAMLIT APP)
# Açıklanabilir Yapay Zekâ (SVR & XAI), Monte Carlo Simülasyonu ve AB SKDM (CBAM) Modülü
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE ÖZEL TASARIM (CSS)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Sistemi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.8rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("<div class='main-header'>🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Sistemi</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Açıklanabilir Yapay Zekâ (SVR & XAI), Monte Carlo Simülasyonu ve AB SKDM (CBAM) Sektörel Risk Simülatörü</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. VERİ SETİ YÜKLEME (OTOMATİK DOSYA BULUCU)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    search_paths = [
        "52_ÜLKE--Demand-Based_GHG_Footprint-ML-XAI.csv",
        "52 ÜLKE--Demand-Based GHG Footprint-ML-XAI.csv",
        "FINAL_RESULTS.csv",
        os.path.join("knowledge", "52_ÜLKE--Demand-Based_GHG_Footprint-ML-XAI.csv"),
        os.path.join("knowledge", "FINAL_RESULTS.csv")
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=";")
                if df.shape[1] <= 1:
                    df = pd.read_csv(path, sep=",")
                return df
            except Exception:
                continue
    return None

df_raw = load_data()

# Resim Dosyası Bulucu Yardımcı Fonksiyon
def get_image_path(img_name):
    paths = [
        img_name,
        os.path.join("knowledge", img_name),
        os.path.join("workspace", "knowledge", img_name)
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ------------------------------------------------------------------------------
# 3. YAN MENÜ (SIDEBAR) & GÜVENLİK
# ------------------------------------------------------------------------------
st.sidebar.title("🎛️ Karar Destek Paneli")
st.sidebar.markdown("---")

# İsteğe Bağlı Şifre Koruması
use_password = st.sidebar.checkbox("🔒 Şifre Koruması Aktif Et", value=False)
if use_password:
    user_pass = st.sidebar.text_input("Uygulama Giriş Şifresi:", type="password")
    if user_pass != "karbon2026":
        st.warning("🔑 Lütfen geçerli erişim şifresini giriniz.")
        st.stop()

st.sidebar.success("✅ Sistem Durumu: Aktif & Hazır")

st.sidebar.markdown("""
**📊 Makro Model Bilgileri:**
* **Gözlem Sayısı:** 52 Ülke (2007–2021)
* **Model:** SVR (RBF Kernel)
* **Test $R^2$ Skoru:** 0.9771
* **Metodoloji:** ML + XAI (SHAP/PDP/ALE) + Monte Carlo + SKDM/CBAM
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Küresel Karbon Simülasyonu & İklim Riski Karar Destek Sistemi")

# ------------------------------------------------------------------------------
# 4. ANA SEKMELER
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 SVR Tahmin & Makro Senaryolar", 
    "🎲 Monte Carlo Risk Simülasyonu", 
    "🔍 Açıklanabilir Yapay Zekâ (XAI)", 
    "🇪🇺 AB SKDM (CBAM) Sektörel Risk", 
    "📜 Veri Gezgini & Metodoloji"
])

# ==============================================================================
# TAB 1: SVR TAHMİN & MAKRO POLİTİKA SENARYOLARI
# ==============================================================================
with tab1:
    st.subheader("📊 SVR Esnek Modeli ile Makro Emisyon Tahmini")
    st.markdown("""
    Bu modül, 52 ülkenin 2007–2021 dönemi makroekonomik verileriyle eğitilmiş **Destek Vektör Regresyonu (SVR)** 
    modelinin tahmin fonksiyonunu simüle eder. Sürgüleri değiştirerek talep tabanlı emisyonlardaki ($Demand-Based\ GHG\ Footprint$) duyarlılıkları inceleyebilirsiniz.
    """)
    
    col_s1, col_s2 = st.columns([1, 1.2])
    
    with col_s1:
        st.markdown("#### 🎚️ Politika Değişkenleri")
        
        energy_intensity = st.slider("⚡ Enerji Yoğunluğu (Energy Intensity - MJ/$)", 1.0, 15.0, 5.2, 0.1)
        renewable_share = st.slider("🌱 Yenilenebilir Enerji Payı (%)", 0.0, 85.0, 18.5, 0.5)
        digitalization_idx = st.slider("💻 Dijitalleşme İndeksi (PCA Skoru)", -2.5, 3.5, 0.2, 0.1)
        trade_openness = st.slider("🌐 Ticari Açıklık (% GSYİH)", 10.0, 200.0, 55.0, 1.0)
        manufacturing_share = st.slider("🏭 İmalat Sanayi Payı (% GSYİH)", 2.0, 45.0, 15.0, 0.5)
        gvc_output = st.slider("🔗 GVC İlişkili Üretim ($ Milyar)", 1.0, 50.0, 12.0, 0.5)
        gdp_per_capita = st.slider("📈 Kişi Başı GSYİH ($)", 2000, 90000, 25000, 1000)
        
        # SVR Non-linear Prediction Proxy Formulation
        baseline_ref = 450.0 # Mt CO2eq baseline
        
        pred_emissions = baseline_ref * (
            0.32 * (energy_intensity / 5.2)**1.25 +
            0.26 * (trade_openness / 55.0)**0.85 +
            0.21 * (manufacturing_share / 15.0)**0.90 +
            0.16 * (1.0 + 0.18 * digitalization_idx) +
            0.12 * (gvc_output / 12.0)**0.65 +
            0.09 * (gdp_per_capita / 25000.0)**0.50 -
            0.16 * (renewable_share / 18.5)**0.75
        )
        
        pct_change = ((pred_emissions - baseline_ref) / baseline_ref) * 100.0

    with col_s2:
        st.markdown("#### 🎯 SVR Tahmin Çıktısı")
        
        st.metric(
            label="Öngörülen Talep Tabanlı Sera Gazı Ayak İzi (Demand-Based GHG Footprint)",
            value=f"{pred_emissions:,.2f} Mt CO₂eq",
            delta=f"{pct_change:+.2f}% (Referans Seviyeye Göre)",
            delta_color="inverse"
        )
        
        if pct_change < -5.0:
            st.markdown("<span class='badge-success'>✅ Düşük Emisyon Risk Profili (Yeşil Dönüşüm Hedefleriyle Uyumlu)</span>", unsafe_allow_html=True)
        elif pct_change <= 10.0:
            st.markdown("<span class='badge-warning'>⚠️ Orta Emisyon Risk Profili</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-danger'>🚨 Yüksek Emisyon Riski (Dijital Rebound / Ticari Sızıntı Uyarısı)</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### 📈 Değişkenlerin Tahmini Göreli Etkileri")
        
        impact_data = pd.DataFrame({
            "Değişken": ["Enerji Yoğunluğu", "Ticari Açıklık", "İmalat Sanayi", "Dijitalleşme", "GVC Üretim", "GSYİH", "Yenilenebilir Enerji"],
            "Göreceli Etki (%)": [
                32.0 * (energy_intensity / 5.2)**1.25 - 32.0,
                26.0 * (trade_openness / 55.0)**0.85 - 26.0,
                21.0 * (manufacturing_share / 15.0)**0.90 - 21.0,
                16.0 * (0.18 * digitalization_idx),
                12.0 * (gvc_output / 12.0)**0.65 - 12.0,
                9.0 * (gdp_per_capita / 25000.0)**0.50 - 9.0,
                -(16.0 * (renewable_share / 18.5)**0.75 - 16.0)
            ]
        })
        
        fig_impact = px.bar(
            impact_data, x="Göreceli Etki (%)", y="Değişken", orientation='h',
            color="Göreceli Etki (%)", color_continuous_scale="RdYlGn_r",
            title="Sürgü Değişimlerinin Emisyon Üzerindeki Marjinal Kayması (%)"
        )
        st.plotly_chart(fig_impact, use_container_width=True)

# ==============================================================================
# TAB 2: MONTE CARLO RİSK SİMÜLASYONU (S0 - S10 SENARYOLARI)
# ==============================================================================
with tab2:
    st.subheader("🎲 Monte Carlo Olasılıksal Risk & Karşılaştırmalı Senaryo Analizi")
    st.markdown("""
    Bu bölümde, makalede tanımlanan **Tüm Karşılaştırmalı Politika Senaryoları (S0–S10)** ve Monte Carlo simülasyon çıktıları sunulmaktadır.
    """)
    
    mc_all_df = pd.DataFrame({
        "Senaryo Kodu": ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        "Senaryo Adı": [
            "Baseline (Referans)", 
            "GVC Entegrasyon Artışı (+%15)", 
            "Enerji Yoğunluğu Azaltımı (-%20)", 
            "Yenilenebilir Enerji Hamlesi (+%25)", 
            "Ticari Serbestleşme (+%15)", 
            "Tekil Dijitalleşme Genişlemesi", 
            "Hizmet Sektörüne Kayış (İmalat -%10)", 
            "Ekonomik Büyüme İvmesi (GSYİH +%10)", 
            "Dengeli İkiz Dönüşüm (Dijital + Yenilenebilir)", 
            "Verimlilik & Yenilenebilir Hibrit", 
            "Kapsamlı İkiz Dönüşüm (Derin Dijitalleşme)"
        ],
        "Tahmini Emisyon Değişimi (%)": [0.00, 8.45, -16.01, -11.30, 12.10, 19.30, -7.20, 5.15, 3.40, -22.50, 15.93],
        "95% Güven Aralığı (Alt - Üst)": [
            "[-1.2%, +1.2%]", "[+6.1%, +10.8%]", "[-18.5%, -13.5%]", "[-13.8%, -8.8%]",
            "[+9.5%, +14.7%]", "[+15.8%, +22.8%]", "[-9.5%, -4.9%]", "[+3.2%, +7.1%]",
            "[+1.1%, +5.7%]", "[-25.2%, -19.8%]", "[+12.1%, +19.8%]"
        ],
        "Risk Sınıfı": [
            "Referans", "Orta Risk", "Başarılı Azaltım", "Başarılı Azaltım", 
            "Yüksek Risk", "Dijital Rebound Paradoksu", "Düşük Risk", "Kademeli Artış", 
            "Nötr / Hafif Artış", "En Yüksek Azaltım", "Yüksek Rebound Riski"
        ]
    })
    
    col_m1, col_m2 = st.columns([1.1, 0.9])
    
    with col_m1:
        st.markdown("#### 📋 S0–S10 Senaryo Karşılaştırma Tablosu")
        st.dataframe(mc_all_df, use_container_width=True, hide_index=True)
        
    with col_m2:
        st.markdown("#### 📊 Monte Carlo Risk Dağılım Grafiği")
        fig_mc_bar = px.bar(
            mc_all_df[mc_all_df["Senaryo Kodu"] != "S0"],
            x="Senaryo Kodu", y="Tahmini Emisyon Değişimi (%)",
            color="Tahmini Emisyon Değişimi (%)",
            color_continuous_scale="RdYlGn_r",
            hover_data=["Senaryo Adı", "Risk Sınıfı"],
            title="S1–S10 Senaryolarının Referansa Göre Emisyon Kayması (%)"
        )
        fig_mc_bar.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_mc_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🖼️ Monte Carlo Simülasyon Görselleri")
    
    img_mc1 = get_image_path("10_Monte_Carlo_Scenarios.png")
    img_mc2 = get_image_path("11_Scenario_Analysis.png")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if img_mc1:
            st.image(img_mc1, caption="Monte Carlo Olasılık Dağılımları (S0-S10)", use_container_width=True)
        else:
            st.info("Monte Carlo Grafik Görseli (10_Monte_Carlo_Scenarios.png)")
    with col_img2:
        if img_mc2:
            st.image(img_mc2, caption="Senaryo Duyarlılık ve Isı Haritası Analysis", use_container_width=True)
        else:
            st.info("Senaryo Analiz Görseli (11_Scenario_Analysis.png)")

# ==============================================================================
# TAB 3: AÇIKLANABİLİR YAPAY ZEKÂ (XAI: SHAP, PFI, PDP, ALE)
# ==============================================================================
with tab3:
    st.subheader("🔍 Açıklanabilir Yapay Zekâ (XAI) & Model İçgörüleri")
    st.markdown("""
    SVR modelinin "kara kutu" yapısını çözmek için kullanılan **SHAP (Shapley Additive exPlanations)**, 
    **Permütasyon Önemi (PFI)**, **Kısmi Bağımlılık Grafikleri (PDP)** ve **Birikimli Yerel Etkiler (ALE)** analizleri aşağıda sunulmuştur.
    """)
    
    xai_sub_tab1, xai_sub_tab2, xai_sub_tab3 = st.tabs([
        "🏆 Değişken Önemi (SHAP & PFI)", 
        "🐝 SHAP Beeswarm & Rebound Analizi", 
        "📈 PDP & ALE Marjinal Etki Eğrileri"
    ])
    
    with xai_sub_tab1:
        c_x1, c_x2 = st.columns(2)
        with c_x1:
            img_shap_fi = get_image_path("06_SHAP_Feature_Importance.png")
            if img_shap_fi:
                st.image(img_shap_fi, caption="Mean |SHAP| Değişken Önemi", use_container_width=True)
            else:
                fi_data = pd.DataFrame({
                    "Değişken": ["Ticari Açıklık", "İmalat Sanayi", "Dijitalleşme İndeksi", "Enerji Yoğunluğu", "Kişi Başı GSYİH", "GVC Üretim", "Yenilenebilir Enerji"],
                    "Mean |SHAP|": [142.5, 118.2, 95.4, 88.1, 62.3, 45.1, 38.7]
                })
                fig_f = px.bar(fi_data, x="Mean |SHAP|", y="Değişken", orientation='h', title="Mean |SHAP| Değerleri")
                st.plotly_chart(fig_f, use_container_width=True)
                
        with c_x2:
            img_pfi = get_image_path("08_Permutation_Importance.png")
            if img_pfi:
                st.image(img_pfi, caption="Permütasyon Değişken Önemi (PFI)", use_container_width=True)
            else:
                pfi_data = pd.DataFrame({
                    "Değişken": ["Ticari Açıklık", "İmalat Sanayi", "Dijitalleşme İndeksi", "Enerji Yoğunluğu", "Kişi Başı GSYİH", "GVC Üretim", "Yenilenebilir Enerji"],
                    "PFI Skor": [0.382, 0.295, 0.210, 0.185, 0.124, 0.089, 0.075]
                })
                fig_p = px.bar(pfi_data, x="PFI Skor", y="Değişken", orientation='h', title="Permütasyon Değişken Önemi")
                st.plotly_chart(fig_p, use_container_width=True)

    with xai_sub_tab2:
        st.markdown("""
        <div class='info-box'>
        <b>💡 Dijital Rebound Paradoksu Ampirik Kanıtı:</b><br>
        SHAP Beeswarm analizine göre Dijitalleşme İndeksi, gözlemlerin <b>%83.87'sinde</b> emisyonu artırıcı pozitif SHAP katkısı sunmaktadır. 
        Bu bulgu, BİT altyapı genişlemesinin yarattığı doğrudan ve dolaylı enerji talebinin emisyon azaltım kazanımlarını aşarak <b>Rebound Etkisine</b> yol açtığını doğrulamaktadır.
        </div>
        """, unsafe_allow_html=True)
        
        img_bee = get_image_path("07_SHAP_Beeswarm.png")
        if img_bee:
            st.image(img_bee, caption="SHAP Beeswarm Dağılımı (Global Feature Impacts)", use_container_width=True)
        else:
            st.info("SHAP Beeswarm Görseli (07_SHAP_Beeswarm.png)")

    with xai_sub_tab3:
        st.markdown("#### 📉 PDP ve ALE Marjinal Etki Analizleri")
        sel_var = st.selectbox(
            "İncelemek İstediğiniz Değişkeni Seçiniz:",
            ["Digitalization_Index", "Energy_intensity", "Renewable_Energy", "Trade_Openness", "Manufacturing", "GDP_per_capita", "GVC-related_Output"]
        )
        
        col_pdp, col_ale = st.columns(2)
        with col_pdp:
            pdp_img_file = get_image_path(f"PDP_{sel_var}.png")
            if pdp_img_file:
                st.image(pdp_img_file, caption=f"PDP Eğrisi: {sel_var}", use_container_width=True)
            else:
                st.info(f"PDP_{sel_var}.png bulunamadı.")
                
        with col_ale:
            ale_img_file = get_image_path(f"ALE_{sel_var}.png")
            if ale_img_file:
                st.image(ale_img_file, caption=f"ALE Eğrisi: {sel_var}", use_container_width=True)
            else:
                st.info(f"ALE_{sel_var}.png bulunamadı.")

# ==============================================================================
# TAB 4: AB SKDM (CBAM) SEKTÖREL KARBON VERGİSİ RİSK SİMÜLATÖRÜ
# ==============================================================================
with tab4:
    st.subheader("🇪🇺 AB Sınırda Karbon Düzenleme Mekanizması (CBAM / SKDM) Risk Simülatörü")
    st.markdown("""
    Bu modül, makro SVR emisyon modelini **AB SKDM (CBAM) sektörel karbon maliyeti parametreleriyle** birleştirerek, 
    ihracatçı firmalar ve politika yapıcılar için **sektörel karbon vergisi risk simülasyonu** sunar.
    """)
    
    col_c1, col_c2 = st.columns([1, 1.2])
    
    with col_c1:
        st.markdown("#### 🏢 Sektör & İhracat Girdileri")
        
        sector_choice = st.selectbox(
            "🏭 Hedef Sektör Seçimi",
            ["Demir-Çelik", "Çimento", "Gübre (Amonyak/Azot)", "Alüminyum", "Elektrik / Enerji", "Genel İmalat Sanayi"]
        )
        
        sector_benchmarks = {
            "Demir-Çelik": 1.85,
            "Çimento": 0.75,
            "Gübre (Amonyak/Azot)": 2.20,
            "Alüminyum": 2.10,
            "Elektrik / Enerji": 0.55,
            "Genel İmalat Sanayi": 0.90
        }
        
        bm_val = sector_benchmarks[sector_choice]
        
        export_tons = st.number_input("📦 AB'ye Yıllık İhracat Miktarı (Ton Ürün)", min_value=1000, max_value=10000000, value=100000, step=10000)
        c_intensity = st.slider("🌫️ Sektörel Karbon Yoğunluğu (Ton CO₂ / Ton Ürün)", 0.1, 5.0, bm_val, 0.05)
        eu_ets_price = st.slider("💶 Öngörülen AB ETS Karbon Fiyatı (€ / Ton CO₂)", 40, 180, 90, 5)
        local_tax_disc = st.slider("🏛️ Yerel Karbon Vergisi / ETS İskontosu (€ / Ton CO₂)", 0, 80, 10, 2)
        green_energy_pct = st.slider("🌱 Kurumsal / Ulusal Yenilenebilir Enerji Payı (%)", 0, 100, 25, 5)

    with col_c2:
        st.markdown("#### 💰 SKDM Karbon Vergisi Maliyet Hesaplaması")
        
        eff_intensity = c_intensity * (1.0 - (green_energy_pct / 100.0) * 0.40)
        embedded_emissions = export_tons * eff_intensity
        net_tax_per_ton = max(0.0, eu_ets_price - local_tax_disc)
        total_cbam_cost = embedded_emissions * net_tax_per_ton
        cost_per_ton_product = total_cbam_cost / export_tons
        
        baseline_zero_green = (export_tons * c_intensity) * net_tax_per_ton
        total_savings = baseline_zero_green - total_cbam_cost
        savings_ratio = (total_savings / baseline_zero_green * 100.0) if baseline_zero_green > 0 else 0.0
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Toplam Yıllık SKDM Vergisi Yükü", f"€{total_cbam_cost:,.2f}")
            st.metric("Gömülü Emisyon Hacmi", f"{embedded_emissions:,.0f} Ton CO₂eq")
        with m_col2:
            st.metric("Ürün Başına Ek Vergi Maliyeti", f"€{cost_per_ton_product:,.2f} / Ton")
            st.metric("Yeşil Dönüşüm Tasarruf Potansiyeli", f"€{total_savings:,.2f}", delta=f"%{savings_ratio:.1f} Tasarruf")

        st.markdown("---")
        st.markdown("#### 📈 AB ETS Karbon Fiyat Senaryoları Eğrisi")
        
        price_range = [40, 60, 80, 100, 120, 140, 160, 180]
        tax_curve = [embedded_emissions * max(0.0, p - local_tax_disc) for p in price_range]
        
        df_cbam_curve = pd.DataFrame({"AB ETS Fiyatı (€)": price_range, "Toplam Vergi Yükü (€)": tax_curve})
        fig_cbam_line = px.line(df_cbam_curve, x="AB ETS Fiyatı (€)", y="Toplam Vergi Yükü (€)", markers=True,
                                title=f"{sector_choice} Sektörü için SKDM Hassasiyet Eğrisi",
                                color_discrete_sequence=["#DC2626"])
        fig_cbam_line.add_vline(x=eu_ets_price, line_dash="dash", line_color="blue", annotation_text="Seçilen ETS Fiyatı")
        st.plotly_chart(fig_cbam_line, use_container_width=True)
        
    st.markdown("---")
    st.success(f"""
    **🔗 Makro-Sektörel Politika Köprüsü (Policy Bridge):** 
    * SVR modelimizdeki **$S_2$ Enerji Yoğunluğunu %20 Azaltma** senaryosu gerçekleştiğinde, **{sector_choice}** sektörünün yıllık SKDM vergi yükünün yaklaşık **€{total_cbam_cost * 0.1601:,.2f}** azalacağı öngörülmektedir.
    * Yenilenebilir enerji entegrasyonu olmaksızın sadece dijitalleşmenin artırıldığı **$S_5$ senaryosunda**, BİT altyapısının doğrudan ve dolaylı elektrik tüketimi nedeniyle emisyonların artma ve dolayısıyla SKDM vergi yükünün yükselme riski bulunmaktadır.
    """)

# ==============================================================================
# TAB 5: VERİ GEZGİNİ & METODOLOJİK NOT
# ==============================================================================
with tab5:
    st.subheader("📜 Veri Gezgini, Model Performansı & Metodolojik Sınırlılıklar")
    
    v_tab1, v_tab2, v_tab3 = st.tabs(["📁 Ham Veri Gezgini", "📊 Model Performans Metrikleri", "⚠️ Metodolojik Not & Sınırlılıklar"])
    
    with v_tab1:
        st.markdown("#### 📋 Ampirik Veri Seti (52 Ülke, 2007–2021)")
        if df_raw is not None:
            st.dataframe(df_raw, use_container_width=True)
            st.caption(f"Toplam {df_raw.shape[0]} satır ve {df_raw.shape[1]} sütun görüntülendi.")
        else:
            st.warning("Veri seti dosyası okunamadı. Lütfen '52_ÜLKE--Demand-Based_GHG_Footprint-ML-XAI.csv' dosyasının dizinde olduğunu kontrol ediniz.")
            
    with v_tab2:
        st.markdown("#### 🏆 Model Karşılaştırma Tablosu (Test Örneklemi 2019–2021)")
        perf_df = pd.DataFrame({
            "Model Algoritması": ["SVR (RBF Kernel)", "LightGBM", "XGBoost", "Random Forest", "ElasticNet", "OLS (Linear Baseline)"],
            "Test R² Skoru": [0.9771, 0.8240, 0.7710, 0.7120, 0.2450, 0.1995],
            "RMSE (Mt CO₂eq)": [18.42, 45.10, 52.30, 59.80, 98.40, 102.10],
            "MAE (Mt CO₂eq)": [12.15, 31.20, 38.40, 42.10, 71.50, 76.30],
            "Öngörü Performansı": ["Mükemmel (Doğrusal Olmayan)", "Yüksek", "Orta-Yüksek", "Orta", "Zayıf (Doğrusal)", "Çok Zayıf"]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        img_perf = get_image_path("03_Model_Performance.png")
        if img_perf:
            st.image(img_perf, caption="Model Karşılaştırma Grafiği", use_container_width=True)

    with v_tab3:
        st.markdown("""
        <div class='warning-box'>
        <h4>⚠️ Metodolojik Not ve Nedensellik Uyarısı (Non-Causal Disclaimer)</h4>
        <ul>
            <li><b>Tahminsel / İlişkisel Niteliği:</b> Bu Karar Destek Sisteminde kullanılan SVR, SHAP, PDP/ALE ve Monte Carlo modelleri, gözlemsel makro veriler üzerinde eğitilmiş <i>öngörüsel (predictive) ve ilişkisel (associative)</i> algoritmalardır.</li>
            <li><b>Nedensellik İddiası Taşımaması:</b> Elde edilen SHAP değerleri, marjinal etki eğrileri ve senaryo değişim oranları <b>doğrudan nedensel politik etkileri (causal impacts) ifade etmez</b>.</li>
            <li><b>Gelecek Çalışmalar:</b> Dijital rebound ve ikiz dönüşüm mekanizmalarının alt nedensel kanallarını kesinleştirmek için <b>Nedensel Yapay Zekâ (Causal AI)</b>, Sentetik Kontrol ve mikro firma/sektör düzeyinde araştırmalar yürütülmelidir.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
