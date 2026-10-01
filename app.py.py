st.set_page_config(
    page_title="Küresel Karbon Ayak İzi Simülatörü",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "🌍 Küresel Tedarik Zinciri & Karbon Ayak İzi Karar Destek Simülatörü"
)
st.markdown(
    "**Açıklanabilir Yapay Zeka (SVR & Monte Carlo)** destekli bu simülatör, makroekonomik ve dijital politika senaryolarının talep tabanlı seragazı emisyonları üzerindeki duyarlılığını canlı olarak öngörür."
)


# Modelleri Yükleme
@st.cache_resource
def load_models():
    svr = joblib.load("svr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return svr, scaler, pca


try:
    svr_model, scaler, pca = load_models()
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi! Hata: {e}")
    st.stop()

# Sol Menü (Politika Girdileri)
st.sidebar.header("🎛️ Politika ve Ekonomi Girdileri")

gdp = st.sidebar.slider(
    "Kişi Başı GSYH ($)", 1000, 80000, 25000, step=1000
)
energy_intensity = st.sidebar.slider(
    "Enerji Yoğunluğu (MJ/$)", 1.0, 15.0, 5.2, step=0.1
)
trade_openness = st.sidebar.slider(
    "Ticari Açıklık (% GSYH)", 20.0, 200.0, 85.0, step=1.0
)
manufacturing = st.sidebar.slider(
    "İmalat Sanayi Payı (% GSYH)", 5.0, 45.0, 18.0, step=0.5
)
renewable_energy = st.sidebar.slider(
    "Yenilenebilir Enerji Payı (%)", 0.0, 80.0, 22.0, step=1.0
)
gvc_output = st.sidebar.slider(
    "GVC Çıktısı ($)", 100.0, 5000.0, 1200.0, step=50.0
)

st.sidebar.subheader("📱 Dijitalleşme Altyapı Göstergeleri")
broadband = st.sidebar.slider("Sabit Geniş Bant Aboneliği", 0.0, 50.0, 20.0)
internet_users = st.sidebar.slider("İnternet Kullanıcı Oranı (%)", 10.0, 100.0, 75.0)
mobile_sub = st.sidebar.slider("Mobil Abonelik (100 Kişide)", 30.0, 200.0, 110.0)

# Hesaplama Butonu ve Tahmin Akışı
if st.button("🚀 Senaryo Tahminini ve Risk Analizini Çalıştır"):
    # 1. PCA ile Dijitalleşme İndeksini Hesaplama
    dig_inputs = np.array([[broadband, internet_users, mobile_sub]])
    dig_index = pca.transform(dig_inputs)[0][0]

    # 2. Model Özellik Matrisi
    raw_features = np.array(
        [[
            gdp,
            energy_intensity,
            trade_openness,
            manufacturing,
            renewable_energy,
            gvc_output,
            dig_index,
        ]]
    )

    # 3. Ölçeklendirme ve SVR Tahmini
    scaled_features = scaler.transform(raw_features)
    pred_emission = svr_model.predict(scaled_features)[0]

    # 4. Monte Carlo %90 Güven Aralığı Simülasyonu
    np.random.seed(42)
    residuals = np.random.normal(0, 32.14, 10000)  # Model RMSE sapması
    mc_distribution = pred_emission + residuals
    lower_bound = np.percentile(mc_distribution, 5)
    upper_bound = np.percentile(mc_distribution, 95)

    # Sonuç Kartları
    col1, col2, col3 = st.columns(3)
    col1.metric("Tahmini Karbon Ayak İzi", f"{pred_emission:.2f} Mt CO₂eq")
    col2.metric("Alt Güven Sınırı (%5)", f"{lower_bound:.2f} Mt CO₂eq")
    col3.metric("Üst Güven Sınırı (%95)", f"{upper_bound:.2f} Mt CO₂eq")

    # Görselleştirme (Plotly Senaryo Dağılımı)
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
        annotation_text="Nokta Tahmin",
    )
    fig.update_layout(
        title="10.000 İterasyonlu Monte Carlo Olasılık Dağılımı",
        xaxis_title="Tahmini Talep Tabanlı GHG Emisyonu (Mt CO₂eq)",
        yaxis_title="Frekans",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)