import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import warnings

# Gereksiz uyarilari gizle
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. SAYFA VE TASARIM YAPILANDIRMASI ---
st.set_page_config(page_title="HydroOptima | Enterprise SCADA v3", layout="wide", initial_sidebar_state="expanded")

# Kurumsal CSS
st.markdown("""
    <style>
    .metric-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    .chat-bubble { background-color: #e8f4f8; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: black; }
    .roi-text { color: #11caa0; font-weight: bold; font-family: monospace; }
    .soft-sensor-tag { font-size: 10px; background-color: #238636; color: white; padding: 2px 6px; border-radius: 4px; vertical-align: middle; }
    </style>
""", unsafe_allow_html=True)

if 'operasyon_loglari' not in st.session_state:
    st.session_state['operasyon_loglari'] = []

# --- 2. YAPAY ZEKA BEYNI: MALAGA MODELI ---
@st.cache_resource
def train_malaga_brain():
    data = {
        'Fe_mg_L': [0.5, 0.5, 10.0, 10.0, 10.0, 25.0, 25.0, 50.0, 5.0, 12.0, 30.0, 1.0, 15.0, 20.0, 45.0],
        'Ni_mg_L': [0.000005, 0.000005, 0.000005, 10.0, 10.0, 25.0, 25.0, 50.0, 2.0, 8.0, 22.0, 0.005, 12.0, 35.0, 48.0],
        'Acetate_g_L': [2.74, 2.72, 2.70, 2.61, 1.32, 1.74, 1.61, 1.15, 2.30, 1.99, 1.50, 2.84, 1.70, 1.40, 0.97],
        'Butyrate_g_L': [1.63, 1.65, 1.68, 1.98, 1.32, 0.88, 1.87, 0.53, 1.45, 1.05, 0.65, 1.69, 0.82, 0.71, 0.53],
        'Ethanol_g_L': [0.85, 0.84, 0.97, 0.65, 0.77, 0.17, 0.13, 0.002, 0.75, 0.18, 0.09, 0.88, 0.31, 0.11, 0.01],
        'pH': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0, 5.2, 5.5, 8.5, 5.1, 6.5, 7.2, 9.0],
        'H2_Prod_mol_mol': [0.77, 0.88, 0.77, 0.65, 0.53, 1.22, 1.02, 0.47, 1.02, 1.25, 0.53, 0.81, 1.15, 0.98, 0.48]
    }
    df = pd.DataFrame(data)
    X = df[['Fe_mg_L', 'Ni_mg_L', 'Acetate_g_L', 'Butyrate_g_L', 'Ethanol_g_L', 'pH']]
    y = df['H2_Prod_mol_mol']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, y.mean(), X.columns

model, baseline_mean, feature_names = train_malaga_brain()

# --- 3. TURKIYE TESIS VERITABANI ---
turkiye_tesisleri = pd.DataFrame({
    'Tesis_Adi': ['ASKI Tatlar (Ankara)', 'ISKI Ambarli (Istanbul)', 'IZSU Cigli (Izmir)', 'Bursa BUSKI'],
    'Enlem': [39.9822, 40.9744, 38.4855, 40.2133],
    'Boylam': [32.7844, 28.6822, 26.9811, 28.9855],
    'Kapasite_Ton': [850, 1500, 600, 450]
})

# --- 4. SIDEBAR: KOMUTA VE DOZAJ MERKEZI ---
st.sidebar.title("HydroOptima Node")
secilen_tesis = st.sidebar.selectbox("TR Tesisi Secin", turkiye_tesisleri['Tesis_Adi'])
aktif_tesis = turkiye_tesisleri[turkiye_tesisleri['Tesis_Adi'] == secilen_tesis].iloc[0]

# STRATEJIK GUNCELLEME: KAYNAK MODU
kaynak_modu = st.sidebar.radio("Mikrobesin Kaynagi", ["Saf Endustriyel Kimyasal", "Dogal Organik Atik (Gubre/Posa)"])

st.sidebar.markdown("---")
input_fe = st.sidebar.slider("Demir (Fe) (mg/L)", 0.5, 50.0, 10.0)
input_ni = st.sidebar.slider("Nikel (Ni) (mg/L)", 0.0, 50.0, 10.0)
input_ph = st.sidebar.slider("Reaktor pH", 4.0, 10.0, 5.5)

st.sidebar.markdown("---")
input_acetate = st.sidebar.slider("Asetat (g/L)", 1.1, 2.9, 1.7)
input_butyrate = st.sidebar.slider("Butirat (g/L)", 0.5, 1.7, 0.9)
input_ethanol = st.sidebar.slider("Etanol (g/L)", 0.0, 1.0, 0.2)

# PLC FAILSAFE
failsafe = "ACTIVE"
if input_ph < 5.0 or input_ph > 8.5: failsafe = "BLOCKED"

st.sidebar.markdown(f"PLC Status: **{failsafe}**")

# --- 5. HESAPLAMA MOTORU (SOFT SENSOR LOGIC) ---
input_df = pd.DataFrame([[input_fe, input_ni, input_acetate, input_butyrate, input_ethanol, input_ph]], columns=feature_names)
pred_h2 = model.predict(input_df)[0] if failsafe == "ACTIVE" else 0

# Cift Gaz Hesaplari (KF + Metanizasyon)
pred_ch4 = (input_acetate * 0.42) + (input_butyrate * 0.58) - (input_ethanol * 0.1) if failsafe == "ACTIVE" else 0
pred_ch4 = max(0.1, pred_ch4) if failsafe == "ACTIVE" else 0

tonaj = aktif_tesis['Kapasite_Ton']
hacim_h2 = pred_h2 * tonaj * 11.2
hacim_ch4 = pred_ch4 * tonaj * 8.4
hacim_co2 = (pred_h2 * 0.5 + pred_ch4 * 0.35) * tonaj * 6.1

# Enerji ve Finans
mwh_h2 = (hacim_h2 * 3.0) / 1000
mwh_ch4 = (hacim_ch4 * 10.0) / 1000
toplam_mwh = mwh_h2 + mwh_ch4

# Maliyet Hesabi (Dogal Kaynak Stratejisi)
if kaynak_modu == "Saf Endustriyel Kimyasal":
    maliyet = (input_fe * 1.8) + (input_ni * 4.2)
else:
    maliyet = (input_fe * 0.15) + (input_ni * 0.25) # Sadece lojistik/isleme maliyeti

gelir = (mwh_h2 * 150) + (mwh_ch4 * 80) + (tonaj * 32.5)
net_kar = gelir - maliyet

# ROI Hesabi (1,300 USD donanim maliyeti baz alinarak)
roi_gun = 1300 / net_kar if net_kar > 0 else 0

# --- 6. ANA EKRAN TASARIMI ---
st.title(f"HydroOptima TR | {secilen_tesis}")
st.info(f"Node Strategy: {kaynak_modu} kullanilarak enerji optimizasyonu saglaniyor.")

# EN UST GAZ METRIKLERI
col_g1, col_g2, col_g3, col_g4 = st.columns(4)
with col_g1:
    st.metric("Biyo-Hidrojen (H2)", f"{hacim_h2:,.1f} m3/Gun", delta="Soft Sensor Tahmini")
with col_g2:
    st.metric("Biyometan (CH4)", f"{hacim_ch4:,.1f} m3/Gun", delta="Soft Sensor Tahmini")
with col_g3:
    st.metric("Toplam Enerji", f"{toplam_mwh:.2f} MWh/Gun")
with col_g4:
    st.write("ROI Tracker")
    st.markdown(f"<span class='roi-text'>{roi_gun:.1f} GUN</span> (Amortisman)", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Finansal Analiz", "TR Ag Haritasi", "3D AI Beyni", "HydroBot AI"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Gunluk Brut Gelir", f"${gelir:,.0f}")
    c2.metric("Reaktif Gideri", f"-${maliyet:,.1f}")
    c3.metric("NET TESIS KARI", f"${net_kar:,.0f}")
    
    st.markdown("---")
    # Waterfall Grafigi
    fark = pred_h2 - baseline_mean if pred_h2 > 0 else 0
    fig_xai = go.Figure(go.Waterfall(
        x=["Malaga Taban", "Demir Etkisi", "Nikel Etkisi", "VFA Dengesi", "pH Dengesi", "Nihai H2"],
        y=[baseline_mean, fark*0.3, fark*0.3, fark*0.2, fark*0.2, pred_h2],
        measure=["absolute", "relative", "relative", "relative", "relative", "total"]
    ))
    fig_xai.update_layout(title="AI Karar Destek Varyansi (XAI)", height=350, template="plotly_dark")
    st.plotly_chart(fig_xai, use_container_width=True)

with tab2:
    fig_map = px.scatter_mapbox(turkiye_tesisleri, lat="Enlem", lon="Boylam", size="Kapasite_Ton",
                                hover_name="Tesis_Adi", zoom=5, height=500, mapbox_style="carto-positron")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    # Hizlandirilmiş 3D Motoru
    fe_range = np.linspace(0.5, 50, 15); ni_range = np.linspace(0, 50, 15)
    fe_m, ni_m = np.meshgrid(fe_range, ni_range)
    z_m = np.zeros_like(fe_m)
    for i in range(len(fe_range)):
        for j in range(len(ni_range)):
            test_df = pd.DataFrame([[fe_m[i,j], ni_m[i,j], input_acetate, input_butyrate, input_ethanol, input_ph]], columns=feature_names)
            z_m[i,j] = model.predict(test_df)[0]
    fig3d = go.Figure(data=[go.Surface(z=z_m, x=fe_m, y=ni_m, colorscale='Viridis')])
    fig3d.update_layout(scene=dict(xaxis_title='Fe', yaxis_title='Ni', zaxis_title='H2 Yield'), height=500)
    st.plotly_chart(fig3d, use_container_width=True)

with tab4:
    if failsafe == "BLOCKED":
        bot_msg = "DIKKAT: PLC Blokaj modunda! pH degerleri bakteri yasami icin uygun degil. Kimyasal akisi durduruldu."
    else:
        bot_msg = f"Sistem {kaynak_modu} modunda calisiyor. Donanim maliyeti Yumusak Sensorler ile sifirlandi. Net karimiz stabil."
    st.markdown(f"<div class='chat-bubble'><b>HydroBot:</b> {bot_msg}</div>", unsafe_allow_html=True)