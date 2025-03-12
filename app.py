# ------------------ Librerías ------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from fpdf import FPDF
import streamlit_authenticator as stauth
from backend import cargar_datos, entrenar_modelo

# --------- AUTENTICACIÓN SEGURA (Corregido definitivamente) ----------
import streamlit as st
import streamlit_authenticator as stauth

# Credenciales con hash generado previamente (debes haberlo generado antes con stauth_hasher().py)
credentials = {
    "usernames": {
        "usuario": {
            "name": "Usuario Demo",
            "password": "$2b$12$AquiPonTuHashGeneradoCorrectamente"
        }
    }
    
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="cookie_abp",
    key="signature_key_abp",
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('🔒 Login', 'main')

if authentication_status == False:
    st.error('❌ Usuario o contraseña incorrectos.')
    st.stop()
elif authentication_status == None:
    st.warning('⚠️ Por favor introduce tu usuario y contraseña.')
    st.stop()
else:
    authenticator.logout('Cerrar sesión', 'sidebar')
    st.sidebar.write(f'👋 Bienvenido/a *{name}*')



# ----------------- Menú Navegación ------------------------
menu = st.sidebar.radio("Menú", ["🏠 Home", "📊 Estadísticas"])

# ----------------- Cargar datos usando backend ----------------
df_players, df_teams = cargar_datos("Big5Leagues_Jugadores.csv")

if menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Bienvenido/a al análisis interactivo de balón parado usando Machine Learning.")
    
elif menu == "📊 Estadísticas":
    st.title("📊 Estadísticas avanzadas")

    # Filtros interactivos obligatorios
    posiciones = st.sidebar.multiselect("⚽ Posiciones", df_players["Pos"].unique(), default=df_players["Pos"].unique())
    minutos = st.sidebar.slider("⏱️ Minutos jugados (mínimos)", 0, 4000, 500)
    jugador_busqueda = st.sidebar.text_input("🔍 Buscar jugador")

    # Aplicación de filtros
    df_filtrado = df_players[(df_players["Min"] >= minutos) & (df_players["Pos"].isin(posiciones))]
    if jugador_busqueda:
        df_filtrado = df_filtrado[df_filtrado["Player"].str.contains(jugador_busqueda, case=False)]

    jugador = st.selectbox("Selecciona jugador", df_filtrado["Player"].unique())
    datos_jugador = df_filtrado[df_filtrado["Player"] == jugador]

    st.write("📈 **Estadísticas del jugador:**", datos_jugador)

    # ----------------- Modelo Machine Learning ----------------------
    modelo, matriz_confusion = entrenar_modelo(df_filtrado)

    st.write(f'🎯 Precisión del modelo: {modelo.best_score_:.2f}')

    # ----------------- Visualizaciones específicas ----------------------
    # Scatterplot interactivo
    fig_scatter = px.scatter(df_filtrado, x="xG", y="Gls", color="Player", title="Relación xG vs Goles")
    st.plotly_chart(fig_scatter)

    # Gráfico Radar
    jugador_seleccionado = df_filtrado[df_filtrado["Player"] == jugador].iloc[0]
    radar_metrics = ['xG', 'xAG', 'PrgP']
    valores_radar = jugador_seleccionado[radar_metrics].tolist()

    fig_radar = go.Figure(go.Scatterpolar(
        r=valores_radar,
        theta=radar_metrics,
        fill='toself',
        name=jugador
    ))
    st.plotly_chart(fig_radar)

    # ----------------- Matriz de confusión (Heatmap) ----------------------
    fig_heatmap = go.Figure(go.Heatmap(
        z=matriz_confusion,
        x=['No Éxito ABP', 'Éxito ABP'],
        y=['Real No Éxito ABP', 'Real Éxito ABP'],
        colorscale='Blues',
        text=matriz_confusion,
        texttemplate="%{text}"
    ))
    fig_heatmap.update_layout(title="📌 Matriz de Confusión")
    st.plotly_chart(fig_heatmap)

    # ----------------- Exportación PDF ----------------------
    if st.button("📄 Exportar Informe en PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt=f"Reporte ABP - {jugador}", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        for metrica, valor in zip(radar_metrics, valores_radar):
            pdf.cell(200, 10, txt=f"{metrica}: {valor}", ln=True)

        pdf.output(f"Reporte_{jugador}.pdf")
        st.success("✅ Reporte PDF generado con éxito.")
