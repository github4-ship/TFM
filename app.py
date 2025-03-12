import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from fpdf import FPDF
import streamlit_authenticator as stauth
from backend import cargar_datos, entrenar_modelo

# --------- AUTENTICACIÓN SEGURA (Corregido definitivo) ----------
credentials = {
    "usernames": {
        "Aithor": {
            "name": "Aithor",
            "password": "$2b$12$j/.wfm4K8jXKVD0URWruZO63P3MZMfGxMibgsW3hLLsC1z3C99Hpe"

        }
    }
}


authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="cookie_abp",
    key="signature_key_abp",
    cookie_expiry_days=30
)

authenticator.login(location='sidebar')

if st.session_state["authentication_status"] is False:
    st.error('❌ Usuario o contraseña incorrectos.')
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.warning('⚠️ Por favor introduce usuario y contraseña.')
    st.stop()
else:
    authenticator.logout('Cerrar sesión', 'sidebar')
    st.sidebar.write(f'👋 Bienvenido/a, {st.session_state["name"]}')

# ----------------- Menú Navegación ------------------------
menu = st.sidebar.radio("Menú", ["🏠 Home", "📊 Estadísticas"])

# ----------------- Cargar datos usando backend ----------------
df_players, df_teams = cargar_datos("Big5Leagues_Jugadores.csv", "Big5Leagues_Equipos.csv")

if menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Bienvenido al Análisis ABP con Machine Learning.")
    
elif menu == "📊 Estadísticas":
    st.title("📊 Estadísticas Avanzadas ABP")

    posiciones = st.sidebar.multiselect("⚽ Posiciones", df_players["Pos"].unique(), default=df_players["Pos"].unique())
    minutos = st.sidebar.slider("⏱️ Minutos jugados (mínimos)", 0, 4000, 500)
    jugador_busqueda = st.sidebar.text_input('🔍 Buscar jugador')

    df_filtrado = df_players[(df_players["Min"] >= minutos) & (df_players["Pos"].isin(posiciones))]
    if jugador_busqueda:
        df_filtrado = df_filtrado[df_filtrado["Player"].str.contains(jugador_busqueda, case=False)]

    jugador = st.selectbox("Selecciona jugador", df_filtrado["Player"].unique())
    datos_jugador = df_filtrado[df_filtrado["Player"] == jugador]

    st.write("📈 **Estadísticas del jugador seleccionado:**", datos_jugador)

    modelo, matriz_confusion = entrenar_modelo(df_filtrado)

    st.write(f'🎯 Precisión del modelo: {modelo.best_score_:.2f}')

    fig_scatter = px.scatter(df_filtrado, x="xG", y="Gls", color="Player", title="Relación xG vs Goles")
    st.plotly_chart(fig_scatter)

    radar_metrics = ['xG', 'xAG', 'PrgP']
    valores_radar = datos_jugador[radar_metrics].iloc[0].tolist()

    fig_radar = go.Figure(go.Scatterpolar(
        r=valores_radar,
        theta=radar_metrics,
        fill='toself',
        name=jugador
    ))
    st.plotly_chart(fig_radar)

    fig_heatmap = go.Figure(go.Heatmap(
        z=matriz_confusion,
        x=['No Éxito ABP', 'Éxito ABP'],
        y=['Real No éxito ABP', 'Real Éxito ABP'],
        colorscale='Blues',
        text=matriz_confusion,
        texttemplate="%{text}"
    ))
    fig_heatmap.update_layout(title="📌 Matriz de Confusión")
    st.plotly_chart(fig_heatmap)

    # ----------------- Aplicación de filtros ----------------------
df_filtrado = df_players[(df_players["Min"] >= minutos) & (df_players["Pos"].isin(posiciones))]
if jugador_busqueda:
    df_filtrado = df_filtrado[df_filtrado["Player"].str.contains(jugador_busqueda, case=False)]

# Validación clave para prevenir errores:
if df_filtrado.shape[0] < 10:
    st.warning("⚠️ Hay muy pocos jugadores después de aplicar filtros. Modifica los filtros para mostrar más jugadores.")
    st.stop()

jugador = st.selectbox("Selecciona jugador", df_filtrado["Player"].unique())
datos_jugador = df_filtrado[df_filtrado["Player"] == jugador]

st.write("📈 **Estadísticas del jugador seleccionado:**", datos_jugador)

# Intentar entrenar modelo únicamente si hay datos suficientes
try:
    modelo, matriz_confusion = entrenar_modelo(df_filtrado)
    st.write(f'🎯 Precisión del modelo: {modelo.best_score_:.2f}')
except ValueError:
    st.error("❌ No se puede entrenar el modelo porque hay muy pocos datos después de filtrar. Ajusta los filtros nuevamente.")
    st.stop()


    # ----------------- Exportación PDF ----------------------
    if st.button("📄 Exportar Informe en PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Reporte ABP - {jugador}", ln=True, align='C')

        for metrica, valor in zip(radar_metrics, valores_radar):
            pdf.cell(200, 10, txt=f"{metrica}: {valor}", ln=True)

        pdf.output(f"Reporte_{jugador}.pdf")
        st.success("✅ Reporte PDF generado con éxito.")
