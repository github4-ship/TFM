# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend import cargar_datos, entrenar_modelo
from fpdf import FPDF
import streamlit_authenticator as stauth

# -------- AUTENTICACIÓN SEGURA (definitivo) --------
credentials = {
    "usernames": {
        "Aithor": {
            "name": "Aithor",
            "password": "$2b$12$hash_generado_con_bcrypt"
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

st.sidebar.write(f'👋 Bienvenido/a, {st.session_state["name"]}')
authenticator.logout('Cerrar sesión', 'sidebar')

# -------- MENÚ NAVEGACIÓN --------
menu = st.sidebar.radio("Menú", ["🏠 Home", "📊 Estadísticas"])

# -------- Cargar datos usando backend (definitivo corregido) --------
df_players, df_teams = cargar_datos("Big5Leagues_Jugadores.csv", "Big5Leagues_Equipos.csv")

if menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Bienvenido al análisis interactivo de balón parado usando Machine Learning.")

elif menu == "📊 Estadísticas":
    st.title("📊 Estadísticas avanzadas ABP")

    # -------- FILTROS INTERACTIVOS DEFINITIVOS --------
    posiciones = st.sidebar.multiselect("⚽ Posiciones", df_players["Pos"].unique(), default=df_players["Pos"].unique())
    minutos = st.sidebar.slider("⏱️ Minutos jugados (mínimos)", 0, 4000, 500)
    jugador_busqueda = st.sidebar.text_input('🔍 Buscar jugador')

    df_filtrado = df_players[
        (df_players["Min"] >= minutos) &
        (df_players["Pos"].isin(posiciones))
    ]

    if jugador_busqueda:
        df_filtrado = df_filtrado[df_filtrado["Player"].str.contains(jugador_busqueda, case=False)]

    # -------- PREVENCIÓN DE ERROR (definitivo) --------
    if df_filtrado.shape[0] < 10:
        st.warning("⚠️ Hay muy pocos jugadores después de aplicar filtros. Modifica los filtros para mostrar más jugadores.")
        st.stop()

    jugador = st.selectbox("Selecciona jugador", df_filtrado["Player"].unique())
    datos_jugador = df_filtrado[df_filtrado["Player"] == jugador]

    st.write("📈 **Estadísticas del jugador seleccionado:**", datos_jugador)

    # -------- ENTRENAR MODELO CON MANEJO DE ERRORES --------
    try:
        modelo, matriz_confusion, accuracy = entrenar_modelo(df_filtrado)
        st.write(f'🎯 Precisión del modelo: {accuracy:.2f}')
    except ValueError:
        st.error("❌ No se puede entrenar el modelo. Ajusta los filtros.")
        st.stop()

    # -------- Visualizaciones específicas --------
    fig_scatter = px.scatter(df_filtrado, x="xG", y="Gls", color="Player", title="Relación xG vs Goles")
    st.plotly_chart(fig_scatter)

    # -------- GRÁFICO RADAR --------
    radar_metrics = ['xG', 'xAG', 'PrgP']
    valores_radar = datos_jugador[radar_metrics].iloc[0].tolist()

    fig_radar = go.Figure(go.Scatterpolar(
        r=valores_radar,
        theta=radar_metrics,
        fill='toself',
        name=jugador
    ))
    fig_radar.update_layout(title=f"📌 Radar de métricas clave para {jugador}")
    st.plotly_chart(fig_radar)

    # -------- MATRIZ DE CONFUSIÓN --------
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

    # -------- EXPORTAR A PDF --------
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

elif menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Bienvenido/a al análisis interactivo de balón parado usando Machine Learning.")

