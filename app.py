# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from fpdf import FPDF
import streamlit_authenticator as stauth
from backend import cargar_datos

# --------- AUTENTICACIÓN SEGURA ----------
import streamlit as st
import streamlit_authenticator as stauth

# Credenciales (modifica estos valores según tu preferencia)
names = ['Usuario Demo']
usernames = ['usuario']
passwords = ['password123']  # Cámbialo por tu contraseña real

# Genera contraseñas cifradas (hazlo previamente y pega el resultado aquí directamente)
hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    'cookie_abp', 'signature_key_abp', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('🔒 Login', 'main')

if authentication_status == False:
    st.error('❌ Usuario o contraseña incorrectos.')
    st.stop()
elif authentication_status == None:
    st.warning('⚠️ Por favor, introduce usuario y contraseña.')
    st.stop()
else:
    authenticator.logout('Cerrar sesión', 'sidebar')
    st.sidebar.write(f'👋 Bienvenido/a, {name}')


# Menú principal
menu = st.sidebar.radio("Menú", ["🏠 Home", "📊 Estadísticas"])

df_players, df_teams = cargar_datos("data/Big5Leagues_Jugadores.csv")

if menu == "🏠 Home":
    st.title("🏠 Home")
    st.write("Bienvenido al Análisis ABP con Machine Learning.")
    
elif menu == "📊 Estadísticas":
    st.title("📊 Estadísticas Avanzadas ABP")

    # Filtros
    posiciones = st.sidebar.multiselect("Posiciones", df_players["Pos"].unique())
    min_minutos = st.sidebar.slider('Minutos mínimos', 0, 3000, 500)
    jugador_busqueda = st.sidebar.text_input('Buscar jugador')

    df_filtrado = df_players[(df_players["Min"] >= min_minutos) & (df_players["Pos"].isin(posiciones))]
    if jugador_busqueda:
        df_filtrado = df_filtrado[df_filtrado["Player"].str.contains(jugador_busqueda, case=False)]

    jugador = st.selectbox("Jugador:", df_filtrado["Player"].unique())

    st.write(df_filtrado[df_filtrado["Player"]==jugador])

    modelo = joblib.load('ml_model.pkl')

    # Visualizaciones
    fig_scatter = px.scatter(df_filtrado, x="xG", y="Gls", color="Player")
    st.plotly_chart(fig_scatter)

    # Radar plot
    jugador_seleccionado = df_filtrado[df_filtrado["Player"] == jugador].iloc[0]
    fig_radar = go.Figure(go.Scatterpolar(
        r=[jugador['xG'],jugador['xAG'],jugador['PrgP']],
        theta=['xG','xAG','PrgP'],
        fill='toself'
    ))
    st.plotly_chart(fig_radar)

    # Exportar PDF
    if st.button("Exportar PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,f"Informe de {jugador}",ln=True,align='C')
        pdf.output(f"Informe_{jugador}.pdf")
        st.success("✅ PDF generado correctamente.")

