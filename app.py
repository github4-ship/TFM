import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 📌 Verificar si los archivos existen antes de cargarlos
if not os.path.exists("Big5Leagues_Jugadores.csv") or not os.path.exists("Big5Leagues_Equipos.csv"):
    st.error("❌ No se encontraron los archivos CSV. Sube 'Big5Leagues_Jugadores.csv' y 'Big5Leagues_Equipos.csv' a GitHub en la misma carpeta que 'app.py'.")
    st.stop()

# 📌 Cargar los datos
df_players = pd.read_csv("Big5Leagues_Jugadores.csv")
df_teams = pd.read_csv("Big5Leagues_Equipos.csv")

# 📌 Convertir columnas a numérico
numeric_columns = ['Gls', 'xG', 'xAG', 'PrgP']
for col in numeric_columns:
    df_players[col] = pd.to_numeric(df_players[col], errors='coerce')

# 📌 Seleccionar variables para el modelo
selected_features = ['xG', 'xAG', 'PrgP']
for feature in selected_features:
    if feature not in df_players.columns:
        st.error(f"❌ Columna '{feature}' no encontrada en df_players. Verifica los datos de scraping.")
        st.stop()

# 📌 Entrenar modelo de Machine Learning
def train_model():
    X = df_players[selected_features]
    y = (df_players['Gls'] > 2).astype(int)  # Clasificación binaria basada en goles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    joblib.dump(model, "ml_model.pkl")

    return accuracy, report, conf_matrix, model

accuracy, report, conf_matrix, model = train_model()

# 📌 Configuración de la aplicación Streamlit
st.title("📊 Análisis de Balón Parado con Machine Learning")
st.sidebar.header("Filtros")

# 📌 Selección de jugadores y equipos
player_selected = st.sidebar.selectbox("Selecciona un jugador", df_players["Player"].unique())
team_selected = st.sidebar.selectbox("Selecciona un equipo", df_teams["Equipo"].unique())

df_filtered_players = df_players[df_players["Player"] == player_selected]
df_filtered_teams = df_teams[df_teams["Equipo"] == team_selected]

st.write("### 📊 Estadísticas del jugador seleccionado")
st.dataframe(df_filtered_players)

st.write("### 📊 Estadísticas del equipo seleccionado")
st.dataframe(df_filtered_teams)

# 📌 Gráfico de relación entre xG y Goles
st.write("### ⚽ Relación entre xG y Goles")
fig = px.scatter(df_players, x="xG", y="Gls", color="Player", size_max=10, title="Relación entre xG y Goles")
st.plotly_chart(fig)

# 📌 Predicción con el modelo de Machine Learning
st.write("### 🏆 Predicción de Éxito en ABP")
xg_input = st.number_input("xG del jugador", min_value=0.0, value=1.0, step=0.1)
xag_input = st.number_input("xAG del jugador", min_value=0.0, value=1.0, step=0.1)
prgp_input = st.number_input("Pases Progresivos", min_value=0, value=5)

if st.button("Predecir éxito en ABP"):
    model = joblib.load("ml_model.pkl")
    pred = model.predict([[xg_input, xag_input, prgp_input]])
    resultado = "Alto Éxito" if pred[0] == 1 else "Bajo Éxito"
    st.success(f"📌 Resultado: {resultado}")

# 📌 Matriz de Confusión del Modelo
st.write("### 🔥 Matriz de Confusión del Modelo")
fig_conf = go.Figure(data=go.Heatmap(
    z=conf_matrix, 
    x=['No ABP', 'ABP'], 
    y=['No ABP', 'ABP'], 
    colorscale='Blues', 
    text=conf_matrix, 
    texttemplate="%{text}"
))
st.plotly_chart(fig_conf)

# 📌 Exportar Reporte a PDF
st.write("### 📄 Exportar Reporte a PDF")
if st.button("Generar PDF"):
    import fpdf
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Reporte de Análisis de Balón Parado - {player_selected}", ln=True, align='C')
    pdf.output("Reporte_ABP.pdf")
    st.success("📄 Reporte generado correctamente.")


import plotly.graph_objects as go

# 📌 Selección del jugador para el radar
st.write("### 🛡️ Análisis de Desempeño del Jugador")
player_radar = st.selectbox("Selecciona un jugador para el radar", df_players["Player"].unique())

# 📌 Filtrar los datos del jugador seleccionado
df_player_radar = df_players[df_players["Player"] == player_radar]

if df_player_radar.empty:
    st.warning("No se encontraron datos para el jugador seleccionado.")
else:
    # 📌 Seleccionar métricas para el radar
    radar_metrics = ["Gls", "xG", "xAG", "PrgP", "PrgC", "PrgR"]
    
    # 📌 Extraer valores
    values = df_player_radar[radar_metrics].values.flatten().tolist()
    
    # 📌 Agregar el primer valor al final para cerrar el gráfico
    values.append(values[0])

    # 📌 Crear gráfico de radar con Plotly
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_metrics + [radar_metrics[0]],
        fill='toself',
        name=player_radar
    ))

    # 📌 Configurar diseño del gráfico
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.2])
        ),
        showlegend=True,
        title=f"📊 Radar de Desempeño - {player_radar}"
    )

    # 📌 Mostrar gráfico en Streamlit
    st.plotly_chart(fig_radar)

