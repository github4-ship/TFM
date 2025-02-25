import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Cargar datos
def load_data():
    df_players = pd.read_csv("/content/Big5Leagues_Jugadores.csv")
    df_teams = pd.read_csv("/content/Big5Leagues_Equipos.csv")

    # Convertir columnas a numérico
    numeric_columns = ['Gls', 'xG', 'xAG', 'PrgP']
    for col in numeric_columns:
        df_players[col] = pd.to_numeric(df_players[col], errors='coerce')

    return df_players, df_teams

df_players, df_teams = load_data()

# Seleccionar variables correctas
selected_features = ['xG', 'xAG', 'PrgP']
for feature in selected_features:
    if feature not in df_players.columns:
        raise KeyError(f"Columna '{feature}' no encontrada en df_players. Verifica los datos de scraping.")

# Entrenar modelo de Machine Learning
def train_model():
    X = df_players[selected_features]
    y = (df_players['Gls'] > 2).astype(int)  # Clasificación binaria basada en goles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    joblib.dump(model, "ml_model.pkl")

    return accuracy, report, conf_matrix, model

accuracy, report, conf_matrix, model = train_model()

# Configuración de la aplicación Streamlit
st.title("📊 Análisis de Balón Parado con Machine Learning")
st.sidebar.header("Filtros")

# Selección de jugadores y equipos
player_selected = st.sidebar.selectbox("Selecciona un jugador", df_players["Player"].unique())
team_selected = st.sidebar.selectbox("Selecciona un equipo", df_teams["Equipo"].unique())

df_filtered_players = df_players[df_players["Player"] == player_selected]
df_filtered_teams = df_teams[df_teams["Equipo"] == team_selected]

st.write("### Estadísticas del jugador seleccionado")
st.write(df_filtered_players)

st.write("### Estadísticas del equipo seleccionado")
st.write(df_filtered_teams)

st.write("### Relación entre xG y Goles")
fig = px.scatter(df_players, x="xG", y="Gls", color="Player", size_max=10)
st.plotly_chart(fig)

st.write("### Predicción de Éxito en ABP")
xg_input = st.number_input("xG del jugador", min_value=0.0, value=1.0, step=0.1)
xag_input = st.number_input("xAG del jugador", min_value=0.0, value=1.0, step=0.1)
prgp_input = st.number_input("Pases Progresivos", min_value=0, value=5)

if st.button("Predecir éxito en ABP"):
    model = joblib.load("ml_model.pkl")
    pred = model.predict([[xg_input, xag_input, prgp_input]])
    resultado = "Alto Éxito" if pred[0] == 1 else "Bajo Éxito"
    st.write(f"### 📌 Resultado: {resultado}")

st.write("### Matriz de Confusión del Modelo")
fig_conf = go.Figure(data=go.Heatmap(z=conf_matrix, x=['No ABP', 'ABP'], y=['No ABP', 'ABP'], colorscale='Blues', text=conf_matrix, texttemplate="%{text}"))
st.plotly_chart(fig_conf)

st.write("### Exportar Reporte a PDF")
if st.button("Generar PDF"):
    import fpdf
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Reporte de Análisis de Balón Parado - {player_selected}", ln=True, align='C')
    pdf.output("Reporte_ABP.pdf")
    st.success("📄 Reporte generado correctamente.")
