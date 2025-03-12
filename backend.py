# backend.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib

def cargar_datos(path_jugadores, path_equipos):
    df_players = pd.read_csv(path_jugadores)
    df_teams = pd.read_csv(path_equipos)

    # Conversión explícita a formato numérico
    numeric_cols = ['Min', 'Gls', 'xG', 'xAG', 'PrgP']
    for col in numeric_cols:
        df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)

    return df_players, df_teams

def entrenar_modelo(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix, accuracy_score
    import joblib

    # Definir variables predictoras y objetivo
    X = df[['xG', 'xAG', 'PrgP']]
    y = (df['Gls'] > 2).astype(int)

    # Comprobar que hay suficientes datos
    if len(df) < 10:
        raise ValueError("Datos insuficientes después del filtrado para entrenar el modelo.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100,200]}, cv=5)
    grid.fit(X_train, y_train)

    # Guardar modelo
    joblib.dump(grid, 'ml_model.pkl')

    # Predicciones y matriz de confusión
    y_pred = grid.predict(X_test)
    matriz_confusion = confusion_matrix(y_test, y_pred)

    return grid, matriz_confusion, accuracy_score(y_test, grid.predict(X_test))
