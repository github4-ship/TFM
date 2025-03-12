# -*- coding: utf-8 -*-
"""backend.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NaUKcCq0Suk_uYKHC1TGVCD5TDf9tod0
"""

# backend.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib

def cargar_datos(path_jugadores, path_equipos):
    df_players = pd.read_csv(path_jugadores)
    df_teams = pd.read_csv(path_equipos)
    
    # Conversión explícita a formato numérico (clave para solucionar error)
    df_players['Min'] = pd.to_numeric(df_players['Min'], errors='coerce').fillna(0)

    # Limpieza básica
    df_players.fillna(0, inplace=True)
    df_teams.fillna(0, inplace=True)

    return df_players, df_teams

def entrenar_modelo(df):
    X = df[['xG', 'xAG', 'PrgP']]
    y = (df['Gls'] > 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100,200]}, cv=5)
    grid.fit(X_train, y_train)

    joblib.dump(grid, 'ml_model.pkl')
    return grid, confusion_matrix(y_test, grid.predict(X_test))
