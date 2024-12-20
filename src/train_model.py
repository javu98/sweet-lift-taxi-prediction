# Importación de librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Función para cargar y procesar los datos
def load_and_prepare_data(file_path):
    """
    Carga y prepara los datos de taxis para el modelo.
    
    Parámetros:
        file_path (str): Ruta al archivo CSV con los datos de taxis.
        
    Retorna:
        pd.DataFrame: Datos procesados y listos.
    """
    # Cargar los datos
    data = pd.read_csv(file_path, index_col=[0], parse_dates=[0])

    # Verificar si las fechas están en orden cronológico
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    # Remuestreo por hora (suma de pedidos)
    data = data.resample('1H').sum()

    # Rellenar valores faltantes con 0 (si existen)
    data = data.fillna(0)

    # Extraer características adicionales
    data['hour'] = data.index.hour  # Hora del día
    data['dayofweek'] = data.index.dayofweek  # Día de la semana

    return data

# Función para entrenar el modelo XGBoost
def train_model(data):
    """
    Entrena un modelo XGBoost para predecir la demanda de taxis.
    
    Parámetros:
        data (pd.DataFrame): Datos procesados con características.
        
    Retorna:
        XGBRegressor: Modelo entrenado.
    """
    # Definir las características (X) y la variable objetivo (y)
    X = data[['hour', 'dayofweek']]
    y = data['num_orders']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Crear y entrenar el modelo XGBoost
    model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE en el conjunto de prueba: {rmse}")

    return model

# Script principal
if __name__ == "__main__":
    # Ruta al archivo de datos
    file_path = "taxi.csv"

    # Cargar y preparar los datos
    data = load_and_prepare_data(file_path)

    # Entrenar el modelo
    model = train_model(data)

    # Guardar el modelo entrenado
    joblib.dump(model, "src/xgboost_model.pkl")
    print("Modelo guardado en 'src/xgboost_model.pkl'")
