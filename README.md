# Predicción de Demanda de Taxis - Sweet Lift Taxi

Esta aplicación predice la demanda de taxis en función de la hora del día y el día de la semana. Fue diseñada para **Sweet Lift Taxi**, utilizando datos históricos recopilados en aeropuertos. El objetivo es ayudar a la compañía a planificar eficientemente la asignación de conductores durante las horas pico.

## Descripción

La aplicación utiliza un modelo de Machine Learning (**XGBoost**) entrenado con datos históricos para predecir la cantidad de pedidos de taxis para una hora específica. También incluye un gráfico interactivo que muestra la demanda promedio estimada para todas las horas del día seleccionado.

## Características

- **Predicciones en tiempo real:** Introduce la hora y el día de la semana para obtener la predicción.
- **Gráfico interactivo:** Visualiza la demanda promedio por hora según el día de la semana.
- **Optimización para Sweet Lift Taxi:** Ayuda a planificar mejor las asignaciones de conductores, mejorando la experiencia de los clientes y maximizando los ingresos.

## Librerías y herramientas utilizadas

- **Python 3.9+**
- **Streamlit**: Para crear la interfaz interactiva.
- **XGBoost**: Algoritmo de aprendizaje automático para el modelo de predicción.
- **Pandas**: Manipulación y procesamiento de datos.
- **NumPy**: Operaciones matemáticas y análisis numérico.
- **Plotly**: Visualización interactiva de gráficos.
- **Scikit-learn**: Para dividir los datos en conjuntos de entrenamiento y prueba.
- **Render.com**: Para el despliegue en la web.
