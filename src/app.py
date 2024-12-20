import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Título de la aplicación
st.title("Predicción de Demanda de Taxis")

# Explicación breve
st.write("""
Esta aplicación utiliza un modelo de Machine Learning entrenado para predecir la demanda de taxis. 
Introduce los parámetros deseados y obtén una predicción en tiempo real.
""")

# Cargar el modelo entrenado
model = joblib.load("src/xgboost_model.pkl")

# Sección de entrada de datos
st.sidebar.header("Parámetros de predicción")
hour = st.sidebar.slider("Hora del día (00:00 - 24:00)", 0, 24, 12)  # Selector de hora
day_of_week = st.sidebar.selectbox(
    "Día de la semana (1 = Lunes, 7 = Domingo)",
    ["1 (Lunes)", "2 (Martes)", "3 (Miércoles)", "4 (Jueves)", "5 (Viernes)", "6 (Sábado)", "7 (Domingo)"]
)
day_of_week_map = {"1 (Lunes)": 1, "2 (Martes)": 2, "3 (Miércoles)": 3, "4 (Jueves)": 4, "5 (Viernes)": 5, "6 (Sábado)": 6, "7 (Domingo)": 7}

# Crear un DataFrame con los datos ingresados
input_data = pd.DataFrame({
    "Hora": [hour],
    "Día de la semana": [day_of_week_map[day_of_week]]
})

# Mostrar los parámetros ingresados
st.write("**Parámetros ingresados:**")
st.dataframe(input_data)  # Muestra la tabla

# Generar la predicción
prediction = model.predict(input_data.rename(columns={"Hora": "hour", "Día de la semana": "dayofweek"}))[0]

# Mostrar la predicción
st.subheader("Demanda estimada")
st.write(f"**{int(prediction)} pedidos de taxis**")

# Generar datos para el gráfico usando el modelo
st.subheader("Demanda promedio por hora según el día seleccionado")

# Crear un DataFrame con todas las horas del día para el día seleccionado
hours = list(range(24))
df_chart = pd.DataFrame({
    "Hora": hours,
    "Día de la semana": [day_of_week_map[day_of_week]] * 24
})

# Predecir la demanda para cada hora usando el modelo y redondear
df_chart["Pedidos promedio"] = model.predict(
    df_chart.rename(columns={"Hora": "hour", "Día de la semana": "dayofweek"})
).round().astype(int)  # Redondear a números enteros

# Crear el gráfico con Plotly
fig = px.bar(
    df_chart,
    x="Hora",
    y="Pedidos promedio",
    title="Demanda promedio por hora",
    labels={"Hora": "Hora del día", "Pedidos promedio": "Número promedio de pedidos"},
    text_auto=True
)

# Mejorar diseño del gráfico
fig.update_layout(
    xaxis=dict(tickmode="linear", dtick=1),  # Etiquetas de hora cada 1
    yaxis=dict(range=[0, max(df_chart["Pedidos promedio"]) + 10]),  # Ajustar el rango del eje Y
    title_font_size=18,
    font=dict(size=14),
    plot_bgcolor="#f9f9f9",  # Fondo claro
    showlegend=False
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)


# Explicación del modelo
st.write("""
### ¿Cómo funciona este modelo?
El modelo fue entrenado con datos históricos de demanda de taxis, utilizando la hora del día y el día de la semana como variables clave. 
Su precisión se mide con un RMSE de 34.64, cumpliendo con los criterios del proyecto.

Sweet Lift Taxi ha recopilado datos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, 
necesitamos predecir la cantidad de pedidos de taxis para la próxima hora.

Esta aplicación ayuda a Sweet Lift Taxi a planificar de manera eficiente la asignación de conductores, mejorando la experiencia de 
los clientes y maximizando las ganancias de la empresa.
""")

