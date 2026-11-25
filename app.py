"""Apartado 5.2 - Entorno productivo controlado de BPA.

Reutiliza el Pipeline del modelo F del notebook del grupo 2.
El archivo modelo_f_jorge_chavez.joblib debe exportarse desde Colab.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pronóstico de pasajeros | Jorge Chávez",
    page_icon="✈️",
    layout="centered",
)

MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

st.title("✈️ Pronóstico mensual de pasajeros")
st.subheader("Aeropuerto Internacional Jorge Chávez")
st.write(
    "Salidas nacionales. Modelo F de regresión lineal "
    "con el número de pasajeros del mes anterior y el mes del calendario."
)

modelo_archivo = Path(__file__).with_name("modelo_f_jorge_chavez.joblib")
if not modelo_archivo.exists():
    st.error(
        "No se encontró 'modelo_f_jorge_chavez.joblib'. "
        "Exporta el Pipeline 'modelo_final' desde tu notebook de Colab "
        "y coloca el archivo junto a 'app.py'."
    )
    st.stop()

@st.cache_resource
def cargar_modelo():
    # Este archivo debe provenir exclusivamente del notebook del grupo.
    return joblib.load(modelo_archivo)

modelo_final = cargar_modelo()

st.markdown("### 1. Ingresa el último mes con datos reales")
col_anio, col_mes = st.columns(2)
with col_anio:
    anio_anterior = st.number_input(
        "Año observado", min_value=2009, max_value=2100,
        value=2024, step=1, format="%d",
    )
with col_mes:
    mes_anterior = st.selectbox(
        "Mes observado", MESES, index=11,
    )

pasajeros_anterior = st.number_input(
    "Cantidad real de pasajeros de ese mes",
    min_value=0, value=643137, step=1, format="%d",
)

numero_mes_anterior = MESES.index(mes_anterior) + 1
numero_mes_siguiente = numero_mes_anterior % 12 + 1
anio_siguiente = int(anio_anterior) + (1 if numero_mes_anterior == 12 else 0)
nombre_mes_siguiente = MESES[numero_mes_siguiente - 1]

st.markdown("### 2. Pronóstico a un mes (t+1)")
st.info(f"El modelo pronosticará **{nombre_mes_siguiente} de {anio_siguiente}**.")

if st.button("Calcular pronóstico", type="primary", use_container_width=True):
    entrada = pd.DataFrame({
        "lag_1": [int(pasajeros_anterior)],
        "mes_num": [numero_mes_siguiente],
    })

    prediccion = float(modelo_final.predict(entrada)[0])
    if prediccion < 0:
        st.warning(
            "El modelo produjo una predicción negativa. "
            "No debe interpretarse como una cantidad válida de pasajeros."
        )
    else:
        st.metric(
            label=f"Pasajeros pronosticados para {nombre_mes_siguiente} de {anio_siguiente}",
            value=f"{round(prediccion):,}".replace(",", " "),
        )
        st.caption(
            "Resultado estimado por el modelo F entrenado. "
            "No representa una cantidad real observada."
        )

with st.expander("Metodología y limitaciones"):
    st.write(
        "El modelo se entrenó con observaciones de enero de 2010 "
        "a diciembre de 2024. Usa exclusivamente las variables 'lag_1' "
        "y 'mes_num'. El último valor observado debe corresponder "
        "al mes anterior al pronóstico. No se reentrena al consultar."
    )
    st.write(
        "Evaluación de doce pronósticos mensuales sucesivos en TEST 2025: "
        "RMSE ≈ 34 820 pasajeros, MAE ≈ 27 887 pasajeros "
        "y R² ≈ 0,2376. Estas métricas no son un porcentaje de aciertos."
    )

st.caption(
    "Trabajo Final de Business Predictive Analytics · Grupo 2 · "
    "Apartado 5.2: demostración en entorno productivo controlado."
)
