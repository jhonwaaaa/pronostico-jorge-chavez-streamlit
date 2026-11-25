"""Pronostico t+1 de salidas nacionales del Aeropuerto Jorge Chavez.

Reutiliza sin reentrenar el Pipeline del modelo F del notebook.
Lee automaticamente el Excel original MTC (hojas 2009-2025).
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
CARPETA = Path(__file__).parent
RUTA_MODELO = CARPETA / "modelo_f_jorge_chavez.joblib"
RUTA_EXCEL = CARPETA / "pasajeros_jorge_chavez_2009_2025.xlsx"


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_serie_mensual():
    """Extrae los 12 meses de Jorge Chavez de cada hoja del Excel original."""
    registros = []
    with pd.ExcelFile(RUTA_EXCEL) as libro:
        for anio in range(2009, 2026):
            tabla = pd.read_excel(libro, sheet_name=str(anio), header=None)
            # En el Excel original el nombre del aeropuerto está en la columna B.
            coincide = tabla.iloc[:, 1].astype(str).str.contains(
                "Jorge Chávez", case=False, regex=False
            )
            filas = tabla.loc[coincide]
            if len(filas) != 1:
                raise ValueError(
                    f"No se encontró una única fila de Jorge Chávez en la hoja {anio}."
                )
            # Las cantidades de enero a diciembre están en las columnas D:O.
            cantidades = filas.iloc[0, 3:15].tolist()
            if len(cantidades) != 12 or any(pd.isna(n) for n in cantidades):
                raise ValueError(f"Hay datos mensuales faltantes en la hoja {anio}.")
            for numero_mes, cantidad in enumerate(cantidades, start=1):
                registros.append({
                    "anio": anio,
                    "mes_num": numero_mes,
                    "pasajeros": int(cantidad),
                })
    return pd.DataFrame(registros)


st.title("✈️ Pronóstico mensual de pasajeros")
st.subheader("Aeropuerto Internacional Jorge Chávez")
st.write(
    "Salidas nacionales · Modelo F de regresión lineal · "
    "Pronóstico a un mes (t+1)."
)

if not RUTA_MODELO.exists():
    st.error("Falta el archivo 'modelo_f_jorge_chavez.joblib' en GitHub.")
    st.stop()
if not RUTA_EXCEL.exists():
    st.error("Falta el archivo 'pasajeros_jorge_chavez_2009_2025.xlsx' en GitHub.")
    st.stop()

try:
    modelo_final = cargar_modelo()
    df_serie = cargar_serie_mensual()
except Exception as error:
    st.error(f"No se pudo cargar el modelo o los datos: {error}")
    st.stop()

st.markdown("### 1. Selecciona el mes que deseas pronosticar")
col_anio, col_mes = st.columns(2)
with col_anio:
    anio_pronostico = st.selectbox(
        "Año del pronóstico",
        options=list(range(2010, 2027)),
        index=15,  # 2025 por defecto
    )
with col_mes:
    # Solo enero de 2026: para febrero necesitaríamos enero de 2026 real.
    opciones_meses = MESES if anio_pronostico <= 2025 else ["Enero"]
    mes_pronostico = st.selectbox("Mes del pronóstico", opciones_meses)

mes_num_pronostico = MESES.index(mes_pronostico) + 1
mes_num_anterior = 12 if mes_num_pronostico == 1 else mes_num_pronostico - 1
anio_anterior = anio_pronostico - 1 if mes_num_pronostico == 1 else anio_pronostico
mes_anterior = MESES[mes_num_anterior - 1]

fila_previa = df_serie.loc[
    (df_serie["anio"] == anio_anterior)
    & (df_serie["mes_num"] == mes_num_anterior),
    "pasajeros",
]
if fila_previa.empty:
    st.error("No hay un dato real del mes anterior disponible en el Excel.")
    st.stop()

pasajeros_previos = int(fila_previa.iloc[0])
st.markdown("### 2. Dato real obtenido automáticamente del Excel")
st.metric(
    f"Pasajeros reales de {mes_anterior} de {anio_anterior}",
    f"{pasajeros_previos:,}".replace(",", " "),
)
st.caption("Fuente: archivo Excel del MTC (hojas 2009–2025).")

st.markdown("### 3. Pronóstico a un mes (t+1)")
if anio_pronostico <= 2024:
    st.warning(
        "Esta es una consulta histórica dentro del periodo de entrenamiento. "
        "No debe presentarse como una evaluación independiente de TEST."
    )
elif anio_pronostico == 2026:
    st.info(
        "Se pronosticará enero de 2026 utilizando diciembre de 2025 real. "
        "No hay datos reales de 2026 en el Excel, por eso no se ofrecen meses posteriores."
    )
else:
    st.info("Pronóstico mensual dentro del periodo TEST de 2025.")

if st.button("Calcular pronóstico", type="primary", use_container_width=True):
    entrada = pd.DataFrame({
        "lag_1": [pasajeros_previos],
        "mes_num": [mes_num_pronostico],
    })
    prediccion = float(modelo_final.predict(entrada)[0])
    if prediccion < 0:
        st.warning(
            "El modelo produjo una cantidad negativa. "
            "No debe interpretarse como un número válido de pasajeros."
        )
    else:
        st.metric(
            f"Pasajeros pronosticados para {mes_pronostico} de {anio_pronostico}",
            f"{round(prediccion):,}".replace(",", " "),
        )
        st.caption(
            "Estimación del modelo F; no corresponde a un valor real observado."
        )

with st.expander("Metodología y limitaciones"):
    st.write(
        "El Pipeline del modelo F fue entrenado con datos hasta diciembre de 2024. "
        "Utiliza 'lag_1' (pasajeros reales del mes anterior) y 'mes_num' "
        "(mes a pronosticar). No se reentrena al consultar la aplicación."
    )
    st.write(
        "TEST 2025: doce pronósticos sucesivos t+1, RMSE ≈ 34 820, "
        "MAE ≈ 27 887 y R² ≈ 0,2376. Estas métricas no corresponden a un "
        "pronóstico recursivo de doce meses realizado desde diciembre de 2024."
    )
    st.write(
        "La aplicación utiliza registros históricos del Excel disponible. "
        "En operación real es necesario confirmar la publicación del dato anterior "
        "antes de pronosticar. Eventos extraordinarios pueden reducir la precisión."
    )

st.caption(
    "Business Predictive Analytics · Grupo 2 · "
    "Apartado 5.2: entorno productivo controlado."
)
