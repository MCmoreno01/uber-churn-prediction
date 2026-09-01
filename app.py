import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================================

st.set_page_config(
    page_title="Predicción de Churn",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# CARGAR MODELO Y OBJETOS
# ==========================================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_churn_final.pkl")
    scaler = joblib.load("scaler_churn.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    return modelo, scaler, columnas


modelo, scaler, columnas = cargar_modelo()


# ==========================================================
# ENCABEZADO
# ==========================================================

st.title("🚗 Predicción de Churn")

st.write(
    """
    Esta aplicación utiliza un modelo de Machine Learning para
    estimar el riesgo de abandono (Churn) de un usuario a partir
    de las características de su viaje.
    """
)

st.info(
    "Ingrese las características del viaje y presione "
    "'Realizar predicción' para obtener el resultado."
)

st.divider()


# ==========================================================
# FORMULARIO
# ==========================================================

st.subheader("📋 Datos del viaje")

col1, col2 = st.columns(2)

with col1:

    vehicle_type = st.number_input(
        "Vehicle Type",
        min_value=0,
        step=1
    )

    pickup_location = st.number_input(
        "Pickup Location",
        min_value=0,
        step=1
    )

    avg_vtat = st.number_input(
        "Avg VTAT",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    booking_value = st.number_input(
        "Booking Value",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    payment_method = st.number_input(
        "Payment Method",
        min_value=0,
        step=1
    )


with col2:

    drop_location = st.number_input(
        "Drop Location",
        min_value=0,
        step=1
    )

    avg_ctat = st.number_input(
        "Avg CTAT",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    ride_distance = st.number_input(
        "Ride Distance",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1,
        step=1
    )


st.divider()


# ==========================================================
# PREDICCIÓN
# ==========================================================

if st.button(
    "🔍 Realizar predicción",
    type="primary",
    use_container_width=True
):

    datos = pd.DataFrame(
        [{
            "Vehicle Type": vehicle_type,
            "Pickup Location": pickup_location,
            "Drop Location": drop_location,
            "Avg VTAT": avg_vtat,
            "Avg CTAT": avg_ctat,
            "Booking Value": booking_value,
            "Ride Distance": ride_distance,
            "Payment Method": payment_method,
            "month": month,
            "day": day
        }]
    )

    try:

        # Mantener exactamente el mismo orden
        # utilizado durante el entrenamiento
        datos = datos[columnas]

        # Aplicar el mismo escalamiento
        datos_scaled = scaler.transform(datos)

        # Predicción
        prediccion = modelo.predict(datos_scaled)[0]

        # Probabilidad
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(datos_scaled)[0]
            prob_churn = probabilidades[1]
        else:
            prob_churn = None


        # ==================================================
        # RESULTADO
        # ==================================================

        st.subheader("📊 Resultado de la predicción")

        if prediccion == 1:

            st.error(
                "⚠️ El cliente presenta riesgo de Churn."
            )

            estado = "RIESGO DE CHURN"

        else:

            st.success(
                "✅ El cliente no presenta riesgo de Churn."
            )

            estado = "SIN RIESGO DE CHURN"


        col_resultado1, col_resultado2 = st.columns(2)

        with col_resultado1:

            st.metric(
                "Clasificación",
                estado
            )

        with col_resultado2:

            if prob_churn is not None:

                st.metric(
                    "Probabilidad de Churn",
                    f"{prob_churn:.2%}"
                )


        # Barra de probabilidad
        if prob_churn is not None:

            st.write("#### Nivel de riesgo")

            st.progress(
                int(prob_churn * 100)
            )

            if prob_churn >= 0.70:

                st.warning(
                    "🔴 Riesgo alto: se recomienda "
                    "priorizar al cliente en una "
                    "estrategia de retención."
                )

            elif prob_churn >= 0.40:

                st.warning(
                    "🟠 Riesgo medio: se recomienda "
                    "realizar seguimiento al cliente."
                )

            else:

                st.success(
                    "🟢 Riesgo bajo de abandono."
                )


        # Mostrar datos ingresados
        with st.expander(
            "Ver datos utilizados para la predicción"
        ):

            st.dataframe(
                datos,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Se presentó un error al realizar la predicción."
        )

        st.exception(e)


# ==========================================================
# INFORMACIÓN DEL MODELO
# ==========================================================

st.divider()

st.subheader("🤖 Información del modelo")

st.write(
    """
    El modelo fue seleccionado después de comparar diferentes
    algoritmos de clasificación y realizar optimización de
    hiperparámetros mediante GridSearchCV.
    """
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "95.57%")
m2.metric("Precision", "100.00%")
m3.metric("Recall", "88.33%")
m4.metric("F1-score", "93.81%")

st.caption(
    "Las métricas corresponden al desempeño obtenido "
    "sobre el conjunto de prueba."
)
