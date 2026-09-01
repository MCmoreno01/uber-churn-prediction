import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Predicción de Churn",
    page_icon="🚗",
    layout="wide"
)


# ==========================================================
# CARGA DE ARCHIVOS
# ==========================================================

@st.cache_resource
def cargar_archivos():

    modelo = joblib.load(
        "modelo_churn_final.pkl"
    )

    scaler = joblib.load(
        "scaler_churn.pkl"
    )

    columnas = joblib.load(
        "columnas_modelo.pkl"
    )

    mapeos = joblib.load(
        "mapeos_categorias.pkl"
    )

    return modelo, scaler, columnas, mapeos


modelo, scaler, columnas, mapeos = cargar_archivos()


# ==========================================================
# ENCABEZADO
# ==========================================================

st.title("🚗 Predicción de Abandono de usuarios")

st.markdown(
    """
    Esta herramienta utiliza un modelo de Machine Learning para
    estimar el riesgo de abandono de un usuario a partir de las
    características asociadas a su viaje.

    Complete la información y seleccione **Analizar riesgo**.
    """
)

st.divider()


# ==========================================================
# DATOS DEL VIAJE
# ==========================================================

st.subheader("📋 Información del viaje")

st.caption(
    "Seleccione las características correspondientes al usuario "
    "que desea analizar."
)


with st.form("formulario_churn"):

    col1, col2 = st.columns(2)


    # ======================================================
    # COLUMNA IZQUIERDA
    # ======================================================

    with col1:

        vehicle_type = st.selectbox(
            "🚘 Tipo de vehículo",
            options=list(
                mapeos["Vehicle Type"].keys()
            ),
            help=(
                "Tipo de vehículo utilizado "
                "en la reserva."
            )
        )

        pickup_location = st.selectbox(
            "📍 Lugar de recogida",
            options=list(
                mapeos["Pickup Location"].keys()
            ),
            help=(
                "Zona o ubicación donde comenzó "
                "el viaje."
            )
        )

        avg_vtat = st.number_input(
            "⏱️ Avg VTAT",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=(
                "Tiempo promedio que tarda el conductor "
                "en llegar al lugar de recogida (en minutos)."
            )
        )

        booking_value = st.number_input(
            "💰 Valor de la reserva",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help=(
                "Valor monetario asociado "
                "a la reserva."
            )
        )

        payment_method = st.selectbox(
            "💳 Método de pago",
            options=list(
                mapeos["Payment Method"].keys()
            ),
            help=(
                "Método de pago utilizado "
                "por el usuario."
            )
        )


    # ======================================================
    # COLUMNA DERECHA
    # ======================================================

    with col2:

        drop_location = st.selectbox(
            "🏁 Lugar de destino",
            options=list(
                mapeos["Drop Location"].keys()
            ),
            help=(
                "Zona o ubicación de destino "
                "del viaje."
            )
        )

        avg_ctat = st.number_input(
            "⏱️ Avg CTAT",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=(
                "Tiempo promedio del trayecto desde el"
                "punto de recogida hasta el destino (en minutos)."
            )
        )

        ride_distance = st.number_input(
            "🛣️ Distancia del viaje",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=(
                "Distancia recorrida durante "
                "el viaje en Km."
            )
        )

        month = st.selectbox(
            "📅 Mes",
            options=list(range(1, 13)),
            format_func=lambda x: {
                1: "Enero",
                2: "Febrero",
                3: "Marzo",
                4: "Abril",
                5: "Mayo",
                6: "Junio",
                7: "Julio",
                8: "Agosto",
                9: "Septiembre",
                10: "Octubre",
                11: "Noviembre",
                12: "Diciembre"
            }[x]
        )

        day = st.number_input(
            "📆 Día del mes",
            min_value=1,
            max_value=31,
            value=1,
            step=1
        )


    st.write("")

    boton = st.form_submit_button(
        "🔍 Analizar riesgo de Churn",
        type="primary",
        use_container_width=True
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

if boton:

    # Convertir categorías visibles
    # a los códigos utilizados por el modelo

    vehicle_code = mapeos[
        "Vehicle Type"
    ][vehicle_type]

    pickup_code = mapeos[
        "Pickup Location"
    ][pickup_location]

    drop_code = mapeos[
        "Drop Location"
    ][drop_location]

    payment_code = mapeos[
        "Payment Method"
    ][payment_method]


    datos = pd.DataFrame(
        [{
            "Vehicle Type": vehicle_code,
            "Pickup Location": pickup_code,
            "Drop Location": drop_code,
            "Avg VTAT": avg_vtat,
            "Avg CTAT": avg_ctat,
            "Booking Value": booking_value,
            "Ride Distance": ride_distance,
            "Payment Method": payment_code,
            "month": month,
            "day": day
        }]
    )


    try:

        # Orden exacto utilizado en entrenamiento
        datos = datos[columnas]

        # Aplicar scaler
        datos_scaled = scaler.transform(datos)

        # Predicción
        prediccion = modelo.predict(
            datos_scaled
        )[0]


        # Probabilidad
        if hasattr(
            modelo,
            "predict_proba"
        ):

            probabilidad = modelo.predict_proba(
                datos_scaled
            )[0][1]

        else:

            probabilidad = None


        # ==================================================
        # RESULTADOS
        # ==================================================

        st.divider()

        st.subheader("📊 Resultado del análisis")


        if probabilidad is not None:

            porcentaje = probabilidad * 100

        else:

            porcentaje = None


        # ----------------------------------------------
        # CLASIFICACIÓN
        # ----------------------------------------------

        if prediccion == 1:

            st.error(
                "⚠️ Usuario con riesgo de Churn"
            )

        else:

            st.success(
                "✅ Usuario sin riesgo de Churn"
            )


        # ----------------------------------------------
        # MÉTRICAS DE RESULTADO
        # ----------------------------------------------

        r1, r2 = st.columns(2)


        with r1:

            if prediccion == 1:

                st.metric(
                    "Clasificación",
                    "CHURN"
                )

            else:

                st.metric(
                    "Clasificación",
                    "NO CHURN"
                )


        with r2:

            if porcentaje is not None:

                st.metric(
                    "Probabilidad estimada de Churn",
                    f"{porcentaje:.1f}%"
                )


        # ----------------------------------------------
        # NIVEL DE RIESGO
        # ----------------------------------------------

        if probabilidad is not None:

            st.write("### Nivel de riesgo")

            st.progress(
                int(probabilidad * 100)
            )


            if probabilidad >= 0.70:

                nivel = "Alto"

                st.error(
                    """
                    🔴 **Riesgo alto**

                    El usuario presenta una alta probabilidad
                    estimada de abandono.
                    """
                )

                st.markdown(
                    """
                    **Recomendación de negocio:** priorizar al
                    usuario dentro de una estrategia de retención,
                    realizando seguimiento y evaluando acciones
                    comerciales o de fidelización.
                    """
                )


            elif probabilidad >= 0.40:

                nivel = "Medio"

                st.warning(
                    """
                    🟠 **Riesgo medio**

                    El usuario presenta señales que podrían estar
                    asociadas con abandono.
                    """
                )

                st.markdown(
                    """
                    **Recomendación de negocio:** realizar
                    seguimiento al comportamiento del usuario y
                    considerar acciones preventivas de retención.
                    """
                )


            else:

                nivel = "Bajo"

                st.success(
                    """
                    🟢 **Riesgo bajo**

                    Actualmente el usuario presenta una baja
                    probabilidad estimada de abandono.
                    """
                )

                st.markdown(
                    """
                    **Recomendación de negocio:** mantener las
                    estrategias actuales de experiencia y
                    fidelización.
                    """
                )


        # ==================================================
        # RESUMEN DEL CASO
        # ==================================================

        st.write("### 📝 Resumen del usuario")

        resumen = pd.DataFrame({
            "Variable": [
                "Tipo de vehículo",
                "Lugar de recogida",
                "Lugar de destino",
                "Avg VTAT",
                "Avg CTAT",
                "Valor reserva",
                "Distancia",
                "Método de pago",
                "Mes",
                "Día"
            ],

            "Valor": [
                vehicle_type,
                pickup_location,
                drop_location,
                avg_vtat,
                avg_ctat,
                booking_value,
                ride_distance,
                payment_method,
                month,
                day
            ]
        })

        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "No fue posible realizar la predicción."
        )

        st.exception(e)


# ==========================================================
# INFORMACIÓN DEL MODELO
# ==========================================================

st.divider()

st.subheader("🤖 Desempeño del modelo")

st.write(
    """
    El modelo final fue seleccionado después de comparar
    diferentes algoritmos de clasificación y realizar
    optimización de hiperparámetros mediante GridSearchCV.
    """
)


m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Accuracy",
    "95.57%"
)

m2.metric(
    "Precision",
    "100.00%"
)

m3.metric(
    "Recall",
    "88.33%"
)

m4.metric(
    "F1-score",
    "93.81%"
)

m5.metric(
    "ROC-AUC",
    "99.32%"
)


st.caption(
    """
    Métricas obtenidas sobre el conjunto de prueba reservado
    para la evaluación final del modelo.
    """
)


# ==========================================================
# INTERPRETACIÓN
# ==========================================================

with st.expander(
    "ℹ️ ¿Cómo interpretar la predicción?"
):

    st.markdown(
        """
        **Churn:** el modelo identifica señales asociadas con
        una posible salida o abandono del usuario.

        **No Churn:** el modelo no identifica actualmente
        suficientes señales para clasificar al usuario como
        potencial abandono.

        La predicción debe utilizarse como una herramienta de
        apoyo para priorizar acciones y no como una decisión
        automática sobre el usuario.
        """
    )
