
import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Clasificación de Reservas - Uber",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# CARGA DE ARCHIVOS
# ==========================================================

@st.cache_resource
def cargar_archivos():
    modelo = joblib.load("modelo_churn_final.pkl")
    scaler = joblib.load("scaler_churn.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    mapeos = joblib.load("mapeos_categorias.pkl")

    return modelo, scaler, columnas, mapeos


try:
    modelo, scaler, columnas, mapeos = cargar_archivos()
except Exception as e:
    st.error("No fue posible cargar los archivos del modelo.")
    st.exception(e)
    st.stop()


# ==========================================================
# VALIDACIÓN DE VARIABLES DEL ESCENARIO A
# ==========================================================

VARIABLES_ESPERADAS = [
    "Vehicle Type",
    "Pickup Location",
    "Drop Location",
    "Avg VTAT",
    "Booking Value",
    "Ride Distance",
    "Payment Method",
    "month",
    "day"
]

columnas = list(columnas)

if (
    len(columnas) != 9
    or set(columnas) != set(VARIABLES_ESPERADAS)
):
    st.error(
        "Los archivos del modelo no corresponden al "
        "escenario A actualizado."
    )
    st.warning(
        "Debes reemplazar los archivos .pkl por los "
        "generados después de excluir Avg CTAT."
    )
    st.write("Columnas encontradas:", columnas)
    st.stop()

if hasattr(modelo, "n_features_in_"):
    if modelo.n_features_in_ != len(columnas):
        st.error(
            "El modelo cargado no coincide con las "
            "nueve variables predictoras."
        )
        st.stop()

if hasattr(scaler, "n_features_in_"):
    if scaler.n_features_in_ != len(columnas):
        st.error(
            "El escalador cargado no coincide con "
            "las nueve variables predictoras."
        )
        st.stop()


# ==========================================================
# ENCABEZADO
# ==========================================================

st.title("🚗 Clasificación de Reservas de Uber")

st.markdown(
    """
    Esta aplicación utiliza un modelo **Random Forest
    optimizado** para clasificar reservas según los patrones
    identificados en los datos históricos.

    **No Churn:** reserva completada.

    **Churn:** reserva no completada.

    Complete las características de la reserva y seleccione
    **Analizar reserva**.
    """
)

st.info(
    "Este prototipo es una demostración académica. "
    "No está validado para anticipar cancelaciones "
    "desde el momento de creación de una reserva."
)

st.divider()

# ==========================================================
# FORMULARIO
# ==========================================================

st.subheader("📋 Información de la reserva")

with st.form("formulario_reserva"):

    col1, col2 = st.columns(2)

    # ------------------------------------------------------
    # COLUMNA IZQUIERDA
    # ------------------------------------------------------

    with col1:

        vehicle_type = st.selectbox(
            "🚘 Tipo de vehículo",
            options=list(
                mapeos["Vehicle Type"].keys()
            )
        )

        pickup_location = st.selectbox(
            "📍 Lugar de recogida",
            options=list(
                mapeos["Pickup Location"].keys()
            )
        )

        avg_vtat = st.number_input(
            "⏱️ Tiempo promedio de llegada del conductor (Avg VTAT)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Tiempo expresado en minutos."
        )

        booking_value = st.number_input(
            "💰 Valor de la reserva",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        payment_method = st.selectbox(
            "💳 Método de pago",
            options=list(
                mapeos["Payment Method"].keys()
            )
        )

    # ------------------------------------------------------
    # COLUMNA DERECHA
    # ------------------------------------------------------

    with col2:

        drop_location = st.selectbox(
            "🏁 Lugar de destino",
            options=list(
                mapeos["Drop Location"].keys()
            )
        )

        ride_distance = st.number_input(
            "🛣️ Distancia del viaje (km)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        nombres_meses = {
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
        }

        month = st.selectbox(
            "📅 Mes",
            options=list(range(1, 13)),
            format_func=lambda x: nombres_meses[x]
        )

        day = st.number_input(
            "📆 Día del mes",
            min_value=1,
            max_value=31,
            value=1,
            step=1
        )

    boton = st.form_submit_button(
        "🔍 Analizar reserva",
        type="primary",
        use_container_width=True
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

if boton:

    try:

        # Transformar categorías usando los mapeos
        # originales del entrenamiento.

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

        # Construir registro con nueve variables.

        datos = pd.DataFrame([{
            "Vehicle Type": vehicle_code,
            "Pickup Location": pickup_code,
            "Drop Location": drop_code,
            "Avg VTAT": avg_vtat,
            "Booking Value": booking_value,
            "Ride Distance": ride_distance,
            "Payment Method": payment_code,
            "month": month,
            "day": day
        }])

        # Respetar el orden original del entrenamiento.
        datos = datos[columnas]

        # Aplicar el mismo escalador.
        datos_scaled = scaler.transform(datos)

        # Clasificación.
        prediccion = int(
            modelo.predict(datos_scaled)[0]
        )

        # Probabilidad estimada de la clase Churn.
        probabilidad = None

        if hasattr(modelo, "predict_proba"):
            clases = list(modelo.classes_)

            if 1 in clases:
                indice_churn = clases.index(1)

                probabilidad = float(
                    modelo.predict_proba(datos_scaled)[
                        0, indice_churn
                    ]
                )

        # ==================================================
        # RESULTADOS
        # ==================================================

        st.divider()
        st.subheader("📊 Resultado del análisis")

        if prediccion == 1:
            st.error(
                "⚠️ Clasificación: RESERVA NO COMPLETADA"
            )
        else:
            st.success(
                "✅ Clasificación: RESERVA COMPLETADA"
            )

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Clasificación",
                "CHURN" if prediccion == 1 else "NO CHURN"
            )

        with r2:
            if probabilidad is not None:
                st.metric(
                    "Estimación del modelo para Churn",
                    f"{probabilidad * 100:.1f}%"
                )
            else:
                st.metric(
                    "Estimación del modelo",
                    "No disponible"
                )

        if probabilidad is not None:

            st.write("### Estimación de la clase Churn")

            st.progress(
                max(
                    0,
                    min(100, int(round(probabilidad * 100)))
                )
            )

            st.caption(
                "El porcentaje corresponde a la salida "
                "predict_proba del clasificador. "
                "No representa una probabilidad calibrada "
                "ni una estimación validada de abandono futuro."
            )

        # ==================================================
        # RESUMEN DE LA RESERVA
        # ==================================================

        st.write("### 📝 Resumen de la reserva")

        resumen = pd.DataFrame({
            "Variable": [
                "Tipo de vehículo",
                "Lugar de recogida",
                "Lugar de destino",
                "Avg VTAT",
                "Valor de la reserva",
                "Distancia del viaje",
                "Método de pago",
                "Mes",
                "Día"
            ],
            "Valor": [
                vehicle_type,
                pickup_location,
                drop_location,
                avg_vtat,
                booking_value,
                ride_distance,
                payment_method,
                nombres_meses[month],
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
            "No fue posible realizar la clasificación."
        )
        st.exception(e)


# ==========================================================
# INFORMACIÓN DEL MODELO
# ==========================================================

st.divider()

st.subheader("🤖 Desempeño del modelo")

st.markdown(
    """
    El modelo final corresponde a **Random Forest optimizado**,
    seleccionado después de comparar cuatro algoritmos
    de clasificación y evaluar sus resultados sobre el
    conjunto de prueba del escenario A.
    """
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", "94.18%")
m2.metric("Precision", "100.00%")
m3.metric("Recall", "84.68%")
m4.metric("F1-score", "91.71%")
m5.metric("ROC-AUC", "98.21%")

st.caption(
    "Resultados del modelo Random Forest optimizado "
    "sobre 30.000 reservas del conjunto de prueba."
)

# ==========================================================
# INTERPRETACIÓN
# ==========================================================

with st.expander(
    "ℹ️ ¿Cómo interpretar los resultados?"
):

    st.markdown(
        """
        **Churn (1):** reserva que no se completó.

        **No Churn (0):** reserva completada.

        **Precision (100 %):** en el conjunto de prueba,
        las reservas clasificadas como Churn
        correspondieron efectivamente a esa categoría.

        **Recall (84,68 %):** el modelo identificó
        aproximadamente 85 de cada 100 reservas
        realmente no completadas.

        **F1-score (91,71 %):** resume el equilibrio
        entre precisión y sensibilidad para la clase Churn.

        **Alcance:** esta aplicación permite explorar
        la clasificación de reservas a partir de
        información histórica. No demuestra que sea
        posible anticipar la no finalización desde
        el momento de la reserva.
        """
    )
