import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import altair as alt

# Load models and scaler
reg_model = load('reg_model.joblib')
clf_model = load('clf_model.joblib')
scaler = load('scaler.joblib')

# Load data
df = pd.read_csv('Cleaned_MarathonData.csv')

# Sidebar navigation
page = st.sidebar.selectbox("Kies een pagina:", ["Overzicht", "Regressie", "Classificatie"])

# -----------------------------------------
if page == "Overzicht":
    st.title("Marathon Dashboard Prototype")
    st.markdown("""
    Deze applicatie voorspelt:
    - De marathontijd op basis van trainingsdata
    - In welke categorie een loper valt

    **Dataset:** Cleaned_MarathonData.csv  
    Aantal renners: **{}**

    **Features gebruikt:**
    - km4week
    - sp4week (snelheid)
    - CrossTraining
    - Wall21 (belasting bij 21k)

    Gemaakt met: `scikit-learn`, `joblib`, `Streamlit`, `Altair`
    """.format(len(df)))

    # Toon samenvatting
    st.subheader("Distributie van marathontijden")
    chart = alt.Chart(df).mark_bar().encode(
        alt.X("MarathonTime", bin=alt.Bin(maxbins=20)),
        y='count()'
    ).properties(width=600)
    st.altair_chart(chart)

# -----------------------------------------
elif page == "Regressie":
    st.title("Voorspel marathontijd")

    km4week = st.slider("Afstand per week (km)", 0.0, 200.0, 100.0)
    sp4week = st.slider("Gemiddelde snelheid (km/u)", 5.0, 20.0, 12.0)
    cross = st.selectbox("Cross-training?", ["Nee", "Ja"])
    wall21 = st.slider("Wall21 score", 0.5, 2.0, 1.0)

    cross = 1 if cross == "Ja" else 0

    input_data = np.array([[km4week, sp4week, cross, wall21]])
    predicted_time = reg_model.predict(input_data)[0]

    st.success(f"Geschatte marathontijd: **{round(predicted_time, 2)} uur**")

# -----------------------------------------
elif page == "Classificatie":
    st.title("Voorspel categorie van de loper")

    km4week = st.slider("Afstand per week (km)", 0.0, 200.0, 100.0)
    sp4week = st.slider("Gemiddelde snelheid (km/u)", 5.0, 20.0, 12.0)
    cross = st.selectbox("Cross-training?", ["Nee", "Ja"])
    wall21 = st.slider("Wall21 score", 0.5, 2.0, 1.0)

    cross = 1 if cross == "Ja" else 0

    input_data = np.array([[km4week, sp4week, cross, wall21]])
    input_scaled = scaler.transform(input_data)

    predicted_class = clf_model.predict(input_scaled)[0]
    st.info(f"Voorspelde categorie: **{predicted_class}**")