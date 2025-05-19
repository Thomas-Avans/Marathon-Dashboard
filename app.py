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
    - km4week (Kilometers gelopen 4 weken voor de marathon)
    - Wall21 (belasting bij helft van de marathon)

    Gemaakt door: Thomas Blom
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

    km4week = st.slider("Afstand 4 weken voor de Marathon (km)", 0.0, 200.0, 100.0)
    wall21 = st.slider("Wall21 score", 1.0, 2.0, 1.0)


    input_data = np.array([[km4week, wall21]])
    predicted_time = reg_model.predict(input_data)[0]

    st.success(f"Geschatte marathontijd: **{round(predicted_time, 2)} uur**")

# -----------------------------------------
elif page == "Classificatie":
    st.title("Voorspel categorie van de loper")

    km4week = st.slider("Afstand 4 weken voor de Marathon (km)", 0.0, 200.0, 100.0)
    wall21 = st.slider("Wall21 score", 1.0, 2.0, 1.0)

    input_data = np.array([[km4week, wall21]])
    input_scaled = scaler.transform(input_data)

    predicted_class = clf_model.predict(input_scaled)[0]
    st.info(f"Voorspelde categorie: **{predicted_class}**")
    probs = clf_model.predict_proba(input_scaled)[0]
categories = clf_model.classes_
probs_df = pd.DataFrame({
    'Categorie': categories,
    'Kans (%)': np.round(probs * 100, 2)
})

st.subheader("Kansverdeling per categorie:")
st.dataframe(probs_df)