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

# Verwijder ongeldige categorieën (alleen 0 t/m 3)
df = df[df['CATEGORY'].isin([0, 1, 2, 3])]

# Sidebar navigation
page = st.sidebar.selectbox("Kies een pagina:", ["Overzicht", "Regressie", "Classificatie"])

# -----------------------------------------
if page == "Overzicht":
    st.title("Marathon Dashboard Prototype")
    st.markdown(f"""
    Deze applicatie voorspelt:
    - De marathontijd op basis van trainingsdata
    - De categorie waar een loper in valt

    **Dataset:** Cleaned_MarathonData.csv  
    Aantal renners: **{len(df)}**

    **Gebruikte variabelen:**
    - `km4week`: kilometers gelopen 4 weken voor de marathon
    - `Wall21`: vermoeidheidsscore rond 21 km

    Gemaakt door: Thomas Blom
    """)

    st.subheader("Distributie van marathontijden")
    chart = alt.Chart(df).mark_bar().encode(
        alt.X("MarathonTime", bin=alt.Bin(maxbins=20)),
        y='count()'
    ).properties(width=600)
    st.altair_chart(chart)

# -----------------------------------------
elif page == "Regressie":
    st.title("Voorspel marathontijd")

    st.markdown(f"""
    Mean squared error: 0.1066
    Het model voorspelt marathontijden met een gemiddelde afwijking van ongeveer 19.6 minuten ten opzichte van de werkelijke tijd.
    """)

    km4week = st.slider("Afstand 4 weken voor de marathon (km)", 0.0, 200.0, 100.0)
    wall21 = st.slider("Wall21-score", 1.0, 2.0, 1.0)

    input_data = np.array([[km4week, wall21]])
    predicted_time = reg_model.predict(input_data)[0]

    st.success(f"Geschatte marathontijd: **{round(predicted_time, 2)} uur**")

# -----------------------------------------
elif page == "Classificatie":
    st.title("Voorspel categorie van de loper")

    st.markdown(f"""
    Accuracy score: 77%
    Mean Absolute error: 0.23
    Het model voorspelt de marathoncategorie met een gemiddelde afwijking van 0.23 categorieën. 
    Dit betekent dat het model in de meeste gevallen de juiste categorie voorspelt, of er slechts één categorie naast zit.
    """)

    km4week = st.slider("Afstand 4 weken voor de marathon (km)", 0.0, 200.0, 100.0)
    wall21 = st.slider("Wall21-score", 1.0, 2.0, 1.0)

    input_data = np.array([[km4week, wall21]])
    input_scaled = scaler.transform(input_data)

    predicted_class = clf_model.predict(input_scaled)[0]
    st.info(f"Voorspelde categorie: **{predicted_class}**")
    categories = [c for c in clf_model.classes_ if c in [0, 1, 2, 3]]

    # Kansverdeling per categorie tonen
    probs = clf_model.predict_proba(input_scaled)[0]
    categories = clf_model.classes_
    probs_df = pd.DataFrame({
        'Categorie': categories,
        'Kans (%)': np.round(probs * 100, 2)
    })

    'Categorie 0: Onder de 3 uur'
    'Categorie 1: Tussen 3 uur en 3 uur en 20 minuten'
    'Categorie 2: Tussen 3 uur en 20 minuten en 3 uur en 40 minuten'
    'Categorie 3: Tussen 3 uur en 40 minuten en 4 uur'

    st.subheader("Kansverdeling per categorie:")
    st.dataframe(probs_df)
