import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import seaborn as sns
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Chargement du modèle et du scaler
model = load_model('models/ANNs_modele_keras.h5')
scaler = joblib.load('models/scaler.pkl')

# Configuration de la page
st.set_page_config(page_title="Prédiction de Risque de Crédit", layout="wide")
st.title("📊 Système d'Évaluation de Crédit")

# Onglets
tab1, tab2 = st.tabs(["🔍 Exploration des Données", "🎯 Prédiction"])

# Onglet 1: Exploration des données
with tab1:
    st.header("Analyse Statistique")
    df = pd.read_csv('data/processed_data.csv')

    # Sélection de visualisation
    option = st.selectbox(
        "Choisissez une visualisation",
        ("Histogramme des Âges", "Distribution des Revenus", "Matrice de Corrélation")
    )

    if option == "Histogramme des Âges":
        fig, ax = plt.subplots()
        df['age'].hist(bins=30, ax=ax)
        st.pyplot(fig)

    elif option == "Distribution des Revenus":
        fig, ax = plt.subplots()
        df['MonthlyIncome'].plot(kind='kde', ax=ax)
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

# Onglet 2: Prédiction
with tab2:
    st.header("Prédiction en Temps Réel")

    # Formulaire de saisie
    with st.form("credit_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Âge", 18, 100, 30)
            revolving_util = st.slider("Taux d'utilisation du crédit (%)", 0, 150, 30)
            debt_ratio = st.slider("Ratio dette/revenu", 0.0, 10.0, 0.5)

        with col2:
            monthly_income = st.number_input("Revenu mensuel ($)", 0, 50000, 3000)
            past_due = st.slider("Nombre de retards (30-59 jours)", 0, 10, 0)
            dependents = st.slider("Personnes à charge", 0, 10, 0)

        submitted = st.form_submit_button("Prédire le risque")

    # Prédiction et explication SHAP
    if submitted:
        # Préparation des données
        input_data = pd.DataFrame([[revolving_util, age, past_due, debt_ratio, monthly_income, dependents]],
                                 columns=['RevolvingUtilizationOfUnsecuredLines', 'age',
                                          'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                                          'MonthlyIncome', 'NumberOfDependents'])

        # Scaling
        scaled_data = scaler.transform(input_data)

        # Prédiction
        proba = model.predict(scaled_data)[0][0]
        risk = "🔴 Risque Élevé" if proba > 0.5 else "🟢 Risque Faible"

        # Affichage des résultats
        st.metric("Probabilité de défaut", f"{proba:.0%}", delta_color="inverse")
        st.subheader(f"Résultat: {risk}")

        # Explication SHAP
        st.subheader("📌 Facteurs Clés Influençant la Décision")
        explainer = shap.Explainer(model)
        shap_values = explainer(scaled_data)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=3, show=False)
        st.pyplot(fig)

        # Conseils selon le risque
        if proba > 0.5:
            st.warning("""
            **Recommandations:**
            - Exiger des garanties supplémentaires
            - Limiter le montant du crédit
            - Surveillance renforcée des paiements
            """)
