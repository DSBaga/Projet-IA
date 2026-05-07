import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.express as px
import cvxopt as opt
from cvxopt import solvers
import os

st.set_page_config(page_title="Robot_advisor", page_icon="🤖", layout="centered", initial_sidebar_state="auto")

# ✅ Chemins relatifs par rapport à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

investors = pd.read_csv(os.path.join(BASE_DIR, "Data", "Robot_advisor", "InputData.csv"), index_col=0)

assets = pd.read_csv(os.path.join(BASE_DIR, "Data", "Robot_advisor", "CAC40Data.csv"), index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
assets = assets.ffill()  # ✅ Remplace fillna(method='ffill') déprécié en pandas 2.x


def predict_riskTolerance(X_input):
    filename = os.path.join(BASE_DIR, "Notebooks", "Robot", "finalized_model.sav")  # ✅ Chemin relatif
    loaded_model = load(open(filename, 'rb'))
    predictions = loaded_model.predict(X_input)
    return predictions


def get_asset_allocation(riskTolerance, stock_ticker):
    assets_selected = assets.loc[:, stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1 - riskTolerance

    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    Alloc = pd.DataFrame(data=np.array(portfolios['x']), index=assets_selected.columns)

    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100
    return Alloc, returns_sum_pd


# UI
st.title("Bienvenu sur l'application de 'Robot Advisor' en investissement sur le marché boursier CAC40")

st.image("https://findependent.ch/wp-content/uploads/2023/09/findependent_robo_advisor_schweiz_blog_banner.png",
         caption="Robot Advisor", use_column_width=True)

st.subheader("Etapes 2 : Repartition d'actifs et la performance du portefeuille")
st.sidebar.title("Etape 1 : saisissez les caractéristiques de l'investisseur")

with st.sidebar:
    age = st.slider('Age:', min_value=investors['AGE07'].min(), max_value=70, value=25)
    net_worth = st.slider('NetWorth:', min_value=-1000000, max_value=3000000, value=10000)
    income = st.slider('Income:', min_value=-1000000, max_value=3000000, value=100000)
    education = st.slider('Education Level (scale of 4):', min_value=1, max_value=4, value=2)
    married = st.slider('Married:', min_value=1, max_value=2, value=1)
    kids = st.slider('Kids:', min_value=investors['KIDS07'].min(), max_value=investors['KIDS07'].max(), value=3)
    occupation = st.slider('Occupation:', min_value=1, max_value=4, value=3)
    willingness = st.slider('Willingness to take Risk:', min_value=1, max_value=4, value=3)

    if st.sidebar.button('Calcul de la tolérance au risque (Risk Tolerance)'):
        X_input = [[age, education, married, kids, occupation, income, willingness, net_worth]]
        risk_tolerance_prediction = predict_riskTolerance(X_input)
        st.sidebar.write(f'Tolérance au risque prédite: {round(float(risk_tolerance_prediction[0] * 100), 2)}')

risk_tolerance_text = st.text_input('Tolérance au risque (échelle de 100) :')
selected_assets = st.multiselect('Actifs à inclure dans le portefeuille:',
                                 options=list(assets.columns),
                                 default=['Air Liquide', 'Airbus', 'Alstom', 'AXA', 'BNP Paribas'])

if st.button('Soumettre'):
    Alloc, returns_sum_pd = get_asset_allocation(float(risk_tolerance_text), selected_assets)

    st.subheader('Répartition des actifs : Répartition en fonction de la moyenne et de la variance')
    fig_alloc = px.bar(Alloc, x=Alloc.index, y=Alloc.iloc[:, 0],
                       labels={'index': 'Assets', '0': 'Allocation'})
    st.plotly_chart(fig_alloc)

    st.subheader("Valeur du portefeuille pour un investissement de 100 euros")
    fig_performance = px.line(returns_sum_pd, x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0],
                              labels={'index': 'Date', '0': 'Portfolio Value'})
    st.plotly_chart(fig_performance)
