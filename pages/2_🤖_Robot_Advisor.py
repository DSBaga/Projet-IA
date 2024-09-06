import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.express as px
import cvxopt as opt # importer cvxopt pour la résolution du problème d'optimisation
from cvxopt import solvers 

st.set_page_config(page_title="Robot_advisor", page_icon="🤖", layout="centered", initial_sidebar_state="auto")


investors = pd.read_csv("C:/Users/bagay/Desktop/PorfolioGitCopie/Data/Robot_advisor/InputData.csv", index_col = 0)

assets = pd.read_csv("C:/Users/bagay/Desktop/PorfolioGitCopie/Data/Robot_advisor/CAC40Data.csv",index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False) # Valeurs manquantes dans les données
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index)) # Supprimer les colonnes avec plus de 30% de valeurs manquantes
assets.drop(labels=drop_list, axis=1, inplace=True) 
assets=assets.fillna(method='ffill') # Remplacer les valeurs manquantes par la valeur précédente

def predict_riskTolerance(X_input): # Predire la tolérance au risque

    filename = 'Notebooks/Robot/finalized_model.sav' # Charger le modèle
    loaded_model = load(open(filename, 'rb')) 
    # estimate accuracy on validation set
    predictions = loaded_model.predict(X_input) # Prédire la tolérance au risque
    return predictions # Retourner la prédiction

#Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance,stock_ticker): 
    #ipdb.set_trace()
    assets_selected = assets.loc[:,stock_ticker] # Sélectionner les actifs choisis
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T # Calculer les rendements
    n = len(return_vec) # Nombre d'actifs
    returns = np.asmatrix(return_vec) # Convertir les rendements en matrice
    mus = 1-riskTolerance # Calculer le rendement attendu

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec)) # Covariance matrix
    pbar = opt.matrix(np.mean(return_vec, axis=1)) 
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1)) 
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b) # Résoudre le problème d'optimisation
    w=portfolios['x'].T # Poids des actifs dans le portefeuille optimal 
    print (w) # Afficher les poids des actifs dans le portefeuille optimal
    Alloc =  pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns) # Créer un dataframe des poids des actifs

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b) # Résoudre le problème d'optimisation
    returns_final=(np.array(assets_selected) * np.array(w)) # Calculer les rendements du portefeuille
    returns_sum = np.sum(returns_final,axis =1) # Calculer la somme des rendements
    returns_sum_pd = pd.DataFrame(returns_sum, index = assets.index ) # Créer un dataframe des rendements du portefeuille
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100 # Normaliser les rendements du portefeuille
    return Alloc,returns_sum_pd # Retourner les poids des actifs et les rendements du portefeuille

# Define the Streamlit app
st.title("Bienvenu sur l'application de 'Robot Advisor' en investissement sur le marché boursier CAC40") # Titre de l'application

# Ajouter une image
st.image("https://findependent.ch/wp-content/uploads/2023/09/findependent_robo_advisor_schweiz_blog_banner.png", caption="Robot Advisor", use_column_width=True)

st.subheader(" Etapes 2 : Repartition d'actifs et la performance du portefeuille") # Sous-titre
st.sidebar.title("Etape 1 : saisissez les caractéristiques de l'investisseur") # Titre de la barre latérale
# Investor Characteristics
# Les caractéristiques de l'investisseur
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
        st.sidebar.write(f'Tolérance au risque prédite: {round(float(risk_tolerance_prediction[0]*100), 2)}')

# Risk Tolerance Charts

risk_tolerance_text = st.text_input('Tolérance au risque (échelle de 100) :')
selected_assets = st.multiselect('Actifs à inclure dans le portefeuille:', 
                                 options=list(assets.columns), 
                                 default=['Air Liquide', 'Airbus', 'Alstom', 'AXA', 'BNP Paribas'])

# Asset Allocation and Portfolio Performance

if st.button('Soumettre'): # 
    Alloc, returns_sum_pd = get_asset_allocation(float(risk_tolerance_text), selected_assets)

    # Display Asset Allocation chart
    st.subheader('Répartition des actifs : Répartition en fonction de la moyenne et de la variance')
    fig_alloc = px.bar(Alloc, x=Alloc.index, y=Alloc.iloc[:, 0], 
                       labels={'index': 'Assets', '0': 'Allocation'})
    st.plotly_chart(fig_alloc)

    # Display Portfolio Performance chart
    st.subheader("Valeur du portefeuille pour un investissement de 100 euros")
    fig_performance = px.line(returns_sum_pd, x=returns_sum_pd.index, y=returns_sum_pd.iloc[:, 0], labels={'index': 'Date', '0': 'Portfolio Value'})
    st.plotly_chart(fig_performance)