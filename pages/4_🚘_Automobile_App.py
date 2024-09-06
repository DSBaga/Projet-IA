import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from streamlit_option_menu import option_menu

# Définir les configurations de la page
st.set_page_config(page_title="Automobile_App", page_icon="🚘", layout="centered", initial_sidebar_state="auto")

# Définir les pages disponibles dans le menu
#with st.sidebar:
    #selected = option_menu(
        #menu_title="Menu",
        #options=["Home","Data", "Visualisations", "Modèle de Prédiction"],
        #icons=["house","database", "bar-chart-line", "robot"],
        #menu_icon="cast",
        #default_index=0,
    #)
# Charger les données
@st.cache_data
def load_data(): # Fonction pour charger les données
    cars = pd.read_csv('Data/Automobile_data.csv') # Charger les données
    # Changer les ? en NaN et supprimer les lignes avec des valeurs manquantes
    cars = cars.replace('?', np.nan)
    cars = cars.dropna()
    cars['price'] = pd.to_numeric(cars['price'], errors='coerce') # Convertir la colonne 'price' en numérique
    return cars # Retourner les données

# Définir la fonction d'inférence pour le modèle de prédiction
def inference(symboling, wheel_base, length, width, height, curb_weight,
            engine_size, compression_ratio, city_mpg, highway_mpg):# Fonction pour l'inférence
    new_data = np.array([symboling, wheel_base, length, width, height, curb_weight,
                        engine_size, compression_ratio, city_mpg, highway_mpg]) # Créer un tableau numpy
    pred = model.predict(new_data.reshape(1, -1)) # Faire une prédiction
    return pred # Retourner la prédiction

# Charger les données et le modèle
cars = load_data() # Charger les données
model = joblib.load(filename='Models/Automibile_final_model.joblib') # Charger le modèle

# Page 1 : Home
#if selected == "Home":
st.title("Bienvenue sur l'Application de Prédiction de Prix de Voitures")
#st.subheader('Cette application est réalisée par Ben H')
#st.markdown('Cette application est une démonstration de Streamlit, une application web pour le déploiement de modèles de Machine Learning')
#st.image("Automobile_image.jpeg", caption="Image de voitures")


    # CSS pour centrer et agrandir l'image
st.markdown(
    """
    <style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%; /* Ajustez cette valeur pour changer la taille de l'image */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<img src="https://deepvisualmarketing.github.io/Project%201_files/car_attr.png">', unsafe_allow_html=True)

# Page 2 : Data
#if selected == "Data":
st.title('Données des Voitures')
st.dataframe(cars)

# Page 3 : Visualisations
#elif selected == "Visualisations":
st.title("Visualisations Interactives")

st.subheader("Diagramme à Barres Interactif")
categorical_variable = st.selectbox("Choisir la variable catégorielle", cars.select_dtypes(include=['object']).columns)
numerical_variable = st.selectbox("Choisir la variable numérique", cars.select_dtypes(include=['number']).columns)
fig = px.bar(cars, x=categorical_variable, y=numerical_variable, title=f"{numerical_variable} vs {categorical_variable}")
st.plotly_chart(fig)

st.subheader("Nuage de Points Interactif")
numeric_cols = cars.select_dtypes(exclude='object').columns.to_list()
var_x = st.selectbox('Choisissez la variable x', numeric_cols)
var_y = st.selectbox('Choisissez la variable y', numeric_cols)
categorical_cols = cars.select_dtypes(include='object').columns.to_list()
var_col = st.selectbox('Choisissez la variable de couleur', categorical_cols)
fig2 = px.scatter(cars, x=var_x, y=var_y, color=var_col, title=f'{var_x} vs {var_y}')
st.plotly_chart(fig2)

# Page 4 : Modèle de Prédiction
#elif selected == "Modèle de Prédiction":
st.title("Modèle de Prédiction de Prix de Voitures")
st.markdown("Saisissez les caractéristiques de la voiture pour prédire son prix")

symboling = st.number_input('symboling', min_value=-3, max_value=3, value=0)
wheel_base = st.number_input('wheel_base', min_value=86.6, max_value=120.9, value=100.0)
length = st.number_input('length', min_value=141.1, max_value=208.1, value=178.2)
width = st.number_input('width', min_value=60.3, max_value=72.3, value=65.5)
height = st.number_input('height', min_value=47.8, max_value=59.8, value=53.1)
curb_weight = st.number_input('curb_weight', min_value=1488, max_value=4066, value=2548)
engine_size = st.number_input('engine_size', min_value=61, max_value=326, value=128)
compression_ratio = st.number_input('compression_ratio', min_value=7.0, max_value=23.0, value=9.0)
city_mpg = st.number_input('city_mpg', min_value=13, max_value=49, value=19)
highway_mpg = st.number_input('highway_mpg', min_value=16, max_value=54, value=25)

if st.button('Predict'):
    pred = inference(symboling, wheel_base, length, width, height, curb_weight,
                        engine_size, compression_ratio, city_mpg, highway_mpg)
    resultat = f"Le prix de la voiture est de : {pred[0] : .2f} $"
    st.success(resultat)
