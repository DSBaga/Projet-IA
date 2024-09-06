import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json

# Définir les configurations de la page
st.set_page_config(page_title="Movies_AppS", page_icon="🎬", layout="centered", initial_sidebar_state="auto")

## Fonctions de chargement et de nettoyage des données de films 
def load_clean_movie_data(movie_file):
    data = pd.read_csv(movie_file)
    data.dropna(inplace=True)
    data[['date_of_release', 'country_of_release']] = data['released'].str.extract(r'(\w+ \d+, \d+) \(([^)]+)\)')
    data.drop(['released', 'date_of_release'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data = data[[
        'name',
        'genre',
        'year',
        'director',
        'writer',
        'star',
        'company',
        'country_of_release',
    ]]
    data['year'] = data['year'].astype('str')
    data['cat_features'] = data[data.columns].apply(lambda x: ' '.join(x), axis=1)

    return data

def get_recommendations(title, df, sim, count=10):
    # Obtenir l'indice de ligne du titre spécifié dans le DataFrame
    index = df.index[df['name'].str.lower() == title.lower()]
    
    # Retourner une liste vide s'il n'y a aucune entrée pour le titre spécifié
    if len(index) == 0:
        return []

    # Vérifier si l'indice est dans les limites de la matrice de similarité
    if index[0] >= len(sim):
        return []

    # Obtenir la ligne correspondante dans la matrice de similarité
    similarities = list(enumerate(sim[index[0]]))
    
    # Trier les scores de similarité dans cette ligne par ordre décroissant
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Obtenir les n meilleures recommandations, en ignorant la première entrée de la liste car
    # elle correspond au titre lui-même (et a donc une similarité de 1.0)
    top_recs = recommendations[1:count + 1]

    # Générer une liste de titres à partir des indices dans top_recs
    titles = []

    for i in range(len(top_recs)):
        # Vérifier si l'indice est dans les limites du DataFrame
        if top_recs[i][0] < len(df):
            title = df.iloc[top_recs[i][0]]['name']
            titles.append(title)

    return titles

# Fonction pour charger les animations Lottie depuis une URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Charger les animations Lottie
with open("Autres/Animation-Hello.json", "r",encoding="utf-8" ) as f:
    lottie_home = json.load(f)
#lottie_home = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")  # Exemple d'URL pour une animation Lottie d'accueil
lottie_data = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_t26law.json")  # Exemple d'URL pour une animation Lottie de recommandation
#lottie_recommendation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")  # Exemple d'URL pour une animation Lottie de recommandation

# Charger l'animation Lottie directement depuis un fichier local
with open("Autres/Animation.json", "r",encoding="utf-8" ) as f:
    lottie_recommendation = json.load(f)


# Définir les pages disponibles dans le menu
#with st.sidebar:
    #selected = option_menu(
        #menu_title="Menu",
        #options=["Home", "Data","Modèle de Prédiction"],
        #icons=["house", "database", "robot"],
        #menu_icon="cast",
        #default_index=0,
    #)

# Charger et nettoyer le DataFrame
movie_data = load_clean_movie_data("Data/movies.csv")

# Page 1 : Home
#if selected == "Home":
col1, col2 = st.columns([3, 1])  # Ajuster la proportion des colonnes selon les besoins
with col1:
    st.title("Bienvenue sur l'Application de recommandation de Films")
    #st.subheader('Cette application est réalisée par Djolo')
    #st.markdown('Le but de cette application est de recommander des films en fonction des films similaires')
    st.image("https://assets.isu.pub/document-structure/230111204101-6aebb8bebfa4befa4a1665901a8c500d/v1/574b1cda9f152ea6756be6b12eb0e3a3.jpeg", use_column_width=True)
with col2:
    st_lottie(lottie_home, height=300, key="home")


# Page 2 : Data
#if selected == "Data":
col1, col2 = st.columns([3, 1])  # Ajuster la proportion des colonnes selon les besoins
with col1:
    st.title('Données des films')
    st.write(movie_data)
with col2:
    st_lottie(lottie_data, height=300, key="data")


# Page 3 : Modèle de Prédiction
#if selected == "Modèle de Prédiction":
col1, col2 = st.columns([3, 1])  # Ajuster la proportion des colonnes selon les besoins
with col1:
    # Affichage du titre de l'application
    st.title('Film Recommendation App')

    # Inputs de l'utilisateur
    name = st.selectbox('Nom du film', movie_data['name'].unique())
    num_recommendations = st.number_input('Nombre de films à recommander', min_value=1, value=5)

    # Bouton pour obtenir les recommandations
    if st.button('Obtenir les recommandations'):
        # Charger la matrice de similarité précalculée
        similarity_matrix_loaded = np.load('Models/Movies_final_model.npy')

        # Utiliser la matrice de similarité précalculée pour les recommandations
        recommendations = get_recommendations(
            title=name, df=movie_data, 
            sim=similarity_matrix_loaded, 
            count=num_recommendations
        )
        st.write('Films recommandés :', recommendations)
with col2:
    st_lottie(lottie_recommendation, height=300, key="recommendation")
