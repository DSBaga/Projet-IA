import streamlit as st
from streamlit_option_menu import option_menu

# Defonir les configurations de la page
st.set_page_config(page_title="HOME", page_icon="🖐️", 
                    layout="centered", initial_sidebar_state="auto")

# Create the home page
choix = st.sidebar.radio("Menu",["Description","Documentation"])
if choix == "Description":
    #st.write("Description")
        # Créer la page d'accueil
    st.title('🖐️ Bienvenue dans la mutlti-application de Machine Learning')
    st.write('📝 Auteur : Projet réalisé par Djolo')

    #Présentation de l'application
    st.header('Description')
    #st.write("Cette application est une démonstration de Streamlit, une application web pour le déploiement de modèles de Machine Learning.")
    
    st.write("Cette multi-application contient quatre applications de Machine Learning :")

    st.write("1. Application Know Your Worth")
    st.write("🎯L'objectif est de construire une application de prédiction de salaires en Data. L'utilisateur renseigne les informations relatives à un poste donné et l'application lui retourne le salaire correspondant à ce poste.")
    
    st.write("2. Application de Robot Advisor")
    st.write("🎯Cette application permet de recommander des portefeuilles d'actifs financiers en fonction de la tolérance au risque de l'investisseur.")

    st.write("3. Application Films")
    st.write("🎯L'objectif de ce projet est de construire une application de système de recommandations de films. L'utilisateur de cette application renseigne les informations relatives à un film donné et l'application lui retourne les n films les plus similaires à ce film donné.")
    
    st.write("4. Application Voitures")
    st.write("🎯Cette application permet de prédire le prix des voitures. L'utilisateur de cette application renseigne les informations relatives à une voiture donnée et celle-ci lui retourne le prix de cette voiture.")
    

else:
    st.write("Documentation")

