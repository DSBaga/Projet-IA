import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.express as px

st.set_page_config(page_title="Data_wage", page_icon="💻", layout="centered", initial_sidebar_state="auto")

# Page 1 : Home
st.title("Bienvenue sur l'Application de Prédiction de Salaires en Data")

# Rajouter une image
st.image("https://datasciencedojo.com/wp-content/uploads/Data-scientist-salaries.jpg")

# Charger les données
@st.cache_data
def load_data():  # Fonction pour charger les données
    wage = pd.read_csv('Data/Data_Science_Salaries.csv', sep=';')  # Charger les données
    # Supprimer les lignes avec des valeurs manquantes
    wage = wage.dropna()
    return wage  # Retourner les données
wage = load_data()  # Charger les données()

# Page 3 : Visualisations

#elif selected == "Visualisations":
st.title("Visualisations Interactives")

st.subheader("Diagramme à Barres Interactif")

categorical_variable = st.selectbox("Choisir la variable catégorielle", wage.select_dtypes(include=['object']).columns) # Choisir la variable catégorielle
numerical_variable = st.selectbox("Choisir la variable numérique", wage.select_dtypes(include=['number']).columns) # Choisir la variable numérique

# Calculer la moyenne des variables numériques par catégorie
mean_values = wage.groupby(categorical_variable)[numerical_variable].mean().reset_index()

fig = px.bar(wage, x=categorical_variable, y=numerical_variable, title=f"{numerical_variable} vs {categorical_variable} ")
st.plotly_chart(fig)

# Charger le modèle
@st.cache_resource
def load_model():
    model_path = "Models/Data_Science_Wage_Model.pkl"
    return joblib.load(model_path)

loaded_model = load_model()


# Encoder les variables catégorielles avec LabelEncoder
label_encoders = {}
categorical_columns = ['Job_Title', 'Employment_Type', 'Experience_Level', 'Expertise_Level',
                    'Company_Location', 'Employee_Residence', 'Company_Size', 'Year']

for col in categorical_columns:
    le = LabelEncoder()
    wage[col] = le.fit_transform(wage[col])
    label_encoders[col] = le

# Page 2 : Data

# Définir la fonction d'inférence pour le modèle de prédiction
def inference(Job_Title, Employment_Type, Experience_Level, Expertise_Level,
            Company_Location, Employee_Residence, Company_Size, Year):
    new_data = pd.DataFrame({ 
        'Job_Title': [Job_Title],
        'Employment_Type': [Employment_Type],
        'Experience_Level': [Experience_Level],
        'Expertise_Level': [Expertise_Level],
        'Company_Location': [Company_Location],
        'Employee_Residence': [Employee_Residence],
        'Company_Size': [Company_Size],
        'Year': [Year]
    })
    
    # Encoder les nouvelles données avec gestion des valeurs inconnues
    for col in categorical_columns:
        le = label_encoders[col]
        if new_data[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, new_data[col].iloc[0])
        new_data[col] = le.transform(new_data[col])
    
    pred = loaded_model.predict(new_data)
    return pred[0]

# Définir les modalités pour chaque variable de type "objet"
job_titles = wage['Job_Title'].unique()
employment_types = wage['Employment_Type'].unique()
experience_levels = wage['Experience_Level'].unique()
expertise_levels = wage['Expertise_Level'].unique()
company_locations = wage['Company_Location'].unique()
employee_residences = wage['Employee_Residence'].unique()   
company_sizes = wage['Company_Size'].unique()
years = wage['Year'].unique()

# Modèle de prédiction
st.title("Modèle de Prédiction de Salaires en Data ")
st.markdown("Ce modèle de prédiction est basé sur les données de salaires en Data Science")

# Inputs de l'utilisateur pour la prédiction du salaire
Job_Title = st.selectbox("Job_Title", label_encoders['Job_Title'].inverse_transform(job_titles))
Employment_Type = st.selectbox("Employment_Type", label_encoders['Employment_Type'].inverse_transform(employment_types))
Experience_Level = st.selectbox("Experience_Level", label_encoders['Experience_Level'].inverse_transform(experience_levels))
Expertise_Level = st.selectbox("Expertise_Level", label_encoders['Expertise_Level'].inverse_transform(expertise_levels))
Company_Location = st.selectbox("Company_Location", label_encoders['Company_Location'].inverse_transform(company_locations))
Employee_Residence = st.selectbox("Employee_Residence", label_encoders['Employee_Residence'].inverse_transform(employee_residences))
Company_Size = st.selectbox("Company_Size", label_encoders['Company_Size'].inverse_transform(company_sizes))
Year = st.selectbox("Year", label_encoders['Year'].inverse_transform(years))

# Taux de change USD/EUR
exchange_rate = 0.85

# Bouton pour prédire le salaire 
if st.button("Prédire le Salaire"):
    prediction_usd = inference(Job_Title, Employment_Type, Experience_Level, Expertise_Level, 
                            Company_Location, Employee_Residence, Company_Size, Year)
    prediction_eur = prediction_usd * exchange_rate
    st.success(f"Le salaire prédit est : {prediction_usd:.2f} $ soit {prediction_eur:.2f} €")
    st.balloons()
