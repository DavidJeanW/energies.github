# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn.tree  import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error  

page = st.sidebar.radio (label = "",
          options =['Présentation & Objectifs',
                    'Jeux de données',
                    'Axes d\'analyse',
                    'Prédiction de la consommation',
                    'Analyse des modèles',
                    'Conclusion'])


# -------- Entete fixe ----------------------
#@st.cache
# affichage d'une photo pour entête fixe
photo = plt.imread('photo_energie2.jpg') 
st.image(photo)

st.header("La consommation énergétique, enjeu présent et futur")



# Datasets
fusion = pd.read_csv('fusion.csv', sep=',')

conso_reg_jour_20 = pd.read_csv('conso_reg_jour_20.csv', sep=',')
conso_nat_jour_20 = pd.read_csv('conso_nat_jour_20.csv', sep=',')
conso_reg_mois_20 = pd.read_csv('conso_reg_mois_20.csv', sep=',')
conso_nat_mois_20 = pd.read_csv('conso_nat_mois_20.csv', sep=',')

conso_reg_jour_19 = pd.read_csv('conso_reg_jour_19.csv', sep=',')
conso_nat_jour_19 = pd.read_csv('conso_nat_jour_19.csv', sep=',')
conso_reg_mois_19 = pd.read_csv('conso_reg_mois_19.csv', sep=',')
conso_nat_mois_19 = pd.read_csv('conso_nat_mois_19.csv', sep=',')

y_mean_2016_to_2018 = pd.read_csv('y_mean_2016_to_2018.csv', sep = ',')
y_mean_2016_to_2019 = pd.read_csv('y_mean_2016_to_2019.csv', sep = ',')
                 
# Fonctions
@st.cache
def score_train(model,X_train,X_test,y_train, y_test):
    model.fit(X_train, y_train)
    return round(model.score(X_train, y_train),2)
@st.cache
def score_test(model,X_train,X_test,y_train, y_test):
    model.fit(X_train, y_train)
    return round(model.score(X_test, y_test),2)
@st.cache
def MSE_test(model,X_train,X_test,y_train, y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return round(mean_squared_error(y_test, y_pred),3)
@st.cache
def MSE_train(model,X_train,X_test,y_train, y_test):
    model.fit(X_train,y_train)
    return round(mean_squared_error(y_test, y_pred),3)

def dataviz_bar_19(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.set(style='whitegrid')
    barWidth = 0.28
    x1 = range(len(data))
    x2 = [r + barWidth for r in x1 ]
    x3 = [r + barWidth for r in x2 ]
    ax.barlist = plt.bar(x1, data['moyenne_conso_16_a_18'], width = barWidth, 
                  label = "Consommation moyenne des années 2016 à 2018")
    plt.bar(x2, data['conso_prédite'], width = barWidth, label = 'Consommation prédite de 2019')
    plt.bar(x3, data['conso_observée_2019'], width = barWidth, label = 'Consommation réelle de 2019')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], ['Janvier','Février','Mars','Avril','Mai','Juin',
                                         'Juillet','Août','Septembre','Octobre','Novembre','Décembre'],rotation='60')
    plt.legend(fontsize=10)
    plt.xlabel('Mois')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de l'année 2019");
    return st.pyplot(fig)

def dataviz_bar_20(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.set(style='whitegrid')
    barWidth = 0.28
    x1 = range(len(data))
    x2 = [r + barWidth for r in x1 ]
    x3 = [r + barWidth for r in x2 ]
    ax.barlist = plt.bar(x1, data['moyenne_conso_16_a_19'], width = barWidth, 
                  label = "Consommation moyenne des années 2016 à 2019")
    plt.bar(x2, data['conso_prédite'], width = barWidth, label = 'Consommation prédite de 2020')
    plt.bar(x3, data['conso_observée_2020'], width = barWidth, label = 'Consommation réelle de 2020')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], ['Janvier','Février','Mars','Avril','Mai','Juin',
                                         'Juillet','Août','Septembre','Octobre','Novembre','Décembre'],rotation='60')
    plt.legend(fontsize=10)
    plt.xlabel('Mois')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de l'année 2020");
    return st.pyplot(fig)

def dataviz_20_mois_31(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_19'], label = "Consommation moyenne des années 2016 à 2019")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2020')
    plt.plot(x, data['conso_observée_2020'], label = 'Consommation réelle de 2020')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2020 par zone géographique");
    st.pyplot(fig)

def dataviz_20_mois_30(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_19'], label = "Consommation moyenne des années 2016 à 2019")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2020')
    plt.plot(x, data['conso_observée_2020'], label = 'Consommation réelle de 2020')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2020 par zone géographique");
    st.pyplot(fig)

def dataviz_20_mois_28(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_19'], label = "Consommation moyenne des années 2016 à 2019")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2020')
    plt.plot(x, data['conso_observée_2020'], label = 'Consommation réelle de 2020')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2020 par zone géographique");
    st.pyplot(fig)

def dataviz_19_mois_31(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_18'], label = "Consommation moyenne des années 2016 à 2018")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2019')
    plt.plot(x, data['conso_observée_2019'], label = 'Consommation réelle de 2019')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2019 par zone géographique");
    st.pyplot(fig)

def dataviz_19_mois_30(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_18'], label = "Consommation moyenne des années 2016 à 2018")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2019')
    plt.plot(x, data['conso_observée_2019'], label = 'Consommation réelle de 2019')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2019 par zone géographique");
    st.pyplot(fig)

def dataviz_19_mois_28(data):
    fig, ax = plt.subplots(figsize=(10,3))      
    x = range(len(data))
    ax = plt.plot(x, data['moyenne_conso_16_a_18'], label = "Consommation moyenne des années 2016 à 2018")
    plt.plot(x, data['conso_prédite'], label = 'Consommation prédite de 2019')
    plt.plot(x, data['conso_observée_2019'], label = 'Consommation réelle de 2019')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])
    plt.legend(fontsize = 9)
    plt.xlabel('Jours')
    plt.ylabel('MW total')
    plt.title("Prédiction de la consommation d'énergies de 2019 par zone géographique");
    st.pyplot(fig)

@st.cache
def geo(conso,region):
    conso_reg = conso.loc[conso['insee'] == region]
    return conso_reg    

def month(conso,mois):
    conso_month = conso.loc[conso['Mois'] == mois]
    return conso_month
    

# Codage Streamlit
if page == 'Prédiction de la consommation':
    photo = plt.imread('energie1.jpg')
    st.image(photo)
    st.header('Prédiction de la consommation')
    st.subheader('Prédiction sur une année complète')
    st.markdown("\n")
    st.markdown('Quelle année souhaitez-vous prédire?')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        annee = st.selectbox("",options=["2019",'2020'])
        
    # 2020
    if annee == '2020':
        # Dataprocessing 2020
        from datetime import date
        fusion['Date'] = pd.to_datetime(fusion['Date'])
        fusion.index = fusion['Date']
        fusion['Date'] = pd.to_numeric(fusion['Date'])

        target = fusion['total_conso']
        data = fusion.drop(['total_conso'], axis = 1)

        X_train = data.loc[data.index < '2020-01-01']
        X_test =  data.loc[data.index >='2020-01-01']
        y_train = fusion.loc[fusion.index < '2020-01-01']['total_conso']
        y_test = fusion.loc[fusion.index >='2020-01-01']['total_conso']
        y_pred = y_mean_2016_to_2019
            
        # Standardisation?
        st.markdown('**Sélection des paramètres du modèle**')
        scale = st.checkbox(label='Standardiser les données')
        if scale:
            scaler = preprocessing.StandardScaler().fit(fusion)
            fusion[fusion.columns] = pd.DataFrame(scaler.transform(fusion), index = fusion.index)
            target = fusion['total_conso']
            data = fusion.drop(['total_conso'], axis = 1)
            X_train = data.loc[data.index < '2020-01-01']
            X_test = data.loc[data.index >='2020-01-01']
            y_train = fusion.loc[fusion.index < '2020-01-01']['total_conso']
            y_test = fusion.loc[fusion.index >='2020-01-01']['total_conso']  
            y_mean_2016_to_2019 = preprocessing.StandardScaler().fit(y_mean_2016_to_2019).transform(y_mean_2016_to_2019)
            y_pred = y_mean_2016_to_2019
            
            # Machine Learning 20
        model = st.radio(label="Sélectionnez l'algorithme à tester :", 
                             options=['LinearRegression','RidgeCV','LassoCV',
                                      'ElasticNet','DecisionTreeRegressor'])
                
        if model == 'LinearRegression':
                
            st.markdown('**Score du modèle choisi**')
            st.markdown('LinearRegression :')
            st.write('score train =',
                score_train(LinearRegression(),X_train,X_test,y_train,y_test),
                '/ score test =',
                score_test(LinearRegression(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(LinearRegression(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(LinearRegression(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 4 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
        
        if model == 'RidgeCV':
            st.markdown('**Score du modèle choisi**')
            st.markdown('RidgeCV :')
            st.write('score train =',
             score_train(RidgeCV(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(RidgeCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(RidgeCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(RidgeCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 4 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        if model == 'LassoCV':
            st.markdown('**Score du modèle choisi**')
            st.markdown('LassoCV :')
            st.write('score train =',
             score_train(LassoCV(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(LassoCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(LassoCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(LassoCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 4 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)') 
            
        if model == 'ElasticNet':
            st.markdown('**Score du modèle choisi**')
            st.markdown('ElasticNet :')
            st.write('score train =',
             score_train(ElasticNetCV(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(ElasticNetCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(ElasticNetCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(ElasticNetCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 4 années qui précèdent l'année prédite.*")            
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        if model == 'DecisionTreeRegressor':
            st.markdown('**Score du modèle choisi**')
            st.markdown('DecisionTreeRegressor :')
            st.write('score train =',
             score_train(DecisionTreeRegressor(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(DecisionTreeRegressor(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(DecisionTreeRegressor(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(DecisionTreeRegressor(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 4 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        #Dataviz' 20
        st.markdown('\n')
        st.markdown('**Représentation graphique des prédictions**')
        time = st.radio(label="Sélectionnez la période à visualiser :", 
                 options = ['Année','Mois'])
           
        # Année/Région
        if time == 'Année':
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox('Sélectionnez la zone géographique :',
        options=["Territoire national",'Pays de la Loire', 'Normandie', 'Grand Est',
       'Bourgogne-Franche-Comté', 'Centre-Val de Loire','Auvergne-Rhône-Alpes', 
       'Bretagne', 'Occitanie', 'Île-de-France',"Provence-Alpes-Côte d'Azur",
       'Hauts-de-France','Nouvelle-Aquitaine'])
            if region == "Territoire national":
                    dataviz_bar_20(conso_nat_mois_20)
            if region == 'Pays de la Loire':
                    dataviz_bar_20(geo(conso_reg_mois_20,52))
            if region == 'Normandie':
                    dataviz_bar_20(geo(conso_reg_mois_20,28)) 
            if region == 'Grand Est':
                    dataviz_bar_20(geo(conso_reg_mois_20,44))    
            if region == 'Bourgogne-Franche-Comté':
                    dataviz_bar_20(geo(conso_reg_mois_20,27))    
            if region == 'Centre-Val de Loire':
                    dataviz_bar_20(geo(conso_reg_mois_20,24))  
            if region == 'Auvergne-Rhône-Alpes':
                    dataviz_bar_20(geo(conso_reg_mois_20,84))      
            if region == 'Bretagne':
                    dataviz_bar_20(geo(conso_reg_mois_20,53))      
            if region == 'Occitanie':
                    dataviz_bar_20(geo(conso_reg_mois_20,76))  
            if region == 'Île-de-France':
                    dataviz_bar_20(geo(conso_reg_mois_20,11))      
            if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_bar_20(geo(conso_reg_mois_20,93))  
            if region == 'Hauts-de-France':
                    dataviz_bar_20(geo(conso_reg_mois_20,32))  
            if region == 'Nouvelle-Aquitaine':
                    dataviz_bar_20(geo(conso_reg_mois_20,75)) 
        
        # Mois/Région
        if time == 'Mois':
            col1, col2 = st.columns(2)
            
            with col1:
                region = st.selectbox('Sélectionnez la zone géographique :',
        options=["Territoire national",'Pays de la Loire', 'Normandie', 'Grand Est',
       'Bourgogne-Franche-Comté', 'Centre-Val de Loire','Auvergne-Rhône-Alpes', 
       'Bretagne', 'Occitanie', 'Île-de-France',"Provence-Alpes-Côte d'Azur",
       'Hauts-de-France','Nouvelle-Aquitaine'])
            
            with col2:
                mois = st.selectbox('Sélectionnez le mois :',
                options=['Janvier','Février','Mars','Avril','Mai','Juin',
                'Juillet','Août','Septembre','Octobre','Novembre','Décembre'])
                
            if mois == 'Janvier':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,1))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),1))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 1],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),1))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),1))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),1))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),1))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),1))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),1))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),1))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),1))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),1))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),1))    
                                    
            if mois == 'Février':
                if region == 'Territoire national':
                    dataviz_20_mois_28(month(conso_nat_jour_20,2))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,52),2))
                if region == 'Normandie':
                    dataviz_20_mois_28(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 2],28))
                if region == 'Grand Est':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,44),2))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,27),2))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,24),2))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,84),2))    
                if region == 'Bretagne':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,53),2))    
                if region == 'Occitanie':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,76),2))    
                if region == 'Île-de-France':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,11),2))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,93),2))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,32),2))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_28(month(geo(conso_reg_jour_20,75),2))
            
            if mois == 'Mars':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,3))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),3))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 3],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),3))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),3))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),3))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),3))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),3))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),3))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),3))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),3))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),3))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),3))
            
            if mois == 'Avril':
                if region == 'Territoire national':
                    dataviz_20_mois_30(month(conso_nat_jour_20,4))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,52),4))
                if region == 'Normandie':
                    dataviz_20_mois_30(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 4],28))
                if region == 'Grand Est':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,44),4))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,27),4))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,24),4))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,84),4))    
                if region == 'Bretagne':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,53),4))    
                if region == 'Occitanie':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,76),4))    
                if region == 'Île-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,11),4))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,93),4))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,32),4))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,75),4))
            
            if mois == 'Mai':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,5))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),5))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 5],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),5))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),5))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),5))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),5))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),5))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),5))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),5))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),5))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),5))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),5))
            
            if mois == 'Juin':
                if region == 'Territoire national':
                    dataviz_20_mois_30(month(conso_nat_jour_20,6))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,52),6))
                if region == 'Normandie':
                    dataviz_20_mois_30(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 6],28))
                if region == 'Grand Est':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,44),6))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,27),6))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,24),6))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,84),6))    
                if region == 'Bretagne':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,53),6))    
                if region == 'Occitanie':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,76),6))    
                if region == 'Île-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,11),6))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,93),6))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,32),6))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,75),6))
            
            if mois == 'Juillet':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,7))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),7))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 7],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),7))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),7))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),7))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),7))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),7))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),7))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),7))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),7))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),7))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),7))
            
            if mois == 'Août':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,8))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),8))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 8],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),8))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),8))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),8))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),8))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),8))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),8))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),8))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),8))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),8))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),8))
            
            if mois == 'Septembre':
                if region == 'Territoire national':
                    dataviz_20_mois_30(month(conso_nat_jour_20,9))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,52),9))
                if region == 'Normandie':
                    dataviz_20_mois_30(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 9],28))
                if region == 'Grand Est':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,44),9))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,27),9))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,24),9))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,84),9))    
                if region == 'Bretagne':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,53),9))    
                if region == 'Occitanie':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,76),9))    
                if region == 'Île-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,11),9))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,93),9))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,32),9))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,75),9)) 
            
            if mois == 'Octobre':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,10))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),10))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 10],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),10))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),10))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),10))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),10))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),10))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),10))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),10))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),10))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),10))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),10))
            
            if mois == 'Novembre':
                if region == 'Territoire national':
                    dataviz_20_mois_30(month(conso_nat_jour_20,11))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,52),11))
                if region == 'Normandie':
                    dataviz_20_mois_30(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 11],28))
                if region == 'Grand Est':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,44),11))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,27),11))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,24),11))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,84),11))    
                if region == 'Bretagne':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,53),11))    
                if region == 'Occitanie':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,76),11))    
                if region == 'Île-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,11),11))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,93),11))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,32),11))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_30(month(geo(conso_reg_jour_20,75),11))
            
            if mois == 'Décembre':
                if region == 'Territoire national':
                    dataviz_20_mois_31(month(conso_nat_jour_20,12))
                if region == 'Pays de la Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,52),12))
                if region == 'Normandie':
                    dataviz_20_mois_31(geo(conso_reg_jour_20.loc[conso_reg_jour_20['Mois'] == 12],28))
                if region == 'Grand Est':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,44),12))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,27),12))    
                if region == 'Centre-Val de Loire':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,24),12))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,84),12))    
                if region == 'Bretagne':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,53),12))    
                if region == 'Occitanie':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,76),12))    
                if region == 'Île-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,11),12))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,93),12))    
                if region == 'Hauts-de-France':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,32),12))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_20_mois_31(month(geo(conso_reg_jour_20,75),12))
            
            
            
            
            
    
            
        # 2019
    if annee == '2019':
        # Dataprocessing 2019
        fusion = fusion.loc[fusion['Date'] < '2020-01-01']
        from datetime import date
        fusion['Date'] = pd.to_datetime(fusion['Date'])
        fusion.index = fusion['Date']
        fusion['Date'] = pd.to_numeric(fusion['Date'])

        target = fusion['total_conso']
        data = fusion.drop(['total_conso'], axis = 1)

        X_train = data.loc[data.index < '2019-01-01']
        X_test =  data.loc[data.index >='2019-01-01']
        y_train = fusion.loc[fusion.index < '2019-01-01']['total_conso']
        y_test = fusion.loc[fusion.index >='2019-01-01']['total_conso']
        y_pred = y_mean_2016_to_2018
            
        # Standardisation?
        st.markdown('**Sélection des paramètres du modèle**')
        scale = st.checkbox(label='Standardiser les données')
        if scale:
            scaler = preprocessing.StandardScaler().fit(fusion)
            fusion[fusion.columns] = pd.DataFrame(scaler.transform(fusion), index = fusion.index)
            target = fusion['total_conso']
            data = fusion.drop(['total_conso'], axis = 1)
            X_train = data.loc[data.index < '2019-01-01']
            X_test = data.loc[data.index >='2019-01-01']
            y_train = fusion.loc[fusion.index < '2019-01-01']['total_conso']
            y_test = fusion.loc[fusion.index >='2019-01-01']['total_conso']  
            y_mean_2016_to_2018 = preprocessing.StandardScaler().fit(y_mean_2016_to_2018).transform(y_mean_2016_to_2018)
            y_pred = y_mean_2016_to_2018
    
        # Machine Learning 19
        col1, col2 = st.columns(2)
        with col1:
            model = st.radio(label="Sélectionnez l'algorithme à tester :", 
                         options=['LinearRegression','RidgeCV','LassoCV',
                                      'ElasticNet','DecisionTreeRegressor'])
                        
        if model == 'LinearRegression':
            st.markdown('**Score du modèle choisi**')
            st.markdown('LinearRegression :')
            st.write('score train =',
             score_train(LinearRegression(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(LinearRegression(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(LinearRegression(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(LinearRegression(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 3 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        if model == 'RidgeCV':
            st.markdown('**Score du modèle choisi**')
            st.markdown('RidgeCV :')
            st.write('score train =',
             score_train(RidgeCV(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(RidgeCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(RidgeCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(RidgeCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 3 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        if model == 'LassoCV':
            st.markdown('**Score du modèle choisi**')
            st.markdown('LassoCV :')
            st.write('score train =',
             score_train(LassoCV(),X_train,X_test,y_train,y_test),
            '/ score test =',
            score_test(LassoCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(LassoCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(LassoCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 3 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)') 
            
        if model == 'ElasticNet':
            st.markdown('**Score du modèle choisi**')
            st.markdown('ElasticNet :')
            st.write('score train =',
             score_train(ElasticNetCV(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(ElasticNetCV(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(ElasticNetCV(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(ElasticNetCV(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 3 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        if model == 'DecisionTreeRegressor':
            st.markdown('**Score du modèle choisi**')
            st.markdown('DecisionTreeRegressor :')
            st.write('score train =',
             score_train(DecisionTreeRegressor(),X_train,X_test,y_train,y_test),
             '/ score test =',
             score_test(DecisionTreeRegressor(),X_train,X_test,y_train,y_test))
            st.write('MSE train* =',
            MSE_train(DecisionTreeRegressor(),X_train,X_test,y_train,y_test),
            '/ MSE test =',
            MSE_test(DecisionTreeRegressor(),X_train,X_test,y_train,y_test))
            st.write('*'"*MSE train prend comme variable de prédiction (y_pred) la moyenne* "
                        "*des 3 années qui précèdent l'année prédite.*")
            st.code('MSE = np.mean((y - y_pred)**2)')
            
        #Dataviz'
        st.markdown('\n')
        st.markdown('**Représentation graphique des prédictions**')
        time = st.radio(label="Sélectionnez la période à visualiser :", 
                 options = ['Année','Mois'])
        
        if time == 'Année':
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox('Sélectionnez la zone géographique :',
        options=["Territoire national",'Pays de la Loire', 'Normandie', 'Grand Est',
       'Bourgogne-Franche-Comté', 'Centre-Val de Loire','Auvergne-Rhône-Alpes', 
       'Bretagne', 'Occitanie', 'Île-de-France',"Provence-Alpes-Côte d'Azur",
       'Hauts-de-France','Nouvelle-Aquitaine'])
            if region == "Territoire national":
                    dataviz_bar_19(conso_nat_mois_19)
            if region == 'Pays de la Loire':
                    dataviz_bar_19(geo(conso_reg_mois_19,52))
            if region == 'Normandie':
                    dataviz_bar_19(geo(conso_reg_mois_19,28)) 
            if region == 'Grand Est':
                    dataviz_bar_19(geo(conso_reg_mois_19,44))    
            if region == 'Bourgogne-Franche-Comté':
                    dataviz_bar_19(geo(conso_reg_mois_19,27))    
            if region == 'Centre-Val de Loire':
                    dataviz_bar_19(geo(conso_reg_mois_19,24))  
            if region == 'Auvergne-Rhône-Alpes':
                    dataviz_bar_19(geo(conso_reg_mois_19,84))      
            if region == 'Bretagne':
                    dataviz_bar_19(geo(conso_reg_mois_19,53))      
            if region == 'Occitanie':
                    dataviz_bar_19(geo(conso_reg_mois_19,76))  
            if region == 'Île-de-France':
                    dataviz_bar_19(geo(conso_reg_mois_19,11))      
            if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_bar_19(geo(conso_reg_mois_19,93))  
            if region == 'Hauts-de-France':
                    dataviz_bar_19(geo(conso_reg_mois_19,32))  
            if region == 'Nouvelle-Aquitaine':
                    dataviz_bar_19(geo(conso_reg_mois_19,75))
          
            
        if time == 'Mois':
            col1, col2 = st.columns(2)
            
            with col1:
                region = st.selectbox('Sélectionnez la zone géographique :',
        options=["Territoire national",'Pays de la Loire', 'Normandie', 'Grand Est',
       'Bourgogne-Franche-Comté', 'Centre-Val de Loire','Auvergne-Rhône-Alpes', 
       'Bretagne', 'Occitanie', 'Île-de-France',"Provence-Alpes-Côte d'Azur",
       'Hauts-de-France','Nouvelle-Aquitaine'])
            
            with col2:
                mois = st.selectbox('Sélectionnez le mois :',
                options=['Janvier','Février','Mars','Avril','Mai','Juin',
                'Juillet','Août','Septembre','Octobre','Novembre','Décembre'])
                
            if mois == 'Janvier':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,1))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),1))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),1))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),1))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),1))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),1))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),1))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),1))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),1))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),1))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),1))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),1))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),1))    
                                    
            if mois == 'Février':
                if region == 'Territoire national':
                    dataviz_19_mois_28(month(conso_nat_jour_19,2))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,52),2))
                if region == 'Normandie':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,28),2))
                if region == 'Grand Est':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,44),2))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,27),2))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,24),2))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,84),2))    
                if region == 'Bretagne':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,53),2))    
                if region == 'Occitanie':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,76),2))    
                if region == 'Île-de-France':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,11),2))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,93),2))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,32),2))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_28(month(geo(conso_reg_jour_19,75),2))
            
            if mois == 'Mars':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,3))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),3))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),3))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),3))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),3))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),3))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),3))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),3))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),3))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),3))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),3))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),3))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),3))
            
            if mois == 'Avril':
                if region == 'Territoire national':
                    dataviz_19_mois_30(month(conso_nat_jour_19,4))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,52),4))
                if region == 'Normandie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,28),4))
                if region == 'Grand Est':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,44),4))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,27),4))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,24),4))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,84),4))    
                if region == 'Bretagne':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,53),4))    
                if region == 'Occitanie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,76),4))    
                if region == 'Île-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,11),4))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,93),4))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,32),4))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,75),4))
            
            if mois == 'Mai':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,5))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),5))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),5))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),5))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),5))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),5))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),5))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),5))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),5))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),5))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),5))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),5))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),5))
            
            if mois == 'Juin':
                if region == 'Territoire national':
                    dataviz_19_mois_30(month(conso_nat_jour_19,6))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,52),6))
                if region == 'Normandie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,28),6))
                if region == 'Grand Est':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,44),6))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,27),6))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,24),6))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,84),6))    
                if region == 'Bretagne':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,53),6))    
                if region == 'Occitanie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,76),6))    
                if region == 'Île-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,11),6))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,93),6))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,32),6))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,75),6))
            
            if mois == 'Juillet':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,7))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),7))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),7))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),7))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),7))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),7))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),7))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),7))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),7))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),7))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),7))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),7))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),7))
            
            if mois == 'Août':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,8))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),8))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),8))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),8))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),8))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),8))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),8))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),8))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),8))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),8))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),8))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),8))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),8))
            
            if mois == 'Septembre':
                if region == 'Territoire national':
                    dataviz_19_mois_30(month(conso_nat_jour_19,9))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,52),9))
                if region == 'Normandie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,28   ),9))
                if region == 'Grand Est':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,44),9))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,27),9))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,24),9))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,84),9))    
                if region == 'Bretagne':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,53),9))    
                if region == 'Occitanie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,76),9))    
                if region == 'Île-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,11),9))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,93),9))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,32),9))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,75),9)) 
            
            if mois == 'Octobre':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,10))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),10))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),10))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),10))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),10))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),10))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),10))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),10))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),10))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),10))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),10))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),10))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),10))
            
            if mois == 'Novembre':
                if region == 'Territoire national':
                    dataviz_19_mois_30(month(conso_nat_jour_19,11))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,52),11))
                if region == 'Normandie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,28),11))
                if region == 'Grand Est':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,44),11))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,27),11))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,24),11))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,84),11))    
                if region == 'Bretagne':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,53),11))    
                if region == 'Occitanie':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,76),11))    
                if region == 'Île-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,11),11))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,93),11))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,32),11))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_30(month(geo(conso_reg_jour_19,75),11))
            
            if mois == 'Décembre':
                if region == 'Territoire national':
                    dataviz_19_mois_31(month(conso_nat_jour_19,12))
                if region == 'Pays de la Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,52),12))
                if region == 'Normandie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,28),12))
                if region == 'Grand Est':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,44),12))    
                if region == 'Bourgogne-Franche-Comté':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,27),12))    
                if region == 'Centre-Val de Loire':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,24),12))    
                if region == 'Auvergne-Rhône-Alpes':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,84),12))    
                if region == 'Bretagne':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,53),12))    
                if region == 'Occitanie':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,76),12))    
                if region == 'Île-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,11),12))    
                if region == "Provence-Alpes-Côte d'Azur":
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,93),12))    
                if region == 'Hauts-de-France':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,32),12))    
                if region == 'Nouvelle-Aquitaine':
                    dataviz_19_mois_31(month(geo(conso_reg_jour_19,75),12))
            
                        
        


