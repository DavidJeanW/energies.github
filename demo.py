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
# le choix de l'utilisateur sera stocké dans page




# -------- Entete fixe ----------------------
#@st.cache
# affichage d'une photo pour entête fixe
photo = plt.imread('photo_energie2.jpg') 
st.image(photo)

st.header("La consommation énergétique, enjeu présent et futur")


if page == 'Présentation & Objectifs':
    
       
    st.markdown("""
                \n\n\n
                """)

    st.markdown("""
                Projet réalisé dans le cadre de la formation Data Analyst de [DataScientest](https://datascientest.com/)
                
                \nPromotion Formation continue janvier 2021
                \nAuteurs :
                        **Simone Mariot** [LinkedIn](https://www.linkedin.com/in/simone-mariot-a0558b72/), 
                        **Julie  Guidez** [LinkedIn](https://www.linkedin.com/in/julie-guidez-a1181b113/),
                        **David Wachowiak** [LinkedIn](https://www.linkedin.com/in/david-wachowiak-32530789/), 
                        **Anthony Foulon** [LinkedIn](https://www.linkedin.com/in/anthony-foulon-761aba22/),

                """)

                
    st.markdown("""
                \n\n\n
                """)
    st.title("Présentation & Objectifs")    
    st.header("1. Projet Energie")
                
    st.markdown("""
                Le projet “Energie”, via la fourniture d’un jeu de données contenant les informations de consommation et 
                production d’électricité en France, a pour objectif d’approfondir les liens étroits entre les différents 
                acteurs du secteur (producteurs, consommateurs, régulateurs…) et le nécessaire équilibre entre la production 
                et la consommation d’électricité.
    """)
    
    st.header("2. Objectifs")        

    st.markdown("""
                Via le jeu de données source de l’ODRE (Open Data Réseaux Energies), contenant l’ensemble des informations 
                de consommation et de production électrique depuis 2013, nous pouvions répondre à **3 pistes de réflexion** proposées par la fiche projet :
                        \n- Analyser les filières de production (par exemple fossiles vs nucléaire vs renouvelables)
                        \n- Identifier une prévision de consommation par région
                        \n- Détailler la consommation et la production des énergies renouvelables, en identifiant par exemple 
                        notre capacité à produire et consommer de manière 100% “renouvelables”
    """)                    
                        
    st.markdown("""                  
             \n Après exploration d’une dizaine de réflexions, dont certaines sont détaillées dans la partie **Axes d'analyse** nous nous sommes orientés 
                sur l’analyse de deux problématiques prédictives à savoir : 
                        \n- être en capacité d'estimer la consommation d'électricité économisée sur 2020, en simulant une consommation "normale" via le Machine Learning et en la comparant avec la consommation réelle
                        \n- identifier une corrélation entre la température et la consommation d'électricité.
             
    """)
    
    
if page == 'Jeux de données':
    st.title("Jeux de données")
    
    # choix de l'affichage
    donnee= st.radio(label=" ", 
                     options=['Dataset principal',
                              'Dataset secondaire',
                              'Dataset utilisé pour nos prédictions'])
    
    if donnee == 'Dataset principal':
        st.header(""" Dataset principal """)
    
        st.subheader(""" 1.Source """)
        st.markdown("""  Le jeu de données principal provient de [l’ODRE (Open Data Réseaux Energies) eco2mix-national-tr](https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-national-tr/information/?disjunctive.nature)
                    """)
                    
        st.subheader(""" 2.Contenu et volumétrie """)
        st.markdown (""" Le jeu de données contient l’ensemble des informations de **consommation et de production d'énergie** par filière énergétique, par pas de 30 minutes, depuis le 01/01/2013 jusqu’au 28/02/2021.
                       Le jeu de données ODRE représente 1 717 056 lignes sur 65 colonnes, chaque ligne équivalant à un relevé de consommation / production d'énergie  par région toutes les 30 minutes.
                       """)
                       
        st.subheader(""" 3.Description des variables du dataset Energie""")
        st.markdown("""
                        - Région (code insee, libellé),
                        - Date, heure, date-heure, 
                        - Consommation,
                        - Production, selon les différentes filières (Thermique, Nucléaire, Eolien, Solaire, Hydraulique, Bioénergie),
                        - Echanges physiques entre régions ou pays étrangers,
                        - Flux physique d'une région/pays vers une autre région/pays,
                        - TCO Thermique, Nucléaire, Eolien, Solaire, Hydraulique, Bioénergie,
                        - TCH Thermique, Nucléaire, Eolien, Solaire, Hydraulique, Bioénergie.
                    """)
                    
        st.markdown ("""  La **variable cible** est “Total Conso”, une variable créée et constituée de la somme des valeurs des colonnes Consommation MW et Pompage MW (Puissance consommée par les pompes dans les Stations de Transfert d'Energie par Pompage (STEP)) du jeu de données de l’ODRE.
                     """)
        st.markdown(""" _Remarques :_ 
                    \n- Le **pompage** et les STEP turbinage : Les "STEP" (stations de transfert d’énergie par pompage) sont des installations hydroélectriques qui puisent aux heures creuses de l'eau dans un bassin inférieur afin de remplir une retenue en amont (lac d'altitude). L'eau est ensuite turbinée aux heures pleines pour produire de l’électricité.
La rubrique STEP turbinage correspond à la puissance produite par l’eau turbinée.
La rubrique pompage représente l’ensemble de la puissance consommée par les STEP.

\nCela signifie que lorsque la variable pompage est négative, la station produit de l'électricité       
                    \n- **TCO** : Taux de COuverture de production. Exemple : TCO thermique se définit comme la part des besoins en énergie couverts par l’énergie thermique. Formule de calcul : Production thermique/ Besoins.

Les données n'existent que pour l'année 2020 (d'où le taux de NA élevé)

                    \n- **TCH** : Le taux de charge d’une unité de production est le ratio entre l’énergie qu’elle produit sur une période donnée et l’énergie qu’elle aurait produite durant cette période si elle avait constamment fonctionné à puissance nominale. Il fournit une indication importante pour calculer la rentabilité d’une installation.

\nLorsque des valeurs dépassent les 100% c'est que la filière a produit au-delà de ce qui était prévu. Ce n'est pas un _outlier_ mais une sur-performance de la filière.

\nLes données n'existent que pour l'année 2020 (d'où le taux de NA élevé)                         
                                              """)    
        st.subheader(""" 4.Sélection des données """)  
        st.markdown (""" Dans le cadre de l'analyse du jeu de données, nous avons fortement allégé notre dataset en suppression les colonnes non pertinentes au regard de nos problématiques, à savoir celles en lien avec le volume d'énergie échangé entre les régions,
                        les taux de consommation et de charges.
                    \n- Les données couvrent la période du 01/01/2013 au 28/02/2021. Nous selectionnons les données de 01/01/2013 au 31/12/2020 pour bénéficier d'un jeu de données sur des années complètes. 
                    \n- Les données de la Corse étant incomplètes, elles sont écartées de notre étude.
                     """)
                        
       
        # pour des raisons de temps d'affichage et de traitement, un fichier de 30 lignes a été préenregistré
        #df20=pd.read_csv('eco2mix-regional-cons-def_30_lignes.csv',sep=(','))
        st.markdown (""" **Voici un apercu du Dataset Energie** """)
        df20= Energ_Get_Data.lire_df_30_lignes()
        st.write(df20)  
    
    if donnee == 'Dataset secondaire':
        st.header(""" Dataset secondaire """)
    
        st.subheader(""" 1.Source """)
        st.markdown(""" Le jeu de données des températures quotidiennes régionales, provient du site [datagouv.fr temperature-quotidienne-regionale-depuis-janvier-2016] (https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-regionale-depuis-janvier-2016/)
                    """)
                    
        st.subheader(""" 2.Contenu et volumétrie """)
        #                     (https://opendata.reseaux-energies.fr/explore/dataset/temperature-quotidienne-regionale/information/?disjunctive.region)""")
        st.markdown (""" Le jeu de données contient les températures minimales, maximales et moyennes journalières par région, du 01/01/2016 au 29/04/2021.
                      """)

        st.subheader(""" 3.Description des variables du dataset Météo""")  
        st.markdown("""
                        - Région (code insee, libellé),
                        - Date, 
                        - Température minimale, moyenne, maximale.
                        """)
                   
        st.subheader(""" 4.Selection des données """)  
        st.markdown ("""  Les données couvrent la période du 01/01/2016 au 29/04/2021. Nous selectionnons les données de 01/01/2016 au 31/12/2020 pour bénéficier d'un jeu de données sur des années complètes. """)
                    
        df_meteo= Energ_Get_Data.lire_df_meteo_50_lignes()
        st.markdown ("""  **Voici un apercu du Dataset Météo**  """)
    
        st.write(df_meteo)

    if donnee == 'Dataset utilisé pour nos prédictions':
        
        st.header("""  Dataset utilisé pour nos prédictions  """)

        st.markdown (""" Nous nous sommes intéressés au lien entre les 2 datasets : 
                     \n- Energie : la consommation / production d'énergie,
                     \n- Température
                 """)  
                   

        st.subheader(""" 1.Traitement des 2 datasets sources""")
        st.markdown(""" \t**1.1 Traitement du dataset Energie**  """)
        
        st.markdown ("""                    
                       Nous avons :
                           \n- conservé les colonnes région, date et total_conso
                           \n- supprimé les lignes ne contenant pas de valeur (NaN) sur les variables retenues
                           \n- supprimé les données liées à la région Corse et celles au-delà du 31/12/2020
                           \n- agrégé nos données pour obtenir une périodicité journalière 
                           
                     """)
        st.markdown(""" _Remarques :_
                    \n Lors de nos premiers essais avec des modèles de prédictions, nous avons obtenu un résultat proche de 99%. 
\n Nous pensons être dans un cas d’**overfitting**, puisque les données de production semblent biaiser le score du modèle.
\n En effet, au regard du nécessaire équilibre entre production et consommation d’électricité, les productions affichées dans le jeu de données sont en réalité déjà adaptées à la consommation. 
\n Cela explique pourquoi le modèle a semblé si performant. 
\n Ainsi, nous avons supprimé l’ensemble des colonnes de production dans nos modèles de prédictions pour ne pas fausser nos modèles de prédiction de la consommation.
\n Nous avons également transformé toutes nos données en données numériques afin de pouvoir optimiser nos modèles de prédiction.
                """)

        

        st.markdown(""" \t**1.2 Traitement du dataset Température**  """)

        st.markdown ("""                    
                       Le jeu de données Température étant plus restreint, nous avons seulement supprimé les données liées à la région Corse et celles au-delà du 31/12/2020 pour permettre la fusion des 2 datasets.
                           
                     """)
        st.subheader(""" 2.Fusion des 2 datasets sources""")

        st.markdown ("""  Une fois la périodicité et la période des données uniformisées, nous avons ensuite fusionné les 2 datasets en regroupant les données par région et date.
                     """)

        st.markdown (""" 
                Nous avons ainsi pu réduire et conserver les données nécessaires aux prédictions, à savoir :
                - Région,                
                - Date	
                - Total_conso	
                - tmin	: température minimale
                - tmoy	: température moyenne
                - tmax	: température maximale	

                """)

if page == "Axes d\'analyse":
    
    st.title("Les axes d'analyses")
    
    st.header(""" 1. Problématique""")
                                  
    st.markdown ("""
                 **Pourquoi est-il necessaire de prédire la consommation énergétique ?**
                 
                      \n**Pour organiser la production afin de : **
                      \n- Prévenir tout risque de blackout en France au cours des années à venir
                      \n- Éviter de devoir générer trop de transferts d’énergie entre les différentes régions
                      
                  \n**Pour maintenir et entretenir notre indépendance énergétique, il faut**
                      \n- Planifier de manière optimale les opérations de maintenance sur les différents sites de production d’énergie
                      \n- Anticiper les besoins de création de nouveaux centres de production
                 """)
    st.header ("""2. Analyse """)
                 
    st.markdown ("""
                 Nous avons orienté notre analyse suivant ces axes dimensionnels :
                 \n- **Type de production** : Quelle est la production d'énergie par filière de production ?
                 \n- **Aspect autosuffisance régionale** : Les régions produisent-elles assez d'énergie pour leurs propres besoins ?
                 \n- **Dimension temporelle** : Les énergies renouvelables (hydraulique, éolien, solaire, bioénergie) sont-elles dépendantes des saisons ? La consommation d'énergie dépend-elle de certaines plages horaires ?
                 \n- **Dépendance de la météo** : Dans quelle mesure la consommation dépend-elle de la météo ?
                 """)
            
    # choix de l'affichage
    st.header ("""  3. Quel axe souhaitez-vous analyser ? 
                 """)
    choix= st.radio(label="", 
                     options=['Type de production',
                              'Aspect autosuffisance régionale',
                              'Dimension temporelle',
                              'Dépendance de la météo'
                              ])
    
    if choix == 'Type de production':
                      
        #---------Analyse par filière de production de Janvier 2013 à 2021     
        st.markdown("\n\n")
        st.header('Quelle est la production d\'énergie par filière ?') 
        
        st.subheader('\n1. Répartition de la production par filière de 2013 à 2020 dans toutes les régions');
        
        #lecture du fichier
        df = Energ_Get_Data.lire_df_pie_2013_2020()
         
        Energ_DataViz.graph_filiere_2013_2020(False,"", df)

        
        st.markdown ("""
                     La production d'énergie **Nucléaire** est la plus importante de 2013 à 2020. Elle représente **72.78%** de la consommation totale. 
                     Le **thermique** représente **7.38%**. Les **énergies renouvelables** (solaire, éolien, hydraulique, bioénergie) 
                     totalisent **19,85%**, avec une prépondérance pour l'hydraulique à **11,91%**. 
                     La consommation totale ne comprend pas les échanges de flux entre les régions et les pays étrangers, faute de données).
                     """)
  
        st.markdown ("""
                     la production d'énergie Nucléaire tend à baisser, avec une forte chute en 2020. Est-ce dû aux conséquences du COVID-19 ?
                     """)
    
        st.subheader('2. Pour une vision par région')

           
        liste_region = ["Auvergne-Rhône-Alpes","Ile-de-France","Bourgogne-Franche-Comté","Bretagne","Centre-Val de Loire","Grand Est","Hauts-de-France",
                        "Normandie","Nouvelle-Aquitaine","Occitanie","Pays de la Loire","Provence-Alpes-Côte d'Azur"]
        liste_code_insee = [84,11,27,53,24,44,32,28,75,76,52,93]
        
        region = st.selectbox("",options=liste_region)
        
        for index,elt in enumerate (liste_region):
            if region == elt:
                code_insee = liste_code_insee [index]

        Energ_DataViz.graph_filiere_2013_2020(True,code_insee, df)
        
        st.markdown (""" \n\n""")
                             
        scale = st.checkbox(label='\n\n Voir les régions au top des énergies renouvelables :', value=True)
        if scale:
            st.markdown ("""  
                         -	Bretagne : 82% renouvelable, 18% thermique
                         -	Bourgogne-Franche-comté : 76% renouvelable, 24% thermique
                         -	Provence-Alpes-Côte d’azur : 65% renouvelables, 35% Thermique
                         """)

             

    if choix == 'Aspect autosuffisance régionale':
     
        # --------Analyse du TCO (taux de couverture) par region            
        st.markdown("\n")
        st.header('Les régions produisent-elles assez d\'énergie pour leurs propres besoins ?') 
        st.subheader('1. Production par filière par région')  
        
        
        Energ_DataViz.graph_TCO_region()

         
        st.markdown ("""
                     La première chose que l'on remarque est que toutes les régions n\'ont pas de filière nucléaire (Bourgogne Franche Comte, Bretagne, Pays de la Loire, Provence Alpes Côte d'azur, Ile-de-France).
                      En effet la France compte 18 centrales sur son territoire:
        \n- 5 en Centre Val de Loire
        \n- 4 en Auvergne Rhône Alpes
        \n- 3 en Normandie
        \n- 3 dans le Grand Est
        \n- 1 dans les Hauts de France
        \n- 1 en Nouvelle Aquitaine
        \n- 1 en Occitanie
                 \nLe **TCO** représente le **Taux de couverture**, c'est à dire le rapport entre l'**énergie produite** par une filière et les **besoins en énergie**.
    On voit que c\'est toujours la **filière nucléaire qui a le TCO le plus élevé**, c'est la première source de production d\'énergie en France et dans toutes les régions où il y a des centrales. 
    
    \nDans les régions **Auvergne-Rhône Alpes, Centre Val de Loire, Grand Est et Normandie**, rien qu'avec la filière nucléaire, le TCO est au-dessus des 100%. Ces régions produisent donc suffisamment d'énergie pour répondre à leurs besoins. Elles sont **exportatrices d\'énergie**. Notamment la région Centre Val de Loire qui est le coeur de production énergétique.
    Toutes les autres régions sont obligées d\'importer de l'énergie car l'ensemble de leurs filières ne couvrent pas leurs besoins. On observe également des régions produisant très peu d\'énergie. et dans ces régions l\'Ile de france. 
         """)
    #---------Analyse de la consommation / Production par région en 2019 et 2020  

        st.markdown("\n\n")
        st.subheader('2. Rapport entre la production et la consommation, en 2020') 
    
            
        Energ_DataViz.graph_Conso_prod_2020()

        st.markdown ("""
                       On constate une production supérieure à la consommation dans les régions Auvergne-Rhône-Alpes, Centre Val de Loire, Grand Est, Normandie 
                       régions qui comptent le plus de centrales nucléaires.
                       
                     """) 
        # *****  Analyse possible : Calcul de l'écart entre prod / conso (en %) par region  : Ile de France, Pays de la loire, Bourgogne Franche comté, Bretagne
        # *****  Pour ces regions, analyse par filière : En Bretagne part de l 'éolien par rapport au solaire, ....
        

        
 # ------   Analyse des saisons et des heures       
    if choix == 'Dimension temporelle':
        st.markdown("\n\n")
        st.subheader("1. La production des énergies renouvelables est-elle soumise à saisonnalité (en 2018 et  2019) ?")
        
        Energ_DataViz.graph_energ_renouv()
   
        
        
        st.markdown("En effet, cette production est globalement soumise à la saisonnalité.")
        st.markdown("""
                    Notamment l'énergie **hydraulique**. En effet, la production d’électricité hydraulique dépend fortement des précipitations 
                    (décrue significative dès juin jusque fin août). Le barrage retient un lac artificiel de grande superficie qui se remplit à la fonte des neiges ou à la saison des pluies. Ce qui fait de l'hydroélectricité la seule énergie modulable. En effet, cette ressource peut être rapidement mobilisable pour répondre aux appels de forte consommation (production en nette hausse dès septembre).
    """)
        st.markdown("""
                    L'énergie produite par les **éoliens** baisse de Mars à Aout, au printemps et en été. Y aurait-il moins de vent à l'approche de l'été ? En revanche, le **solaire** suit une courbe inverse, avec une augmentation en été. La **bioénergie** est pratiquement constante.""" )
       


        st.subheader("2. Quelle est la consommation par heure et par région ?")

        Energ_DataViz.graph_energ_conso_heure_region()
        
        st.markdown("""
                     On peut observer que la consommation suit la même courbe quelle que soit la région. A partir de 6h la consommation augmente pour atteindre un pic vers 12h. Puis la consommation baisse jusqu'à 17h. Elle augmente de nouveau entre 17h et 19h (fort pic à 19h).
                     """)
                     
    if choix == "Dépendance de la météo":
        st.subheader("1. Quelle est la consommation suivant la température, par région ?")
        
        Energ_DataViz.graph_conso_temperature()             
                     
        st.markdown("""
                     On peut observer que la consommation baisse lorsque la température augmente jusqu'à 18/20 degrés. 
                     Puis elle amorce une légère augmentation au dessus de 23 degrés, dû probablement à la climatisation.
                    """)
        
        st.markdown (""" \n\n""")
                             
        scale = st.checkbox(label='\n\n Voir le graphique des régions')
        if scale:             
            Energ_DataViz.graph_conso_temperature_region()   
        

if page == 'Conclusion':
    st.title("CONCLUSION")
    st.subheader("""Critiques""")
    st.markdown ("""
                 - Les modèles dépendent des données météorologiques donc dépendant de leur échelle de temps (on ne peut pas prédire l’année 2050). Dans notre contexte de changement climatique, les données peuvent évoluer rapidement en moins de 2 ans.
                 - Les modèles s’appuient en grande partie sur les données historiques et pourraient prendre en compte d’autres facteurs pour être plus performants dans le cas d’une année comme 2020.

                 """)
                 
    st.subheader("""Perspectives, liés à la Météo et autres""")
    st.markdown ("""
                 - Ajouter les données d’échanges énergétiques entre régions et avec autres pays
                 - Utiliser des données météorologiques plus complètes et plus précises (à l’heure plutôt qu’à la journée). Nous n’avons pas pu affiner jusqu’au cycle jour/nuit par exemple
                 - Identifier le jour de semaine pour mener une analyse sur l’impact du week-end, des jours fériés, sur la consommation. Y a-t-il un jour de la semaine sur lequel on constate systématiquement une consommation plus importante ?
                 - Quelle serait la consommation si la température est de 5 degré, 20, ou 30 ? nuancés par les jours de semaine type, par région. Pour une température donnée, la consommation est-elle constante ?  oui/non, si non pourquoi ?
                 - Analyser les régions dans lequelles les énergies renouvelables sont les plus faibles, et chercher des pistes pour augmenter la production. Quel est le nombre de journées ensoleillées ? quelle est la vitesse du vent ? quelle est la pluviométrie ?
                 - Analyser plus en détail les impacts des événements exceptionnels (grève,COVID-19)
                 - Récupérer des données plus détaillées sur la consommation d'énergies par type de consommateurs (Entreprises, Grande industrie, PME-PMI, Professionnels, Résidentiels) et les analyser. 
                 - Pour chaque énergie renouvelable, étudier leur implantation géographique. Est ce possible de les développer dans telle et telle région ? oui/non, pourquoi ?
                 - Récupérer les consommations détaillées au niveau local (ville par exemple), pour analyser les pôles consommateurs (exemple : éclairage urbain, éclairage des magasins la nuit, grandes entreprises, ...)
                 """)

#Analyse du meilleur modèle
if page =='Analyse des modèles':
    
    st.title("Analyse des modèles")
    
    st.subheader("Performance des modèles")
    
    # affichage d'une photo
    tableau = plt.imread('tableau.jpg') 
    st.image(tableau)
    
    
    st.markdown ("""
                  Nous avons ci-dessus un récapitulatif des performances des modèles que nous avons testés.
Le MSE (Mean Squared Error ou Erreur Quadratique Moyenne) est une mesure caractérisant la précision de notre modèle. L’objectif est d’avoir un MSE le plus proche de 0.

La première chose que nous pouvons observer est que tous nos modèles ont des performances très similaires.
Notre hypothèse est que cela vient du nombre de données différentes que nous avons fourni aux modèles (5). Plus un large choix de types de données est offert au modèle, plus les modèles auront des résultats différents.

Ensuite, nous pouvons voir pour la prédiction de l’année 2019 par exemple, que le score obtenu au test est bon (0.9, le maximum étant 1).
Cependant, le MSE de notre prédiction est supérieur au MSE obtenu lorsqu’on fait la moyenne des consommations sur les 3 années précédentes.

Cela revient à dire que notre consommation d’une année à l’autre est presque similaire et que notre modèle est perturbé par les autres informations que nous lui fournissons.

                 """)
   

    
    
    st.subheader("Analyse du meilleur modèle")
    
    
    st.markdown ("""
                 Tous nos modèles ont des résultats presques identiques. Le modèle LassoCV se démarque très légèrement sur 2019, notamment en terme de MSE.
                 
Nous avons donc effectué une analyse en composantes principales (PCA) du modèle LassoCV.
Ce modèle permet de visualiser l’influence de chaque variable sur la variable cible.

                 """)
    
    # affichage d'une photo
    graphe = plt.imread('inter.jpg') 
    st.image(graphe)
   
    st.markdown ("""
                 Sur l’image ci-dessus nous pouvons voir que les 3 variables ayant le plus d’influence sont les températures : min, max et moyenne. Leur corrélation avec la variable cible est négative. 
                 Cela signifie que lorsque la température augmente, la consommation électrique va tendre à diminuer et inversement.
                 
                 Concernant l’influence des régions elle est moins importante et varie en fonction de la région.
                 Par exemple, si le logement est situé en région Grand Est, sa localisation va automatiquement augmenter la prédiction de consommation.
                 A l’inverse, si le logement est situé en Provence-Alpes Côtes d’Azur ou en Nouvelle-Aquitaine, la prédiction va diminuer la prédiction de consommation.

                 """)
 
   

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
    col1, col2, col3, col4, col5 = st.beta_columns(5)
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
            col1, col2 = st.beta_columns(2)
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
            col1, col2 = st.beta_columns(2)
            
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
        col1, col2 = st.beta_columns(2)
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
            col1, col2 = st.beta_columns(2)
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
            col1, col2 = st.beta_columns(2)
            
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
            
                        
        


