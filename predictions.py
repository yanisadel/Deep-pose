import pandas as pd
from utils import *
import random as rd
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from detection_position import *
from data import *

n_neighbors=5

def knn_entraine(path='data_train/signes.csv',type_prediction='signe',n_neighbors=n_neighbors):
    """
    knn_entraine prend en entrée le chemin d'un excel et retourne un knn entraine

    Arguments
    ---------
   
    path: str
        chemin de l'excel contenant les données du data

    type_prediction : str
        type que l'on veut analyser soit signe ou position 
        cela permet de normaliser les données si on veut signe

	n_neighbors= nombre de voisin du knn

    Returns
    -------
   knn : Classierfier
   reseau knn entrainé
      
    """
    n=rd.randint(1,40)
    x_train, y_train = read_csv(path)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    if type_prediction=='signe':
        x_train=normalize_data(x_train)
    knn.fit(x_train, y_train)
    return knn


def knn_entraine_excel(path='data_train/signes.csv',type_prediction='signe',test_size_knn = 0.1,n_neighbors=n_neighbors):
    """
    knn_entraine prend en entrée le chemin d'un excel split les données en des données test et des donnees d'entrainement et retourne un knn entraine et les données test, les vecteurs x_test et les labels y_test

    Arguments
    ---------
   
    path: str
        chemin de l'excel contenant les données du data

    type_prediction : str
        type que l'on veut analyser soit signe ou position 
        cela permet de normaliser les données si on veut signe
    
    test_size_knn = float
        pourcentage de données test (nombre entre 0 et 1)

	n_neighbors= nombre de voisin du knn

    Returns
    -------
   knn : Classierfier
   reseau knn entrainé

   x_test : dataframe
   df qui contient des vecteurs tests pour vérifier la précision du knn

   y_test : label
   contient les classes des vecteurs tests
      
    """   
    n=rd.randint(1,40)
    x, y = read_csv(path)

    # Séparation des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=n)
    # Entrainement du knn
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    if type_prediction=='signe':
        x_train=normalize_data(x_train)
    knn.fit(x_train, y_train)
    return knn,x_test,y_test


def predictions_excel(path,type_prediction='signe',test_size_knn = 0.1):
    """
    predictions_excel prend en entrée le chemin d'un excel utilise knn_entraine_excel et 
    teste les données test pour renvoyer les predictions de classes faites par le knn 
    et les classes qui correspondent aux réelles classes des vecteurs

    Arguments
    ---------
   
    path: str
        chemin de l'excel contenant les données du data

    type_prediction : str
        type que l'on veut analyser soit signe ou position 
        cela permet de normaliser les données si on veut signe
    
    test_size_knn = float
        pourcentage de données test (nombre entre 0 et 1)

    Returns
    -------
    predictions : 
    contient les classes predites par le knn

   y_test : label
   contient les classes des vecteurs tests
      
    """ 
    # Entrainement du knn
    knn,x_test,y_test=knn_entraine_excel(path,type_prediction)
    # prediction
    if type_prediction=='signe':
        x_test=normalize_data(x_test)
    predictions=knn.predict(x_test)
    return predictions,y_test

def predictions_excel_proba(path,type_prediction='signe',test_size_knn = 0.1):
    """
    predictions_excel_proba prend en entrée le chemin d'un excel utilise knn_entraine_excel et 
    teste les données test pour renvoyer les predictions de classes faites par le knn 
    et le pourcentage des classes dans les n voisins
    Arguments
    ---------
   
    path: str
        chemin de l'excel contenant les données du data

    type_prediction : str
        type que l'on veut analyser soit signe ou position 
        cela permet de normaliser les données si on veut signe
    
    test_size_knn = float
        pourcentage de données test (nombre entre 0 et 1)

    Returns
    -------
    predictions : 
    contient les classes predites par le knn

   probability_class : list
   renvoie une list de liste dont chaque liste contient en i position le pourcentage de la classe i dans les n voisins
      
    """ 
    # Entrainement du knn
    knn,x_test,y_test=knn_entraine_excel(path,type_prediction)
    # prediction
    if type_prediction=='signe':
        x_test=normalize_data(x_test)
    predictions=knn.predict(x_test)
    probability_class= knn.predict_proba(x_test)
    return predictions,probability_class

def pourcentage_reussite(predi,y_test):
    """
   pourcentage réussite prend en entrée les predictions et les réelles classes des vecteurs
   elle renvoie le pourcentage de réussite et la liste des couples erreurs

    Arguments
    ---------
   
    predi  :   tuple 
    contient les precitions faites par knn

    y_test : 
    contient les vraies classes

    Returns
    -------

    pourcentage de reussite : float
    pourcentage de reussite du knn

    res : list
    liste de tuple des couples erreurs/classe réelles
      
    """ 
    n = len(predi)
    c=0
    yt2 =list(y_test)
    res=[]
    for k in range(n):
        if predi[k]==yt2[k]:
            c+=1
        else:
            res.append([predi[k],yt2[k]])
    return (c/n*100,res)

def prediction_image(knn, image,type_prediction, min_detection_confidence=0.7):
    """
    prediction_image renvoie la prediction de la classe de l'image en fontion du type de prediction que l'on veut

    Arguments
    ---------
   knn : Classifier
   reseau knn entrainé

   image : image cv2
   l'image sur laquelle on veut faire la prediction

   type_de_prediction : str
   type de prédiction que l'on veut faire 'signe' ou 'position'

    min_detection_confidence : float
    degree de confiance dans l'analyse

    Returns
    -------
    predictions : list
    prediction de la classe de l'image  
    """ 
    if type_prediction=='signe':
        l = points_image(image, min_detection_confidence=min_detection_confidence)
    else:
        l = vector_to_face_dlib(image, min_detection_confidence=min_detection_confidence)
    if type(l)!=type(None):
        if type_prediction=='signe':
            l=normalize_list_points(l)
        prediction = knn.predict([l])
        return prediction

def prediction_image_from_path(knn, path,type_prediction,min_detection_confidence=0.7):
    """
    prediction_image renvoie la prediction de la classe de l'image en fontion du type de prediction que l'on veut

    Arguments
    ---------
   knn : Classifier
   reseau knn entrainé

   path : str
   chemin de l'image sur laquelle on veut faire la prediction

   type_de_prediction : str
   type de prédiction que l'on veut faire 'signe' ou 'position'

    min_detection_confidence : float
    degree de confiance dans l'analyse

    Returns
    -------
    predictions : list
    prediction de la classe de l'image  
    """ 
    image = cv2.imread(path)
    return prediction_image(knn, image,type_prediction)

def prediction_image_proba(knn, image,type_prediction, min_detection_confidence=0.7):
    """
    prediction_image renvoie la prediction de la classe de l'image en fontion du type de prediction que l'on veut

    Arguments
    ---------
   knn : Classifier
   reseau knn entrainé

   image : image cv2
   l'image sur laquelle on veut faire la prediction

   type_de_prediction : str
   type de prédiction que l'on veut faire 'signe' ou 'position'

    min_detection_confidence : float
    degree de confiance dans l'analyse

    Returns
    -------
    predictions : list
    prediction de la classe de l'image  

    probability_class : list
    renvoie une liste dont la ieme position contient le pourcentage de la classe i dans les n voisins
      
    """ 
    if type_prediction=='signe':
        l = points_image(image, min_detection_confidence=min_detection_confidence)
    else:
        l = vector_to_face_dlib(image, min_detection_confidence=min_detection_confidence)
    if type(l)!=type(None):
        if type_prediction=='signe':
            l=normalize_list_points(l)
        prediction = knn.predict([l])
        probability_class= knn.predict_proba([l])
        return (prediction[0],list(probability_class[0]))

def prediction_image_proba_from_path(knn, path,type_prediction):
    """
    prediction_image renvoie la prediction de la classe de l'image en fontion du type de prediction que l'on veut

    Arguments
    ---------
   knn : Classifier
   reseau knn entrainé

   path : str
   chemin de l'image sur laquelle on veut faire la prediction

   type_de_prediction : str
   type de prédiction que l'on veut faire 'signe' ou 'position'

    min_detection_confidence : float
    degree de confiance dans l'analyse

    Returns
    -------
    predictions : list
    prediction de la classe de l'image  

    probability_class : list
    renvoie une liste dont la ieme position contient le pourcentage de la classe i dans les n voisins
      
    """ 
    image = cv2.imread(path)
    return prediction_image_proba(knn, image,type_prediction)


knn_signes,_,_ = knn_entraine_excel('data_train/signes.csv','signe') # retourne le knn pour les signes entrainé
knn_position,_,_=knn_entraine_excel('data_train/face.csv','position')
knn_position_dlib,_,_=knn_entraine_excel('data_train/dlib.csv','position')

if __name__ == '__main__':



    a,b,c,n=0,0,0,100
    for i in range(1,n+1):
        predictions_1,y_test_1=predictions_excel('data_train/signes.csv')
        predictions_2,y_test_2=predictions_excel('data_train/face.csv','position')
        predictions_3,y_test_3=predictions_excel('data_train/dlib.csv','position')
        pourcentage_reussite_1,erreur_1 = pourcentage_reussite(predictions_1,y_test_1)
        pourcentage_reussite_2,erreur_2 = pourcentage_reussite(predictions_2,y_test_2)
        pourcentage_reussite_3,erreur_3 = pourcentage_reussite(predictions_3,y_test_3)
        a=c+pourcentage_reussite_1
        b=c+pourcentage_reussite_2
        c=c+pourcentage_reussite_3
    print(a/n,b/n,c/n)
    #print(pourcentage_reussite_1,erreur_1)
    #print(pourcentage_reussite_2,erreur_2)
    #print(pourcentage_reussite_3,erreur_3)
    #print(prediction_image_proba_from_path(knn_signes, 'data_test/LPC/WIN_20210415_08_45_49_Pro.jpg','signe'))
    #print(predictions_proba('data_train/dlib.csv','position'))
    #s='data_test/LPC/'
    #for path in listdir(s):
        #print(prediction_image_from_path(knn_signes, s + path,'signe'))
    




