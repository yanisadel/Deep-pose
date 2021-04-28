import pandas as pd
from utils import *
import random as rd
from sklearn.model_selection import train_test_split
import data
from pandas import DataFrame
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from detection_position import *
from data import *


n_neighbors=3

def knn_entraine(path='data_train/signes.csv',type_prediction='signe',test_size_knn = 0.1):
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
    """Prend en entrée le chemin d'accès à un excel scinde les données en une partie d'entrainement et une partie de test
    Parametres:
    path = chemin excel des données
    type= string de la predictions que l'on veut signe ou position
    """ 
    # Entrainement du knn
    knn,x_test,y_test=knn_entraine(path,type_prediction)
    # prediction
    if type_prediction=='signe':
        x_test=normalize_data(x_test)
    predictions=knn.predict(x_test)
    return predictions,y_test

def predictions_excel_proba(path,type_prediction='signe',test_size_knn = 0.1):
    """Prend en entrée le chemin d'accès à un excel scinde les données en une partie d'entrainement et une partie de test
    Parametres:
    path = chemin excel des données
    type= string de la predictions que l'on veut signe ou position
    """ 
    # Entrainement du knn
    knn,x_test,y_test=knn_entraine(path,type_prediction)
    # prediction
    if type_prediction=='signe':
        x_test=normalize_data(x_test)
    predictions=knn.predict(x_test)
    probability_class= knn.predict_proba(x_test)
    return predictions,probability_class

def pourcentage_reussite(predi,y_test):
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
    """Prediction du signe depuis une image
    Parametres: 
    knn = le réseau entraine
    image = l'image traités
    type_prediction = le type de prédiction : signe ou position"""
    if type_prediction=='signe':
        l = points_image(image, min_detection_confidence=min_detection_confidence)
    else:
        l = vector_to_face_dlib(image, min_detection_confidence=min_detection_confidence)
    if type(l)!=type(None):
        if type_prediction=='signe':
            l=normalize_list_points(l)
        prediction = knn.predict([l])
        return prediction
def prediction_image_from_path(knn, path,type_prediction):
    """Prediction du signe depuis le chemin d'une image"""
    image = cv2.imread(path)
    return prediction_image(knn, image,type_prediction)

def prediction_image_proba(knn, image,type_prediction, min_detection_confidence=0.7):
    """Retourne la prédiction et la liste des probabilité de chaque classe"
    Parametres: 
    knn = le réseau entraine
    image = l'image traités
    type_prediction = le type de prédiction : signe ou position"""
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
    """Prediction du signe depuis le chemin d'une image"""
    image = cv2.imread(path)
    return prediction_image_proba(knn, image,type_prediction)


knn_signes,_,_ = knn_entraine('data_train/signes.csv','signe') # retourne le knn pour les signes entrainé
knn_position,_,_=knn_entraine('data_train/face.csv','position')
knn_position_dlib,_,_=knn_entraine('data_train/dlib.csv','position')

if __name__ == '__main__':

    predictions_1,y_test_1=predictions_excel('data_train/signes.csv')
    predictions_2,y_test_2=predictions_excel('data_train/face.csv','position')
    predictions_3,y_test_3=predictions_excel('data_train/dlib.csv','position')

    pourcentage_reussite_1,erreur_1 = pourcentage_reussite(predictions_1,y_test_1)
    pourcentage_reussite_2,erreur_2 = pourcentage_reussite(predictions_2,y_test_2)
    pourcentage_reussite_3,erreur_3 = pourcentage_reussite(predictions_3,y_test_3)

    print(pourcentage_reussite_1,erreur_1)
    print(pourcentage_reussite_2,erreur_2)
    print(pourcentage_reussite_3,erreur_3)
    print(prediction_image_proba_from_path(knn_signes, 'data_test/LPC/WIN_20210415_08_45_49_Pro.jpg','signe'))
    #print(predictions_proba('data_train/dlib.csv','position'))
    #s='dataset/LPC/'
    #for path in listdir(s):
        #print(prediction_signe_image_from_path(knn_signes, s + path))
    




