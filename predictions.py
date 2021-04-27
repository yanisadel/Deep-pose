import pandas as pd
from utils import *
import random as rd
from sklearn.model_selection import train_test_split
import data
import detection_position as dp
from pandas import DataFrame
from os import listdir
from sklearn.neighbors import KNeighborsClassifier



def knn_entraine(path='data_train/signes.csv',test_size_knn = 0.1):
    n=rd.randint(1,40)
    x, y = read_csv(path)

    # Séparation des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=n)

    # Entrainement du knn
    knn = initialize_knn(x_train, y_train)
    train_knn(knn,x_train,y_train)
    return knn

def knn_entraine(path='data_train/signes.csv',type,test_size_knn = 0.1):
    n=rd.randint(1,40)
    x, y = read_csv(path)

    # Séparation des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=n)

    # Entrainement du knn
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    if type=='signe':
        x_train=normalize_data(x_train)
    knn.fit(x_train, y_train)
    return knn,x_test,y_test


def predictions(path,type='signe',test_size_knn = 0.1):
    """Prend en entrée le chemin d'accès à un excel scinde les données en une partie d'entrainement et une partie de test
    Parametres:
    path = chemin excel des données
    type= string de la predictions que l'on veut signe ou position
    """ 
    # Entrainement du knn
    knn,x_test,y_test=knn_entraine(path,type)
    # prediction
    if type=='signe':
        x_test=normalize_data(x_test)
    predictions=knn.predict(x_test)
    return predictions,y_test

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


""" UTILISATION DU CODE SUR UNE IMAGE AVEC DATASET DEJA FAIT """
def test_une_image(path):
    list_coord=dp.vector_to_face_from_path(path, min_detection_confidence=0.7, display=True)
    if type(list_coord)!=type(None):
        columns=data.labels_csv_face()[1:]
        df_coord = DataFrame([list_coord],columns=columns)
        return(predictions_knn_niveau(knn_3, df_coord))


if __name__ == '__main__':
    predictions_1,y_test_1=predictions('data_train/signes.csv')
    predictions_2,y_test_2=predictions('data_train/face.csv','position')
    pourcentage_reussite_1,erreur_1 = pourcentage_reussite(predictions_1,y_test_1)
    pourcentage_reussite_2,erreur_2 = pourcentage_reussite(predictions_2,y_test_2)
    print(pourcentage_reussite_1,erreur_1)
    print(pourcentage_reussite_2,erreur_2)
    
    #s='data_test/LPC/'
    #for path in listdir(s):
        #print(test_une_image(s + path))
    




