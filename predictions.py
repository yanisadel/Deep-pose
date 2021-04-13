import pandas as pd
from utils import *
from niveaux import *
from signes import *
import random as rd
from sklearn.model_selection import train_test_split
import data
import detection_position as dp
from pandas import DataFrame




def retourne_knn_entraine(path='Data/signes.csv'):
    n=rd.randint(1,40)
    x, y = read_csv(path)

    # Séparation des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=n)

    # Entrainement du knn
    knn = initialize_knn(x_train, y_train)
    train_knn(knn,x_train,y_train)
    return knn


"""PARAMETRES"""
test_size_knn = 0.1 # Proportion pour les données de train et de test
test_size_tree = 0.1


"""PREDICTION DU SIGNE DE LA MAIN"""
# Chargement des données
path_signes = 'Data/signes.csv'
x, y = read_csv(path_signes)

# Séparation des données
n=rd.randint(1,40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=n)

# Entrainement du knn
knn_1 = initialize_knn(x_train, y_train)
train_knn(knn_1,x_train,y_train)

predictions = predictions_knn(knn_1, x_test)

# Execution 
        
"""ENTRAINEMENT DU KNN POUR LA POSITION"""
# Chargement des données
path_niveaux = 'Data/niveaux.csv'
x2, y2 = read_csv(path_niveaux)

# Séparation des données

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=test_size_knn,random_state=n)
#random_state 1 coupure aléatoire 0 toujours au même endroit
# Entrainement du knn
knn_2 = initialize_knn_niveau(x_train2, y_train2)
train_knn_niveau(knn_2,x_train2,y_train2)

predictions2 = predictions_knn_niveau(knn_2, x_test2)

"""ENTRAINEMENT DU KNN POUR LA POSITION"""
# Chargement des données
path_face = 'Data/face.csv'
x3, y3 = read_csv(path_face)

# Séparation des données

x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=test_size_knn,random_state=n)
# Entrainement du knn
knn_3 = initialize_knn_niveau(x_train3, y_train3)
train_knn_niveau(knn_3,x_train3,y_train3)

predictions3 = predictions_knn_niveau(knn_3, x_test3)

def pourcentage_bon(predi,y_test):
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

pourcentage_réussite,erreur = pourcentage_bon(predictions,y_test)
pourcentage_réussite_2,erreur2 = pourcentage_bon(predictions2,y_test2)
pourcentage_réussite_3,erreur = pourcentage_bon(predictions3,y_test3)

""" UTILISATION DU CODE SUR UNE IMAGE AVEC DATASET DEJA FAIT """
def test_une_image(path):
    list_coord=dp.vector_to_face_from_path(path, min_detection_confidence=0.7, display=True)
    if type(list_coord)!=type(None):
        columns=data.labels_csv_face()[1:]
        df_coord = DataFrame([list_coord],columns=columns)
        return(predictions_knn_niveau(knn_3, df_coord))


if __name__ == '__main__':
    """ print(predictions)
    print(list(y_test))
    print(pourcentage_réussite,erreur)

    print(predictions3)
    print(list(y_test3))
    print(pourcentage_réussite_3,erreur)"""
    print(test_une_image('dataset/LPC/Capture3.PNG'))
    




