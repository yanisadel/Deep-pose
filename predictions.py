import pandas as pd
from utils import *
from niveaux import *
from signes import *

from sklearn.model_selection import train_test_split



"""PARAMETRES"""
test_size_knn = 0.1 # Proportion pour les données de train et de test
test_size_tree = 0.1


"""PREDICTION DU SIGNE DE LA MAIN"""
# Chargement des données
path_signes = 'Data/signes.csv'
x, y = read_csv(path_signes)

# Séparation des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=0)

# Entrainement du knn
knn = initialize_knn(x_train, y_train)
train_knn(knn,x_train,y_train)

predictions = predictions_knn(knn, x_test)




"""PREDICTION DU NIVEAU DE MAIN"""
"""# Chargement des données
path_niveaux = 'Data/niveaux.csv'
x2, y2 = read_csv(path_niveaux)

# Séparation des données
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=test_size_tree, random_state=0)

# Entrainement du modèle
tree = initialize_decision_tree_classifier()
train_decision_tree_classifier(tree, x_train2, y_train2)
predictions2 = predictions_decision_tree_classifier(tree, x_test2)"""

"""ENTRAINEMENT DU KNN POUR LA POSITION"""
# Chargement des données
path_niveaux = 'Data/niveaux.csv'
x2, y2 = read_csv(path_niveaux)

# Séparation des données
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=test_size_knn, random_state=0)
#random_state 1 coupure aléatoire 0 toujours au même endroit
# Entrainement du knn
knn = initialize_knn_niveau(x_train2, y_train2)
train_knn_niveau(knn,x_train2,y_train2)

predictions2 = predictions_knn_niveau(knn, x_test2)

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

pourcentage_réussite_2,erreur2 = pourcentage_bon(predictions2,y_test2)
pourcentage_réussite,erreur = pourcentage_bon(predictions,y_test)

if __name__ == '__main__':
    print(predictions2)
    print(list(y_test2))
    print(pourcentage_réussite_2,erreur2)
    print(predictions)
    print(list(y_test))
    print(pourcentage_réussite,erreur)

    

