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

def show_results (y_test, predictions) :
    """
    Retourne un Pandas Dataframe de la forme :

                  geste  prediction
            52       3           3
            182      8           8
            5        1           1
            18       1           1

    Arguments
    ---------
    y_test 
    predictions 

    Returns
    -------
    df: DataFrame
        
    
    """
    index = y_test.index.to_numpy()
    y = y_test.array.to_numpy()
    df = pd.DataFrame({ 'geste': pd.Series(y, index = index), 'prediction' : pd.Series(predictions, index = index)})
    return df

def echecs (y_test, predictions): 
    """ 
    Calcule le pourcentage d'echecs dans le test

    Arguments
    ---------
       y_test 
       predictions 

    Returns
    -------
    echec : float
           pourcentage d'echecs
    """
    df = show_results(y_test,predictions)
    ges = df['geste'].array.to_numpy()
    pre = df['prediction'].array.to_numpy()
    n = len(df)
    p = 0
    for i in range(n):
        if ges[i] != pre[i]:
            p+=1
    echec = p*100/n
    return echec

# Execution 
print(show_results(y_test,predictions))
print("Le pourcentage d'échecs est : "+ str(int(echecs(y_test,predictions)))+'%')            



"""PREDICTION DU NIVEAU DE MAIN"""
# Chargement des données
path_niveaux = 'Data/niveaux.csv'
x2, y2 = read_csv(path_niveaux)

# Séparation des données
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=test_size_tree, random_state=0)

# Entrainement du modèle
tree = initialize_decision_tree_classifier()
train_decision_tree_classifier(tree, x_train2, y_train2)
predictions2 = predictions_decision_tree_classifier(tree, x_test2)

