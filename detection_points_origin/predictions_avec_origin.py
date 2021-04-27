import pandas as pd
from utils_avec_origin import *
from signes_avec_origin import *

from sklearn.model_selection import train_test_split



"""PARAMETRES"""
test_size_knn = 0.1 # Proportion pour les données de train et de test
test_size_tree = 0.1


"""PREDICTION DU SIGNE DE LA MAIN"""
# Chargement des données
path_signes = 'Data_avec_origin/signes.csv'
x, y = read_csv(path_signes)

# Séparation des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=0)

# Entrainement du knn
knn = initialize_knn(x_train, y_train)
train_knn(knn,x_train,y_train)

predictions = predictions_knn(knn, x_test)

# Afficher les résultats et calculer le pourcentage d'echecs

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






