import pandas as pd
import matplotlib.pyplot as plt

def normalize_list_points(l):
    """
    normalize_list_points prend en argument une liste de coordonnées comprises entre 0 et 1, et effectue une normalisation de ces coordonnées, pour 
    que les valeurs 0 et 1 soient atteintes
    Ca peut par exemple résoudre le problème lié au fait que les gens n'ont pas tous la même taille de main

    Arguments
    ---------
    l: list
        Liste des coordonnées de la forme [pos1, pos2, pos3, ...]
    
    Returns
    -------
    list
    """

    l = l[:]
    n = len(l)
    maxi = max(l)
    

    # On récupère le coefficient par lequel on va multiplier tous les points
    coeff = 0
    if (maxi > 0):
        coeff = 1.0/maxi

    # On met tout à jour
    l = l*coeff

    return l


def normalize_data(df):
    """
    normalize_data applique la fonction normalize_list_points sur chaque ligne du dataframe df
    """
    df_copy = df.copy()
    n = len(df)
    for i in range(n):
        df_copy.values[i] = normalize_list_points(df_copy.values[i])
    return df_copy



def read_csv(path='Data/signes.csv'):
    """
    read_csv lit un fichier excel du dataset, et renvoie deux dataframe pandas : un qui contient les labels, et l'autre qui contient les positions de main

    Arguments
    ---------
    path: str
        chemin du fichier

    Returns
    -------
    x: DataFrame
        dataframe qui contient les positions de la main
    
    y: DataFrame
        dataframe qui contient les labels de chaque image
    """
    df = pd.read_csv(path, sep=',', index_col=False)
    y = df['label']
    x = df.drop(columns=['label'])
    return x, y

