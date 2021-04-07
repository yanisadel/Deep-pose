<<<<<<< HEAD
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
        Liste des coordonnées de la forme [pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, pos2x...]
    
    Returns
    -------
    list
    """

    l = l[:]
    n = len(l)
    l_x = l[::3] # La liste des positions de x
    l_y = l[1::3] # La liste des positions de y
    mini_x, maxi_x = min(l_x), max(l_x)
    mini_y, maxi_y = min(l_y), max(l_y)
    
    # On place la main "en haut à gauche de l'écran" (càd on fait en sorte que la coordonnée minimale selon x soit 0, en retirant le minimum), et on fait pareil avec y
    for i in range(0,n,3):
        l[i] -= mini_x
        if (i < n-1):
            l[i+1] -= mini_y

    # On met à jour les max
    maxi_x -= mini_x
    maxi_y -= mini_y

    # On récupère le coefficient par lequel on va multiplier tous les points
    maxi = max(maxi_x, maxi_y)
    coeff = 0
    if (maxi > 0):
        coeff = 1.0/maxi

    # On met tout à jour
    for i in range(0,n,3):
        l[i] *= coeff
        if (i < n-1):
            l[i+1] *= coeff

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




def verification_normalisation(x_train):
    """C'est juste une fonction qui affiche une liste de coordonnées, et la même liste mais normalisée, pour vérifier si la 
    fonction normalize_list_points fonctionne bien"""
    l = x_train.values[2]
    lcopy = l[:]

    def decoupe(l):
        x = l[::3]
        y = l[1::3]
        return x,y

    n = normalize_list_points(l)

    x1, y1 = decoupe(lcopy)
    x2, y2 = decoupe(n)

    plt.scatter(x1,y1)
    plt.figure()
    plt.scatter(x2,y2)
=======
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
        Liste des coordonnées de la forme [pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, pos2x...]
    
    Returns
    -------
    list
    """

    l = l[:]
    n = len(l)
    l_x = l[::3] # La liste des positions de x
    l_y = l[1::3] # La liste des positions de y
    mini_x, maxi_x = min(l_x), max(l_x)
    mini_y, maxi_y = min(l_y), max(l_y)
    
    # On place la main "en haut à gauche de l'écran" (càd on fait en sorte que la coordonnée minimale selon x soit 0, en retirant le minimum), et on fait pareil avec y
    for i in range(0,n,3):
        l[i] -= mini_x
        if (i < n-1):
            l[i+1] -= mini_y

    # On met à jour les max
    maxi_x -= mini_x
    maxi_y -= mini_y

    # On récupère le coefficient par lequel on va multiplier tous les points
    maxi = max(maxi_x, maxi_y)
    coeff = 0
    if (maxi > 0):
        coeff = 1.0/maxi

    # On met tout à jour
    for i in range(0,n,3):
        l[i] *= coeff
        if (i < n-1):
            l[i+1] *= coeff

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




def verification_normalisation(x_train):
    """C'est juste une fonction qui affiche une liste de coordonnées, et la même liste mais normalisée, pour vérifier si la 
    fonction normalize_list_points fonctionne bien"""
    l = x_train.values[2]
    lcopy = l[:]

    def decoupe(l):
        x = l[::3]
        y = l[1::3]
        return x,y

    n = normalize_list_points(l)

    x1, y1 = decoupe(lcopy)
    x2, y2 = decoupe(n)

    plt.scatter(x1,y1)
    plt.figure()
    plt.scatter(x2,y2)
>>>>>>> c9ff0ab8f8b2e63cddd478911dd1c90068c48ae4
    plt.show()