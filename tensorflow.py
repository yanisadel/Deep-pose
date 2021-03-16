import tensorflow as tf

def transforme_dictionnaire_points(d):
    """
    Transforme le dictionnaire des points en une liste de points
    """
    if d == None:
        print("on passe ici")
        return []
    
    else:
        l = []
        for i in range(21):
            l.append([d[i].x, d[i].y, d[i].z])
        return l


def transforme_data(l):
    """
    Transforme la liste des points d'une vidéo (qui contient des dictionnaires) en une liste de listes (grâce à la fonction transforme_dictionnaire_points)
    """
    
    n = len(l)
    main_gauche = []
    main_droite = []
    for i in range(n):
        main_gauche.append(transforme_dictionnaire_points(l[i]['Left']))
        main_droite.append(transforme_dictionnaire_points(l[i]['Right']))
    return main_gauche, main_droite

from detection_points import *

l = points_video_from_path()
g, d = transforme_data(l)

a = l[0]['Right']
b = transforme_dictionnaire_points(a)
