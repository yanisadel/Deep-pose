import tensorflow as tf
from detection_points import *

def transforme_liste_points(d):
    if d == None:
        return []
    
    else:
        l = []
        for i in range(21):
            l.append([d[i][0], d[i][1], d[i][2]])
        return l


def transforme_data(l):
    n = len(l)
    main_gauche = []
    main_droite = []
    for i in range(n):
        main_gauche.append(transforme_liste_points(l[i]['Left']))
        main_gauche.append(transforme_liste_points(l[i]['Right']))

    return main_gauche, main_droite

list_points_video = points_video_from_path()
