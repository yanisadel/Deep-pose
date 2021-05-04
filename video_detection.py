import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from utils import *
import random as rd
from sklearn.model_selection import train_test_split
from predictions import *
from formatage import *
import matplotlib.pylab as plt
import time

def graphique_video_signe(video, min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, 
    Pour chacune des frames elle fait la prediction et stocke les probabilités dans des listes pour chacun de signes de longueur 10
    Pour chaque frame elle fait la moyenne de chachune de ces listes et stockes les résultats
    donc on a la moyenne de probabilité d'apparition de chacun des signes pour chaque frame
    On retourne le graphique correspondant avec en abscisse les frames et en ordonnées les probabilites

    Arguments
    ---------
    video: video cv2
        video qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon
    
    returns
    -----------
    none affiche graphique
    """
    c=0
    i=0
    X=[]
    l=[0 for k in range(10)]
    Y_temp=[l[:] for k in range(8)]
    Y_moyenne=[[],[],[],[],[],[],[],[]]
    for frame in generateur_decoupe_video(video):
        knn=knn_entraine('data_train/signes.csv','signe')
        if(prediction_image_proba(knn,frame,'signe', min_detection_confidence)!=None):
            c+=1
            X.append(c)
            prediction,probas=prediction_image_proba(knn,frame,'signe', min_detection_confidence)
            for j in range(8):
                Y_temp[j][i]=probas[j]
                Y_moyenne[j].append(sum(Y_temp[j])/10)
            if i==9:
                i=0
            else:
                i+=1
    for j in range(8):
        plt.plot(X,Y_moyenne[j],label='y=signe ' + str(j+1))
    plt.legend()
    plt.grid(True)
    plt.show()

def graphique_video_signe_from_path(path, min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, 
    Pour chacune des frames elle fait la prediction et stocke les probabilités dans des listes pour chacun de signes de longueur 10
    Pour chaque frame elle fait la moyenne de chachune de ces listes et stockes les résultats
    donc on a la moyenne de probabilité d'apparition de chacun des signes pour chaque frame
    On retourne le graphique correspondant avec en abscisse les frames et en ordonnées les probabilites

    Arguments
    ---------
    path: str
        chemind de la video qu'on analyse

    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 

    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon

    returns
    -----------
    none affiche graphique
    """
    video = cv2.VideoCapture(path)
    graphique_video_signe(video, min_detection_confidence, display)



def graphique_video_position(video, min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, 
    Pour chacune des frames elle fait la prediction et stocke les probabilités dans des listes pour chacun de signes de longueur 10
    Arguments
    ---------
    video: video cv2
        video qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon
    """
    c=0
    i=0
    X=[]
    l=[0 for k in range(10)]
    Y_temp=[l[:] for k in range(5)]
    Y_moyenne=[[],[],[],[],[],[],[],[]]
    knn=knn_entraine('data_train/dlib.csv','position')
    for frame in generateur_decoupe_video(video):
        if(prediction_image_proba(knn,frame,'position', min_detection_confidence)!=None):
            c+=1
            X.append(c)
            prediction,probas=prediction_image_proba(knn,frame,'position', min_detection_confidence)
            for j in range(5):
                Y_temp[j][i]=probas[j]
                Y_moyenne[j].append(sum(Y_temp[j])/10)
            if i==9:
                i=0
            else:
                i+=1
    for j in range(5):
        plt.plot(X,Y_moyenne[j],label='y=niveau ' + str(j+1))
    plt.legend()
    plt.grid(True)
    plt.show()

def graphique_video_position_from_path(path, min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, 
    Pour chacune des frames elle fait la prediction et stocke les probabilités dans des listes pour chacun de signes de longueur 10
    Arguments
    ---------
    video: video cv2
        video qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon
    """
    video = cv2.VideoCapture(path)
    graphique_video_position(video, min_detection_confidence, display)

if __name__ == '__main__':
    print(graphique_video_signe_from_path('data_test/video_maison/WIN_20210415_10_12_37_Pro_Trim.mp4'))      
    #print(graphique_video_position_from_path('data_test/video_maison/WIN_20210415_10_12_37_Pro_Trim.mp4'))