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
from display_vecteur import *


def main(path,min_detection_confidence=0.8,display=True):
    """
    La fonction main prend en argument le chemin d'une vidéo, et affiche la vidéo avec les signes et positions détectés
    Elle initialise et entraine un modèle, puis analyse la vidéo
    """
    cap = cv2.VideoCapture(path)
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)
    c=0
    i_signe,i_position=0,0
    X=[]
    n_signe=5
    n_position=5
    l_signe=[0 for k in range(n_signe)]
    l_position=[0 for k in range(n_position)]
    Y_temp_signe=[l_signe[:] for k in range(8)]
    Y_moyenne_signe=[[],[],[],[],[],[],[],[]]
    Y_temp_position=[l_position[:] for k in range(5)]
    Y_moyenne_position=[[],[],[],[],[],[],[],[]]
    knn_signe= knn_entraine('data_train/signes.csv','signe')
    knn_position=knn_entraine('data_train/dlib.csv','position')
    
    while True  :
        success, frame = cap.read()
        frameflip = frame.copy()
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB) 
        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)   

        if(prediction_image_proba(knn_signe,frame,'signe')!=None) and (prediction_image_proba(knn_position,frame,'position')!=None):
            frameflip= display_vector_from_image(frame.copy()) 
            c+=1
            X.append(c)
            prediction_signe,probas_signe=prediction_image_proba(knn_signe,frame,'signe')
            prediction_position,probas_position=prediction_image_proba(knn_position,frame,'position')
            for j in range(8):
                Y_temp_signe[j][i_signe]=probas_signe[j]
                Y_moyenne_signe[j].append(sum(Y_temp_signe[j])/n_signe)
                if Y_moyenne_signe[j][-1]>=min_detection_confidence:
                    cv2.putText(frameflip, 'signe : '+ str(j+1) , (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,2), 2)
            for j in range(5):
                Y_temp_position[j][i_position]=probas_position[j]
                Y_moyenne_position[j].append(sum(Y_temp_position[j])/n_position)
                if Y_moyenne_position[j][-1]>=min_detection_confidence:
                    cv2.putText(frameflip, 'position : '+ str(j+1) , (10,110), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,2), 2)
            if i_signe==n_signe-1:
                i_signe=0
            else:
                i_signe+=1
            if i_position==n_position-1:
                i_position=0
            else:
                i_position+=1 
        cv2.imshow('cam', frameflip)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    #print(main('data_test/video_maison/WIN_20210415_10_12_37_Pro_Trim.mp4'))
    #print(main('data_test/video_maison/repetezlecode.mp4'))
    #print(main('data_test/video_maison/alphabet.mp4'))
    #print(main('data_test/video_maison/1.mp4'))
    #print(main('data_test/video_maison/1_question.mp4'))
    print(main('data_test/video_maison/1_grenouille.mp4'))
