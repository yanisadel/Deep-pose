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

def graphique_video(video, min_detection_confidence=0.7, display=True):
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
    Y_temp=[l[:] for k in range(8)]
    Y_moyenne=[[],[],[],[],[],[],[],[]]
    for frame in generateur_decoupe_video(video):
        if(prediction_image_proba(knn_signes,frame,'signe', min_detection_confidence)!=None):
            c+=1
            X.append(c)
            prediction,probas=prediction_image_proba(knn_signes,frame,'signe', min_detection_confidence)
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
def graphique_video_from_path(path, min_detection_confidence=0.7, display=True):
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
    graphique_video(video, min_detection_confidence, display)
"""
def video_from_path(path):
        cap = cv2.VideoCapture(path)
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5)

        mpDraw = mp.solutions.drawing_utils
        while (cap.isOpened())  :

            success, frame = cap.read()
            
            if success == False :
                break
            
            frameflip = cv2.flip(frame.copy(), 1)
            
            imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB) 

            res = {}

            knn = knn_entraine(path='data_train/signes.csv','signe')

            if results.multi_hand_landmarks:
                for handml in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
                l = results.multi_hand_landmarks[0]
                for indice in range(21):
                    position = l.landmark[indice]
                    res['pos'+str(indice)+'x'] = [position.x]
                    res['pos'+str(indice)+'y'] = [position.y]
                    res['pos'+str(indice)+'z'] = [position.z]
                #position = l.landmark[20]
                #res['pos'+str(20)+'x'] = [position.x]
                #res['pos'+str(20)+'y'] = [position.y]
                #for id, ln in enumerate(handml.landmark):
                        #print (id, ln)
                df = pd.DataFrame(res)
                predictions  = predictions_knn(knn, df)
                #print (predictions)
                pr = str(predictions[0])
                cv2.putText(frameflip, 'Geste : '+ pr , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
                #print (df)        
                #print (results.multi_hand_landmarks)
                #print (results.multi_handedness) 
                probability_class= knn.predict_proba(df)
                return(probability_class)


            
            cv2.imshow('cam', frameflip)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        cap.release()
        cv2.destroyAllWindows()"""


if __name__ == '__main__':
    print(graphique_video_from_path('data_test/video_maison/WIN_20210415_10_12_37_Pro_Trim.mp4'))      