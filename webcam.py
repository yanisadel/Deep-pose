import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from utils import *
from niveaux import *
from signes import *
import random as rd
from sklearn.model_selection import train_test_split
from detection_position import *
from data import *


""" Executer le script """
def retourne_knn_entraine(path='Data/signes.csv'):
    x, y = read_csv(path)

    # Séparation des données
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    # Entrainement du knn
    knn = initialize_knn(x_train, y_train)
    train_knn(knn,x_train,y_train)
    return knn
def webcam_signe():
    """q pour sortir"""
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)

    mpDraw = mp.solutions.drawing_utils
    while True  :

        success, frame = cap.read()

        
        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        knn = retourne_knn_entraine(path='Data/signes.csv')

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
            l = results.multi_hand_landmarks[0]
            for indice in range(21):
                position = l.landmark[indice]
                res['pos'+str(indice)+'x'] = [position.x]
                res['pos'+str(indice)+'y'] = [position.y]
                res['pos'+str(indice)+'z'] = [position.z]
            """position = l.landmark[20]
            res['pos'+str(20)+'x'] = [position.x]
            res['pos'+str(20)+'y'] = [position.y]"""
            """for id, ln in enumerate(handml.landmark):
                    print (id, ln)"""
            df = pd.DataFrame(res)
            predictions  = predictions_knn(knn, df)
            #print (predictions)
            pr = str(predictions[0])
            cv2.putText(frameflip, 'Geste : '+ pr , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
            #print (df)        
            #print (results.multi_hand_landmarks)
            #print (results.multi_handedness) 

        
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()

def webcam_position():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)

    mpDraw = mp.solutions.drawing_utils
    while True  :

        success, frame = cap.read()
        
        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        knn = retourne_knn_entraine(path='Data/face.csv')

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
            l = results.multi_hand_landmarks[0]
            list_coord=vector_to_face(frame, min_detection_confidence=0.7, display=True)
            if type(list_coord)!=type(None):
                columns=labels_csv_face()[1:]
                df = pd.DataFrame([list_coord],columns=columns)
                """position = l.landmark[20]
                res['pos'+str(20)+'x'] = [position.x]
                res['pos'+str(20)+'y'] = [position.y]"""
                """for id, ln in enumerate(handml.landmark):
                        print (id, ln)"""
                predictions  = predictions_knn(knn, df)
                #print (predictions)
                pr = str(predictions[0])
                cv2.putText(frameflip, 'Position : '+ pr , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
                #print (df)        
                #print (results.multi_hand_landmarks)
                #print (results.multi_handedness) 

        
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    """q pour sortir"""
    #print(webcam_signe())
    print(webcam_position())
    