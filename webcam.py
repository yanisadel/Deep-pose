import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from utils import *
import random as rd
from sklearn.model_selection import train_test_split
from detection_position import *
from data import *
from predictions import *
from video_detection import *

def webcam_signe():
    """q pour sortir"""
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)

    mpDraw = mp.solutions.drawing_utils
    knn= knn_entraine('data_train/signes.csv','signe')
    while True  :

        success, frame = cap.read()

        
        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}
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
            df = normalize_data(df)
            predictions  = knn.predict(df)
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
    knn= knn_entraine('data_train/face.csv','position')
    while True  :

        success, frame = cap.read()
        
        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
            l = results.multi_hand_landmarks[0]
            prediction=prediction_position_image(knn,frame,0.7)
            #print (predictions)
            if type(prediction)!=type(None):
                pr = str(prediction[0])
                cv2.putText(frameflip, 'Position : '+ pr , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
            #print (df)        
            #print (results.multi_hand_landmarks)
            #print (results.multi_handedness) 

        
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()

def webcam_position_dlib():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)

    mpDraw = mp.solutions.drawing_utils
    knn= knn_entraine('data_train/dlib.csv','position')
    while True  :

        success, frame = cap.read()
        
        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
            l = results.multi_hand_landmarks[0]
            prediction=prediction_image(knn,frame,'position',0.7)
            #print (predictions)
            if type(prediction)!=type(None):
                pr = str(prediction[0])
                cv2.putText(frameflip, 'Position : '+ pr , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
            #print (df)        
            #print (results.multi_hand_landmarks)
            #print (results.multi_handedness) 

        
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()

def webcam_graphique_signe(min_detection_confidence=0.7):
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)
    c=0
    i=0
    X=[]
    l=[0 for k in range(5)]
    Y_temp=[l[:] for k in range(8)]
    Y_moyenne=[[],[],[],[],[],[],[],[]]
    knn= knn_entraine('data_train/signes.csv','signe')

    while True  :
        success, frame = cap.read()

        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
        if(prediction_image_proba(knn,frame,'signe')!=None):
            c+=1
            X.append(c)
            prediction,probas=prediction_image_proba(knn,frame,'signe')
            for j in range(8):
                Y_temp[j][i]=probas[j]
                Y_moyenne[j].append(sum(Y_temp[j])/5)
                if Y_moyenne[j][-1]>=min_detection_confidence:
                    cv2.putText(frameflip, 'signe : '+ str(j+1) , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
            if i==4:
                i=0
            else:
                i+=1     
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for j in range(8):
        plt.plot(X,Y_moyenne[j],label='y=signe ' + str(j+1))
    plt.legend()
    plt.grid(True)
    plt.show()
    cap.release()

def webcam_graphique_position(min_detection_confidence=0.7):
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)
    c=0
    i=0
    X=[]
    l=[0 for k in range(5)]
    Y_temp=[l[:] for k in range(8)]
    Y_moyenne=[[],[],[],[],[],[],[],[]]
    knn= knn_entraine('data_train/dlib.csv','positione')

    while True  :
        success, frame = cap.read()

        frameflip = cv2.flip(frame.copy(), 1)
        
        imgRGB = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB) 

        res = {}

        if results.multi_hand_landmarks:
            for handml in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frameflip, handml, mpHands.HAND_CONNECTIONS)
        if(prediction_image_proba(knn,frame,'position')!=None):
            c+=1
            X.append(c)
            prediction,probas=prediction_image_proba(knn,frame,'position')
            for j in range(5):
                Y_temp[j][i]=probas[j]
                Y_moyenne[j].append(sum(Y_temp[j])/5)
                if Y_moyenne[j][-1]>=min_detection_confidence:
                    cv2.putText(frameflip, 'position : '+ str(j+1) , (10,70), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,2), 2)
            if i==4:
                i=0
            else:
                i+=1     
        cv2.imshow('cam', frameflip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for j in range(5):
        plt.plot(X,Y_moyenne[j],label='y=position ' + str(j+1))
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    """q pour sortir"""
    #print(webcam_signe())
    #print(webcam_position())
    print(webcam_position_dlib())
    #print(webcam_graphique_signe())

    