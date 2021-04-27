import detection_tete.face_detection
import mediapipe as mp
import cv2
import csv
from formatage import *
from cv2 import imread, resize
from os import listdir
import data
import numpy as np
min_detection_confidence=0.7
display=True
import face
from points_dlib import *


def vector_to_rectangle(img, min_detection_confidence=0.7, display=True):
    points=data.points_image(img, min_detection_confidence=0.7, display=True)
    res=[]
    if type(points)!=type(None) and type(detection_tete.face_detection.coordinate_face(img))!=type(None):
        u1,u2,u3,u4=detection_tete.face_detection.coordinate_face(img)
        l=[u1,u2,u3,u4]
        for i in [0,5,9,13,17]:
            xr,yr=points[i:i+2]          
            for j in range(4):
                res+=list((xr-l[j][0],yr-l[j][1]))
        return(res)
    else:
        if points==None:
            print('Pas de points détectés')
        else:
            print('pas de tête détectée')
        return(None)
def vector_to_rectangle_from_path(path, min_detection_confidence=0.7, display=True):    
    """
    vector_to_rectangle_from_path fait la même chose que points_image, la seule différence est qu'elle prend le chemin de l'image, et non l'image directement
    """
    # On charge l'image
    image = cv2.imread(path)

    return vector_to_rectangle(image, min_detection_confidence, display)

def norme(u,v):
    u,v=np.asarray(u),np.asarray(v)
    norm = np.linalg.norm(u-v)
    return norm
    
def vector_to_face(img, min_detection_confidence=0.7, display=True):
    hand_points=data.points_image(img, min_detection_confidence=0.7, display=True)
    res=[]
    head_points=face.face_img(img)
    if type(hand_points)!=type(None) and type(head_points)!=type(None):
        u1,u2,u3,u4=[head_points['mouth'].x,head_points['mouth'].y],[head_points['reye'].x,head_points['reye'].y],[head_points['leye'].x,head_points['leye'].y],[head_points['nose'].x,head_points['nose'].y]
        norm=norme(u2,u3)
        l=[u1,u2,u3,u4]
        #l=[(float(l[i][0]),float(l[i][1]))  for i in range(4)]
        for i in [0,5,9,13,17]:
            xr,yr=hand_points[i:i+2]       
            for j in range(4):
                res+=list(((xr-l[j][0])/norm,(yr-l[j][1])/norm))
        return(res)
    else:
        if hand_points==None:
            print('Pas de points détectés')
        else:
            print('pas de tête détectée')
        return(None)

def vector_to_face_from_path(path, min_detection_confidence=0.7, display=True):
    img=cv2.imread(path)
    return(vector_to_face(img))
###########################################################################################################
def vector_to_face_dlib(img, min_detection_confidence=0.7, display=True):
    hand_points=data.points_image(img, min_detection_confidence=0.7, display=True)
    res=[]
    head_points=points_face(img)
    if type(hand_points)!=type(None) and type(head_points)!=type(None):
        l=[]
    
        norm=norme(head_points[43],head_points[40])
        for i in [2,4,9,14,16,28,34,37,40,43,46,49,55]:
            l.append(head_points[i])
        for i in [0,5,9,13,17]:
            xr,yr=hand_points[i:i+2]       
            for j in range(len(l)):
                res+=list(((xr-l[j][0])/norm,(yr-l[j][1])/norm))
        return(res)
    else:
        if hand_points==None:
            print('Pas de points détectés')
        else:
            print('pas de tête détectée')
        return(None)
#[2,4,9,14,16,28,34,37,40,43,46,49,55]
def vector_to_face_dlib_from_path(path, min_detection_confidence=0.7, display=True):
    img=cv2.imread(path)
    return(vector_to_face_dlib(img))

if __name__ == '__main__':
    print(vector_to_face_dlib_from_path('WIN_20210427_22_01_26_Pro.jpg'))
    print(vector_to_face_from_path('WIN_20210427_22_01_26_Pro.jpg'))
    print(vector_to_face_dlib_from_path('WIN_20210427_22_01_38_Pro.jpg'))
    print(vector_to_face_from_path('WIN_20210427_22_01_38_Pro.jpg'))
    #print(vector_to_face_from_path('data_train/niveaux/1/1-1.jpg'))
    #print(normalize_vector((1,2)))




