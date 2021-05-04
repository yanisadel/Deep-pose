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
import affichage as aff
import matplotlib as plt


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
    """
    vector_to_face_dlib prend en entrée une image et retourne une liste contenant les coordonnées de vecteurs de la tête à la main
    avec les points de la tête obtenue avec mediapipe

    Arguments
    ---------
    image: image cv2
        image qu'on analyse, qui contient une main DROITE
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    list
        Retourne une liste du style :
           [ux0point0, uy0point0, ux0point1, ..]

        qui représente les points de la main (DROITE)
      
    """
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

    """
    vector_to_face_dlib prend en entrée une image et retourne une liste contenant les coordonnées de vecteurs de la tête à la main
    avec les points de la tête obtenue avec mediapipe

    Arguments
    ---------
    path: str
        chemin de l'image qu'on analyse, qui contient une main DROITE
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    list
        Retourne une liste du style :
           [ux0point0, uy0point0, ux0point1, ..]

        qui représente les points de la main (DROITE)
      
    """    
    img=cv2.imread(path)
    return(vector_to_face(img))

def vector_to_face_dlib(img, min_detection_confidence=0.7, display=True):
    """
    vector_to_face_dlib prend en entrée une image et retourne une liste contenant les coordonnées de vecteurs de la tête à la main
    avec les points de la tête obtenue avec dlib

    Arguments
    ---------
    image: image cv2
        image qu'on analyse, qui contient une main DROITE
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    list
        Retourne une liste du style :
           [ux0point1, uy0point1, ux0point4, ..]

        qui représente les points de la main (DROITE)
      
    """
    hand_points=data.points_image(img, min_detection_confidence=0.7, display=True)
    res=[]
    head_points=points_face(img)
    if type(hand_points)!=type(None) and type(head_points)!=type(None) and len(head_points)==68:
        l=[]
        norm=norme(head_points[43],head_points[40])
        for i in [1,4,9,14,17,28,34,37,40,43,46,49,55]:
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


#[1,4,9,14,17,28,34,37,40,43,46,49,55]/[0,5,9,13,17]
def vector_to_face_dlib_from_path(path, min_detection_confidence=0.7, display=True):
    """
    vector_to_face_dlib_from_path prend en entrée le chemin d'une image et fait la même chose que vector_to_face_dlib

    Arguments
    ---------
    path: str
        path de l'image qui contient une main DROITE
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    list
        Retourne une liste du style :
           [ux0point1, uy0point1, ux0point4, ..]

        qui représente les points de la main (DROITE)
      
    """
    img=cv2.imread(path)
    return(vector_to_face_dlib(img))

def points_face2(path,detector=detector,predictor=predictor):
    """
    points_face prend en entrée une image et affiche sur la tête les points détectés par dlib sur la tête si il en détecte

    Arguments
    ---------

    image: image cv2
    image que l'on veut analyser

    detector : truc sombre

    predictor: truc sombre

    Returns
    -------
    None (affiche tête avec les points sur la tête) 
    """
    image= cv2.imread(path)
    image = imutils.resize(image,width=600)
    height,width,_=image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # détecter les visages
    rects = detector(gray, 1)
    points=[]
    # Pour chaque visage détecté, recherchez le repère.
    for (i, rect) in enumerate(rects):
        # déterminer les repères du visage for the face region, then
        # convertir le repère du visage (x, y) en un array NumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
        # dessiner le cadre de sélection
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # boucle sur les coordonnées (x, y) pour les repères faciaux
        # et dessine-les sur l'image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            points.append((x/width,y/height))
    # afficher l'image de sortie avec les détections de visage + repères de visage
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    return(image)

def vect_to_face_img(img):
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handml in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handml, mpHands.HAND_CONNECTIONS)
    """X=[]
    Y=[]
    implot = plt.imshow(img)
    for u in head_points :
        X.append(u[0])
        Y.append(u[1])    
    plt.scatter(X,Y)
    plt.show()
    for u in head_points : 
        print(u)
        cv2.circle(img, (200,200), 1, (0,0,255), -1)"""
    aff.affiche_image(img)

if __name__ == '__main__':
    #print(vector_to_face_dlib_from_path('data_test/LPC/WIN_20210415_08_45_49_Pro.jpg'))
    #print(vector_to_face_from_path('data_test/LPC/WIN_20210415_08_45_49_Pro.jpg'))
    #print(vector_to_face_dlib_from_path('WIN_20210427_22_01_38_Pro.jpg'))
    #print(vector_to_face_from_path('WIN_20210427_22_01_38_Pro.jpg'))
    #print(vector_to_face_from_path('data_train/niveaux/1/1-1.jpg'))
    #print(normalize_vector((1,2)))
    vect_to_face_img()



