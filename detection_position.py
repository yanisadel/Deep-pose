import detection_tete.face_detection
import mediapipe as mp
import cv2
import csv
from formatage import *
from cv2 import imread, resize
from os import listdir
from data import *
min_detection_confidence=0.7
display=True

def normalize_vector(u):
    return((u[0]/(u[0]**2+u[1]**2)**0.5),(u[1]/(u[0]**2+u[1]**2)**0.5))


def vector_to_rectangle(img, min_detection_confidence=0.7, display=True):
    points=points_image(img, min_detection_confidence=0.7, display=True)
    res=[]
    if type(points)!=type(None) and type(detection_tete.face_detection.coordinate_face(img))!=type(None):
        u1,u2,u3,u4=detection_tete.face_detection.coordinate_face(img)
        l=[u1,u2,u3,u4]
        for i in range (0,21,4):
            xr,yr=points[i:i+2]          
            for j in range(4):
                res+=list((normalize_vector((xr-l[j][0],yr-l[j][1]))))
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

def fill_csv_niveaux(min_detection_confidence=0.7, show_error=True):
    """
    fill_csv_signes crée et remplit le fichier excel Data/niveaux.csv, qui contient les vecteur des doigts aux sommets du rectangle de la tête, avec les labels correspondant
    aux niveaux de main

    Arguments
    ---------
    min_detection_confidence: int
        degré de confiance qu'on veut (compris entre 0 et 1)
    
    show_error: bool
        True si on veut que la fonction affiche sur quelles photos elle n'arrive pas à détecter la main
        False sinon
    """

    with open('Data/niveaux.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quotechar='/', quoting=csv.QUOTE_MINIMAL)
        # On met les labels
        labels = labels_csv()
        writer.writerow(labels)

        # On complète les données
        s = "Data/Niveaux/"
        echecs = [] # Contient juste le nombre d'échecs de reconnaissance des points
        for i in range(1,6):
            s += str(i) + "/"
            compteur_echecs = 0
            compteur_total = 0
            for path in listdir(s):
                l = vector_to_rectangle_from_path(s + path, min_detection_confidence=min_detection_confidence)
                if (l != None):
                    l = [i] + l
                    writer.writerow(l)
                else:
                    compteur_echecs += 1
                    print("Mediapipe n'a pas réussi à détecter les points sur : " + str(i) + "/" + path)
                compteur_total += 1

            s = "Data/Niveaux/"
            pourcentage = compteur_echecs/compteur_total*100
            pourcentage = str(pourcentage) + '%'
            echecs.append((i,pourcentage))

        if show_error:
            print("Le pourcentage d'échecs par catégorie est : ", echecs)

if __name__ == '__main__':
    #print(normalize_vector((1,2)))
    fill_csv_niveaux()





