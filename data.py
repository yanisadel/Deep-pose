import mediapipe as mp
import cv2
import csv
from formatage import *
from face import *
from cv2 import imread, resize
from os import listdir

def image_process(image, min_detection_confidence=0.7):
    # On charge le modèle
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence)
    
    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output and process it with MediaPipe Hands.
    results = hands.process(image_bgr_to_rgb(image))

    return results.multi_handedness, results.multi_hand_landmarks


def points_image(image, min_detection_confidence=0.7, display=True):
    """
    points_image retourne une liste contenant les positions des différents points de la main (index, pouce...)
    L'image en argument n'est censée contenir que la MAIN DROITE (sinon il peut éventuellement y avoir des erreurs)
    Il y a 21 points différents accessibles indexés de 0 à 20 (voir la documentation de mediapipe)

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
           [pos1x, pos1y, pos1z, pos2x, pos2y, ...]

        qui représente les points de la main (DROITE)
        x, y (et pour z je ne sais pas) sont les coordonnées des points (divisées par la longueur et la largeur de l'image)
    """

    # On initialise le résultat
    res = []

    # On flip l'image parce que la documentation dit qu'on a des meilleurs résultats comme ça
    image = flip_image(image)

    # On récupère les données du process
    list_hands, multi_hand_landmarks = image_process(image, min_detection_confidence=min_detection_confidence)

    # Si on n'a trouvé 0 main, on renvoie None
    if list_hands == None:
        #print("Aucune main n'a été reconnue sur l'image")        
        return None

    # Ici, on a donc trouvé soit une, soit deux mains    
    else: # Si on a trouvé qu'une seule main
        l = multi_hand_landmarks[0]
        for indice in range(21):
            # Chaque indice représente une partie de la main (voir la documentation, par exemple 0 = poignet)
            position = l.landmark[indice]
            res += [position.x, position.y, position.z]

        return res


def points_image_from_path(path="main.jpg", min_detection_confidence=0.7, display=True):    
    """
    points_image_from_path fait la même chose que points_image, la seule différence est qu'elle prend le chemin de l'image, et non l'image directement
    """

    # On charge l'image
    image = cv2.imread(path)

    return points_image(image, min_detection_confidence, display)


def labels_csv():
    """
    labels_csv() renvoie la 1ère ligne des tableaux excel (dans l'ordre : label, pos0x, pos0y, pos0z, pos1x, pos1y...)
    Elle renvoie une liste
    """

    l = ["label"]
    for i in range(21):
        l.append("pos" + str(i) +"x")
        l.append("pos" + str(i) +"y")
        l.append("pos" + str(i) +"z")

    return l[:-1]



def fill_csv_signes(min_detection_confidence=0.7, show_error=True):
    """
    fill_csv_signes crée et remplit le fichier excel Data/signes.csv, qui contient les coordonnées des mains de toutes les images du dataset, avec les labels correspondant aux signes

    Arguments
    ---------
    min_detection_confidence: int
        degré de confiance qu'on veut (compris entre 0 et 1)
    
    show_error: bool
        True si on veut que la fonction affiche sur quelles photos elle n'arrive pas à détecter la main
        False sinon
    """

    with open('Data/signes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quotechar='/', quoting=csv.QUOTE_MINIMAL)
        # On met les labels
        labels = labels_csv()
        writer.writerow(labels)

        # On complète les données
        s = "Data/Signes/"
        echecs = [] # Contient juste le nombre d'échecs de reconnaissance des points
        for i in range(1,9):
            s += str(i) + "/"
            compteur_echecs = 0
            compteur_total = 0
            for path in listdir(s):
                l = points_image_from_path(s + path, min_detection_confidence=min_detection_confidence)
                if (l != None):
                    l = [i] + l
                    writer.writerow(l)
                else:
                    compteur_echecs += 1
                    print("Mediapipe n'a pas réussi à détecter les points sur : " + str(i) + "/" + path)
                compteur_total += 1

            s = "Data/Signes/"
            pourcentage = compteur_echecs/compteur_total*100
            pourcentage = str(pourcentage) + '%'
            echecs.append((i,pourcentage))

        if show_error:
            print("Le pourcentage d'échecs par catégorie est : ", echecs)


def fill_csv_niveaux(min_detection_confidence=0.7, show_error=True):
    """
    fill_csv_signes crée et remplit le fichier excel Data/niveaux.csv, qui contient les coordonnées des mains de toutes les images du dataset, avec les labels correspondant
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
                l = points_image_from_path(s + path, min_detection_confidence=min_detection_confidence)
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

def face_csv(show_error=True):
    l = ["label"]
    for i in range(4):
        l.append("pos" + str(i) +"x")
        l.append("pos" + str(i) +"y")
        l.append("pos" + str(i) +"z")
    with open('Data/face.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quotechar='/', quoting=csv.QUOTE_MINIMAL)
        # On met les labels
        labels = l
        writer.writerow(labels)
        s = "Data/Signes/"
        
        for i in range(1,6):
            p = s+ str(i) + "/"
            for path in listdir(p):
                face = face_img(p+ path)
                if (face != None):
                    M=[]
                    for mark in face :
                        M.append(face[mark].x)
                        M.append(face[mark].y)
                        M.append(0)
                    M= [i]+ M
                    writer.writerow(M)




# Il faut ces lignes là pour remplir les fichiers excel (qui constituent le dataset)
min_detection_confidence = 0.5
#fill_csv_signes(min_detection_confidence=min_detection_confidence)
#fill_csv_niveaux(min_detection_confidence=min_detection_confidence)
face_csv()