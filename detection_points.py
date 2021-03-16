import mediapipe as mp
import cv2
from formatage import *
from affichage import *


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
    points_image retourne un dictionnaire contenant les positions des différents points de la main (index, pouce...)
    Il y a 21 points différents accessibles indexés de 0 à 20 (voir la documentation de mediapipe)

    Arguments
    ---------
    image: image cv2
        image qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    dict
        Retourne un dictionnaire du style :
        { 'Left' : {0 : {x: 0.54, y: 0.87, z: 0},    (0 = WRIST)
                    1 : {x: 0.72, y: 0.11, z: 0},  (1 = THUMB_CMC)
                    2 : {x: 0.13, y: 0.17, z: 0}   (2 = THUMB_MCP)
                    },

          'Right' : {0 : {x: 0.54, y: 0.87, z: 0},    (0 = WRIST)
                    1 : {x: 0.72, y: 0.11, z: 0},  (1 = THUMB_CMC)
                    2 : {x: 0.13, y: 0.17, z: 0}   (2 = THUMB_MCP)
                    }
        }

        qui représente les points de la main
        x, y (et pour z je ne sais pas) sont les coordonnées des points (divisées par la longueur et la largeur de l'image)
    """

    # On initialise le résultat
    res = dict()
    res['Left'] = None
    res['Right'] = None

    # On flip l'image parce que la documentation dit qu'on a des meilleurs résultats comme ça
    image = flip_image(image)

    # On récupère les données du process
    list_hands, multi_hand_landmarks = image_process(image)

    # On prépare l'image qui contiendra les annotations (les points sur la main)
    annotated_image = image.copy()

    # Si on n'a trouvé 0 main, on renvoie None
    if list_hands == None:
        print("Aucune main n'a été reconnue sur l'image")
        if display:
            affiche_image(image)
        return None, None

    # Ici, on a donc trouvé soit une, soit deux mains    
    else:
        n = len(list_hands)
        # Les prochaines lignes (if/else) servent juste à récupérer les bons indices pour la main droite et la main gauche
        l = []
        if n == 1:
            l.append(list_hands[0].classification[0].label)
        else:
            if (list_hands[0].classification[0].index == 0):
                l.append(list_hands[0].classification[0].label)
                l.append(list_hands[1].classification[0].label)
            else:
                l.append(list_hands[1].classification[0].label)
                l.append(list_hands[0].classification[0].label)
    
        i = 0    
        for hand_landmarks in multi_hand_landmarks: # Ici, results.multi_hand_landmarks est un liste de longueur 0 si on a trouvé 0 main, 1 si une main, et 2 si deux mains
            res[l[i]] = dict()
            for indice in range(21):
                # Chaque indice représente une partie de la main (voir la documentation, par exemple 0 = poignet)
                position = hand_landmarks.landmark[indice] # éventuellement multiplier par image_width pour les x, et height pour les y
                res[l[i]][indice] = position

            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            i += 1

        annotated_image = flip_image(annotated_image)
        if display:
            affiche_image(annotated_image)
            

        return res, annotated_image


def points_image_from_path(path="photos/hand5.jpg", min_detection_confidence=0.7, display=True):    
    """
    points_image retourne un dictionnaire contenant les positions des différents points de la main (index, pouce...)
    Il y a 21 points différents accessibles indexés de 0 à 20 (voir la documentation de mediapipe)

    Arguments
    ---------
    path: str
        chemin de l'image qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher l'image
        False sinon

    Returns
    -------
    dict
        Retourne un dictionnaire du style :
        { 'Left' : {0 : {x: 0.54, y: 0.87, z: 0},    (0 = WRIST)
                    1 : {x: 0.72, y: 0.11, z: 0},  (1 = THUMB_CMC)
                    2 : {x: 0.13, y: 0.17, z: 0}   (2 = THUMB_MCP)
                    },

          'Right' : {0 : {x: 0.54, y: 0.87, z: 0},    (0 = WRIST)
                    1 : {x: 0.72, y: 0.11, z: 0},  (1 = THUMB_CMC)
                    2 : {x: 0.13, y: 0.17, z: 0}   (2 = THUMB_MCP)
                    }
        }

        qui représente les points de la main
        x, y (et pour z je ne sais pas) sont les coordonnées des points (divisées par la longueur et la largeur de l'image)
    """
    # On charge l'image
    image = cv2.imread(path)

    return points_image(image, min_detection_confidence, display)


def points_video(video, min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, et pour chacune des frames, récupère les points de la main grâce à la fonction points_image_from_image

    Arguments
    ---------
    video: video cv2
        video qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon

    Returns
    -------
    Une liste où l'indice i correspond à la frame i de la vidéo, et l[i] contient le dictionnaire {left: [...]}
    """

    list_points = []

    for frame in generateur_decoupe_video(video):
        points, image = points_image(frame, min_detection_confidence, display=False)
        if (type(points) != type(None)):
            list_points.append(points)
    
    return list_points


def points_video_from_path(path='video_test.mp4', min_detection_confidence=0.7, display=True):
    """
    Cette fonction récupère tous les points de la main sur une vidéo
    Elle découpe la vidéo en frames, et pour chacune des frames, récupère les points de la main grâce à la fonction points_image_from_image

    Arguments
    ---------
    path: str
        chemin de la vidéo qu'on analyse
    
    min_detection_confidence: float
        le degré de confiance que l'on veut quant à la précision de l'analyse 
    
    display: bool
        vaut True si la fonction doit afficher la vidéo
        False sinon
    """
    video = cv2.VideoCapture(path)
    return points_video(video, min_detection_confidence, display)

