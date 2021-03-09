import mediapipe as mp
import cv2


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

    # On initialise le résultat
    res = dict()
    res['Left'] = None
    res['Right'] = None

    # On charge le modèle
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils 
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.6)
    
    # On charge l'image
    image = cv2.imread(path)
    name = 'image_test' # nom associé à l'image
    image_hight, image_width, _ = image.shape

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output and process it with MediaPipe Hands.
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    # On prépare l'image qui contiendra les annotations (les points sur la main)
    annotated_image = cv2.flip(image.copy(), 1)

    # On récupère les infos sur le résultat (nombre de mains récupérées, et précision)
    list_hands = results.multi_handedness

    # Si on n'a trouvé 0 main, on renvoie None
    if list_hands == None:
        print("Aucune main n'a été reconnue sur l'image")
        if display:
            cv2.imshow("image", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return None

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
        for hand_landmarks in results.multi_hand_landmarks: # Ici, results.multi_hand_landmarks est un liste de longueur 0 si on a trouvé 0 main, 1 si une main, et 2 si deux mains
            res[l[i]] = dict()
            for indice in range(21):
                # Chaque indice représente une partie de la main (voir la documentation, par exemple 0 = poignet)
                position = hand_landmarks.landmark[indice] # éventuellement multiplier par image_width pour les x, et height pour les y
                res[l[i]][indice] = position

            mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            i += 1

        if display:
            cv2.imshow("image", cv2.flip(annotated_image, 1))
            cv2.waitKey()
            cv2.destroyAllWindows()

        return res, cv2.flip(annotated_image, 1)


def points_image_from_image(image, min_detection_confidence=0.7, display=True):
    """
    points_image retourne un dictionnaire contenant les positions des différents points de la main (index, pouce...)
    Il y a 21 points différents accessibles indexés de 0 à 20 (voir la documentation de mediapipe)

    Arguments
    ---------
    image: ouverte avec cv2.imread(path)
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

    # On charge le modèle
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils 
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.6)
    
    # On charge l'image
    name = 'image_test' # nom associé à l'image
    image_hight, image_width, _ = image.shape

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output and process it with MediaPipe Hands.
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    # On prépare l'image qui contiendra les annotations (les points sur la main)
    annotated_image = cv2.flip(image.copy(), 1)

    # On récupère les infos sur le résultat (nombre de mains récupérées, et précision)
    list_hands = results.multi_handedness

    # Si on n'a trouvé 0 main, on renvoie None
    if list_hands == None:
        print("Aucune main n'a été reconnue sur l'image")
        if display:
            cv2.imshow("image", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return None

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
        for hand_landmarks in results.multi_hand_landmarks: # Ici, results.multi_hand_landmarks est un liste de longueur 0 si on a trouvé 0 main, 1 si une main, et 2 si deux mains
            res[l[i]] = dict()
            for indice in range(21):
                # Chaque indice représente une partie de la main (voir la documentation, par exemple 0 = poignet)
                position = hand_landmarks.landmark[indice] # éventuellement multiplier par image_width pour les x, et height pour les y
                res[l[i]][indice] = position

            
            mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            i += 1

        if display:
            cv2.imshow("image", cv2.flip(annotated_image, 1))
            cv2.waitKey()
            cv2.destroyAllWindows()
            
        return res, cv2.flip(annotated_image, 1)

