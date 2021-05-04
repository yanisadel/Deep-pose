# importer les paquets nécessaires
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import mediapipe as mp
import affichage as aff
import detection_position as dp



# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()
# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def points_face(image,detector=detector,predictor=predictor):
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
	return(points)

def points_face_from_path(path,detector=detector,predictor=predictor):
	"""
	points_face_from_path prend en entrée le chemin d'une image et affiche sur la tête les points détectés par dlib sur la tête si il en détecte

    Arguments
    ---------
   
    path: chemin
        le chemin de l'image que l'on veut analyser

    detector : truc sombre

	predictor: truc sombre
	
    Returns
    -------
    None (affiche tête avec les points sur la tête)
      
	"""
	# charger l'image d'entrée, redimensionner et convertir en niveaux de gris
	image = cv2.imread(path)
	return(points_face(image,detector,predictor))

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
            points.append((x,y))
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

	#path='data_train/signes/1/WIN_20210407_15_34_27_Pro.jpg'
	#print(points_face_from_path(path,detector,predictor))
	vect_to_face_img(points_face2('data_train/niveaux/1/1-1.jpg'))