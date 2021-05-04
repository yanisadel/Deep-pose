# importer les paquets nécessaires
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2



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
	print(height,width)
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
			points.append((1-x/width,1-y/height))
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

if __name__ == '__main__':

	path='data_train/signes/1/WIN_20210407_15_34_27_Pro.jpg'
	print(points_face_from_path(path,detector,predictor))