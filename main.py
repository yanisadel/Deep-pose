from predictions import *
from data import *
import cv2

def prediction_signe_image(knn, image, min_detection_confidence=0.5):
    """Prediction du signe depuis une image"""
    l = points_image(image, min_detection_confidence=min_detection_confidence)
    prediction = knn.predict([l])
    return prediction

def prediction_signe_image_from_path(knn, path):
    """Prediction du signe depuis le chemin d'une image"""
    image = cv2.imread(path)
    return prediction_signe_image(knn, image)


knn_signes = retourne_knn_signes_entraine() # retourne le knn pour les signes entrainé
predictions = prediction_signe_image_from_path(knn, 'Data/Signes/1/IMG_20210408_190721.jpg')



