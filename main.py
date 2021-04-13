from predictions import *
from data import *
import cv2
from detection_position import *
from os import listdir

def prediction_signe_image(knn, image, min_detection_confidence=0.5):
    """Prediction du signe depuis une image"""
    l = points_image(image, min_detection_confidence=min_detection_confidence)
    prediction = knn.predict([l])
    return prediction

def prediction_signe_image_from_path(knn, path):
    """Prediction du signe depuis le chemin d'une image"""
    image = cv2.imread(path)
    return prediction_signe_image(knn, image)
    
def prediction_position_image(knn, image, min_detection_confidence=0.5):
    """Prediction du signe depuis une image"""
    l = vector_to_face(image, min_detection_confidence=min_detection_confidence)
    prediction = knn.predict([l])
    return prediction

def prediction_position_image_from_path(knn, path):
    """Prediction du signe depuis le chemin d'une image"""
    image = cv2.imread(path)
    return prediction_position_image(knn, image)


knn_signes = retourne_knn_entraine('Data/signes.csv') # retourne le knn pour les signes entrainé
knn_position=retourne_knn_entraine('Data/face.csv')

if __name__ == '__main__':
    s='Data/Signes/1/'
    for path in listdir(s):
        print(prediction_signe_image_from_path(knn_signes, s + path))
        #print(prediction_position_image_from_path(knn_position, s + path))



