import mediapipe as mp
import cv2

from image import *
from video import *

def points_video(path='video_test.mp4', min_detection_confidence=0.7, display=True):
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

    for frame in generateur_decoupe_video(path):
        points, image = points_image_from_image(frame, min_detection_confidence, display=False)
        cv2.imshow("frame", image)
        cv2.waitKey()
    cv2.destroyAllWindows()

points_video()
