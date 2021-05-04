import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils


def face_cam(min_detection_confidence=0.7):
   
      
  """
  face_cam allume la webcam et affiche sur la tête les points détectés par mediapipe sur la tête si il en détecte

  Arguments
  ---------
  
  min_detection_confidence: float
      le degré de confiance que l'on veut quant à la précision de l'analyse 
  
  Returns
  -------
  None (affiche webcam)
      
  """
  cap = cv2.VideoCapture(0)
  with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          Nose=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP)
          mp_drawing.draw_detection(image, detection)
        print (Nose)  

      cv2.imshow('MediaPipe Face Detection', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    
  cap.release()
  cv2.destroyAllWindows()

def face_img(image , min_detection_confidence=0.7):
  """
  face_img prend en entrée une image et retourne un dictionnaire contenant les points de la tête détectés par mediapipe l={'mouth':(x,y),'reye':,'leye','nose'}

  Arguments
  ---------
  image: image cv2
      image qu'on analyse
  
  min_detection_confidence: float
      le degré de confiance que l'on veut quant à la précision de l'analyse 
  
  Returns
  -------
  dict
      Retourne un dict du style :
          l={'mouth':(x,y),'reye':,'leye','nose'}

      qui représente les points de la tête si il y en a une
    
  """
  l={}
  with  mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    if type(image)!=None:
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      if results.detections!= None:
        annotated_image = image.copy()
        for detection in results.detections:
          l['mouth']=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.MOUTH_CENTER)
          l['reye']=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.RIGHT_EYE)
          l['leye']=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.LEFT_EYE)
          l['nose']=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP)
          return(l)
        #mp_drawing.draw_detection(annotated_image, detection)
        #cv2.imshow("image",annotated_image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    else : 
      return ( "error")

def face_img_from_path(path , min_detection_confidence=0.7):
  """
  face_img_from path prend en entrée le chemin d'une image et retourne un dictionnaire contenant les points de la tête détectés par mediapipe l={'mouth':(x,y),'reye':,'leye','nose'}

  Arguments
  ---------
  path: str
      chemin de l'image qu'on analyse
  
  min_detection_confidence: float
      le degré de confiance que l'on veut quant à la précision de l'analyse 
  
  Returns
  -------
  dict
      Retourne un dict du style :
          l={'mouth':(x,y),'reye':,'leye','nose'}

      qui représente les points de la tête si il y en a une
    
  """
  image=cv2.imread(path)
  return(face_img(image))

def face_vid(vidcap ,min_detection_confidence=0.7):
  """
  face_vid prend en entrée une vidéo et affiche sur la tête les points détectés par mediapipe sur la tête si il en détecte

  Arguments
  ---------
  
  min_detection_confidence: float
      le degré de confiance que l'on veut quant à la précision de l'analyse 
  
  Returns
  -------
  None (affiche vidéo)
    
  """
  success,image = vidcap.read()
  
  
  with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while success:
      success,image = vidcap.read()
      if not success:
        print("Ignoring empty camera frame.")
        break
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          Nose=mp.solutions.face_detection.get_key_point(detection, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP)
          mp_drawing.draw_detection(image, detection)
          print ("Nose:"+str (Nose) ) 

      cv2.imshow('MediaPipe Face Detection', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    
  cv2.destroyAllWindows()

def face_vid_from_path(path ,min_detection_confidence=0.7):
  """
  face_vid_from_path prend en entrée le chemin d'une vidéo et affiche sur la tête les points détectés par mediapipe sur la tête si il en détecte

  Arguments
  ---------
  
  min_detection_confidence: float
      le degré de confiance que l'on veut quant à la précision de l'analyse 
  
  Returns
  -------
  None (affiche vidéo)
    
  """
  vidcap=vidcap = cv2.VideoCapture(path)
  return face_vid(vidcap)

if __name__ == '__main__':
    #print(face_img_from_path('dataset/LPC/Capture2.PNG'))
    print(face_cam())
    