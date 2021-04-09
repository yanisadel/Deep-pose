import cv2
import detection_tete.face_detection
from os import listdir
import detection_points
def affiche_image(image, name="image"):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def affiche_image_from_path(path, name="image"):
    image = cv2.imread(path)
    affiche_image(image, name)

def affichage_video(video):
    # Read until video is completed
    while(video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True: 
            # Display the resulting frame
            cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def affichage_video_from_path(path):
    video = cv2.VideoCapture(path)
    print(type(video))
    affichage_video(video)
    
def affichage_folder(s):
    for path in listdir(s):
        img=cv2.imread(s + "/" + path)
        _,annoted_image=detection_points.points_image(img,0.7,False)
        if type(annoted_image)!=type(None):
            image=detection_tete.face_detection.detect_face(annoted_image)
            if type(image)!=type(None):
                affiche_image(image)



<<<<<<< HEAD
=======
if __name__ == '__main__':
    #affichage_video_from_path("output.avi")
    affichage_folder('Data/Signes/1')
    
>>>>>>> eff5d249ad3c174aa0d653b42484d04145964dd0
