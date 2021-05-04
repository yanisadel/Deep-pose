import cv2
import detection_tete.face_detection
from os import listdir
import detection_points
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
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
# A verifier
def make_video(images,outimg=None,fps=5,size=None,is_color=True,format='XVID'):
    """
    make_video prend en entrée une liste d'image et en fait une vidéo
    Arguments
    ---------
    images: list
        liste d'images
    
    Returns
    -------
    None (affiche vidéo)
 
    """
    fourcc=VideoWriter_fourcc(*format)
    vid=None
    for image in images:
        img=image
        if vid is None:
            if size is None:
                size=img.shape[1],img.shape[0]
            vid=VideoWriter('output.avi',fourcc,float(fps),size, is_color)
        if size[0]!=img.shape[1] and size[1]!=img.shape[0]:
            img=cv2.resize(img,size)
        vid.write(img)
    print(type(vid))
    #affichage_video(vid)
    vid.release()

if __name__ == '__main__':
    pass
    #affichage_video_from_path("output.avi")
