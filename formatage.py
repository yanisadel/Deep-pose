import cv2

def image_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flip_image(image):
    return cv2.flip(image.copy(), 1)


def generateur_decoupe_video(video):
    """
    generateur_decoupe_video renvoie un générateur qui contient toutes les frame d'une vidéo

    Arguments
    ---------
    video: video cv2
        chemin de la vidéo qu'on découpe
    
    Returns
    -------
    Generateur

    """
    
    success, image = video.read()
    count = 0
    while success:
        yield image
        success, image = video.read()
        count += 1
    
    print("Nombre de frames : ", count)


def generateur_decoupe_video_from_path(path='video_test.mp4'):
    video = cv2.VideoCapture('video_test.mp4')
    generateur_decoupe_video(video)



