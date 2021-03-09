<<<<<<< HEAD
import cv2

def generateur_decoupe_video(path='video_test.mp4'):
    """
    generateur_decoupe_video renvoie un générateur qui contient toutes les frame d'une vidéo

    Arguments
    ---------
    path: str
        chemin de la vidéo qu'on découpe
    
    Returns
    -------
    Generateur

    """
    
    video = cv2.VideoCapture('video_test.mp4')

    success, image = video.read()
    count = 0
    while success:
        yield image
        success, image = video.read()
        count += 1
    
    print("Nombre de frames : ", count)


def read_video_from_path(path='video_test.mp4'):
    video = cv2.VideoCapture(path)

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


def read_video_from_video(video):
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

=======
import cv2

def generateur_decoupe_video(path='video_test.mp4'):
    """
    generateur_decoupe_video renvoie un générateur qui contient toutes les frame d'une vidéo

    Arguments
    ---------
    path: str
        chemin de la vidéo qu'on découpe
    
    Returns
    -------
    Generateur

    """
    
    video = cv2.VideoCapture('video_test.mp4')

    success, image = video.read()
    count = 0
    while success:
        yield image
        success, image = video.read()
        count += 1
    
    print("Nombre de frames : ", count)


def read_video_from_path(path='video_test.mp4'):
    video = cv2.VideoCapture(path)

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


def read_video_from_video(video):
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

>>>>>>> master
