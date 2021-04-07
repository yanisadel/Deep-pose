import numpy as np
import cv2
#from .load_data import load
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
from skimage import io


def load(filename):
    return cv2.imread(filename) 

face_cascade = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('face_detection/haarcascade_eye.xml')
#img = cv2.imread('joe-biden1.jpg')
img = cv2.imread('data/tetris_blocks.jpg')


def HOG(image):
    """renvoie le gradient de l'image image selon x et y"""
    im = np.float32(image) / 255.0  # Calculate gradient
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bin_n = 16
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))
    
    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


def BGR_to_gray(image):
    """passe une image de l'espace RGB en Noir et Blanc"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def detect_face(image):
    '''detecte un visage sur une image en noir et blanc'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print("0 --> ", faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("1 --> ", cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2))
        print("2 --> ", roi_gray)
        print("3 --> ", roi_color)
        print("4 --> ", eyes)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        print("Coordonnées du rectangle autour du visage --> ", (x,y),(x+w,y+h))
    return (x, y), (x + w, y + h)

        
def HOG_Gray(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled

if __name__ == "__main__":
    img = cv2.imread('face_detection/3F2509DA0B68469280C4EC51E5661F6839CC295A.jpeg')
    #cv2.imshow('img',img)
    #N = BGR_to_gray(img)
    detect_face(img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # hist = HOG(img)

    """N = BGR_to_gray(img)
    image = HOG_Gray(N)
    detect_face(img)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(N)
    ax1.set_title('Input image')
    ax2.axis('off')
    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()"""
    
