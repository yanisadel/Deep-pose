from data import *
from points_dlib import *
from affichage import *

def display_vector_from_path(path,display=False):
    image=cv2.imread(path)
    display_vector_from_image(image,display)

def display_vector_from_image(image,display=False):
    heigth, width,_ = image.shape
    hand_points=points_image(image)
    head_points = points_face(image)
    if head_points!=None and hand_points!=None :
        for i in [0,5,9,13,17]:
            for j in [1,4,9,14,17,28,34,37,40,43,46,49,55]:
                xm,ym = int((1-hand_points[3*i])*width),int(hand_points[3*i+1]*heigth)
                xt,yt = int(head_points[j-1][0]*width),int(head_points[j-1][1]*heigth)
                cv2.line(image,(xm,ym),(xt,yt),(0,0,255),1)
    else:
        print('rien')
    if display==True:
        affiche_image(image)
    return(image)


if __name__ == '__main__':
    display_vector_from_path('data_train/niveaux/3/4-4.jpg',True)