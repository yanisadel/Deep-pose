from utils import *
from signes import *
from data import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pylab

test_size_knn = 0.1
path_signes ='Data\signes.csv'
x, y = read_csv(path_signes)

# Séparation des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_knn, random_state=0)

# Entrainement du knn
knn = initialize_knn(x_train, y_train, n_neighbors=3)
train_knn(knn,x_train,y_train)

def bloc_image(L): 
    # L est une liste d'images(chemins(path) vers ces images)                
    n = len(L)
    H = []
    G = [0 for i in range(1,9)]
    F = []
    for i in range(n):
        xtest = points_image_from_path(path=L[i], min_detection_confidence=0.7, display=True)
        a = knn.predict([xtest])
        H.append(a[0])
    for j in range(n):
        G[H[j]] = G[H[j]]+1
    M = max(G)
    for j in range(n):
        if G[H[j]] == M:
            F.append(H[j])

    return [F[0],(M*100)/n]



def list_bloc_image(L):
    # L est une liste de blocs d'images (chaque bloc d'images contient des chemins(path) vers ces dernières)
    H = []
    for i in range(len(L)):
        H.append(bloc_image(L[i]))
    return H

def color_signe(i):
    if i==1:
        return 'blue'
    elif i==2:
        return 'red'
    elif i==3:
        return 'black'
    elif i==4:
        return 'green'
    elif i==5:
        return 'yellow'
    elif i==6:
        return 'orange'
    elif i==7:
        return 'purple'
    else:
        return 'brown'

def histogramme_video(V):
    # V est une liste de blocs d'images
    """
    H = list_bloc_image(V)
    L = [H[i][1] for i in range(len(H))]
    plt.figure(figsize=(12,10))
    plt.hist(L, range = (0, len(V)+1), bins =[i for i in range(len(V)+1)], color = 'yellow',
            edgecolor = 'red')
    plt.title("taux de fiabilité de détection des signes au cours du temps")
    plt.xlabel("temps(s)")
    plt.ylabel("taux de fiabilité(%)")
    plt.show()
    """
    H = list_bloc_image(V)
    fig = plt.figure()
    x = [i for i in range(len(V))]
    height = [H[i][1] for i in range(len(H))]
    width = 0.5
    BarName = [i for i in range(1,len(V)+1)]

    bars=plt.bar(x, height, width, color='blue' )
    for i in range(len(bars)):
        c = color_signe(H[i][0])
        bars[i].set_facecolor(c)

    #plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(0,len(V)-1)
    plt.ylim(0,100)
    plt.grid()
    plt.xlabel('numéro de bloc')
    plt.ylabel('taux de fiabilité(%)')
    plt.title('Diagramme en Batons !')

    pylab.xticks(x, BarName, rotation=40)

    plt.show()  
#print(knn.predict(xtest))
    
"""
def predictions_knn_1(knn, xtest):
    xtest = pd.DataFrame(columns=labels_csv()[1:]+["pos20z"], data=[xtest])
    return predictions_knn(knn, xtest) """ 
#V=[['Data\Signes\3\WIN_20210407_15_34_45_Pro.jpg']]
#histogramme_video(V)