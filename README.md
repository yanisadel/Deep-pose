**Projet Deep pose**  

Ce projet vise à reconnaitre les différentes positions du langage parlé complété (LPC) par des algorithmes d'intelligence artificielle.  
Il a été réalisé par Guillaume Grasset-Gothon, Yanis Adel, Mehdi Dahmani, Robin Charleuf, Amine Ettafs, Fahd Et-Tahery

**0) Modules prérequis**  
Utilisez la commande suivante pour installer les modules : **pip install -r requirements.txt** (cela va installer les modules suivants : mediapipe, opencv-python, scikit-learn, numpy, pandas, matplotlib)  

**1) Lancer le programme**  
Il suffit de lancer le fichier main.py
Le fichier main.py contient la fonction main(path), qui prend en argument le chemin (path) d'une vidéo (sous format .mp4 par exemple).  
La fonction initialise le modèle et l'entraine, et une fois celui-ci entrainé, elle affiche la vidéo avec cette fois-ci les différents signes et positions de main.1  

**2) Les données d'entrainement**  
- Les données d'entrainement sont situées dans le dossier data_train.  
- Les images sont situées dans les dossiers "niveaux" et "signes", et à chaque fois dans la bonne catégorie (il y a 8 catégories de signes et 5 catégories de niveaux).  
- Le fichier data.py permet alors d'exploiter toutes ces images, et de remplir un les fichiers Excel présents dans data_train et qui contiennent toutes les features importantes des images, ainsi que leur label (le signe ou la position de la main).  
Ces fichiers Excel sont déjà remplis, mais il est possible de les mettre à jour en ajoutant des images dans la base de données (il suffit de mettre les images dans le bon dossier), et en utilisant les différentes fonctions fill_csv() du fichier data.py     

En ce qui concerne les données de test, elles sont situées dans data_test et fonctionnent de la même façon  

**3) Les différents signes et positions de main possibles**  
Il y a 8 signes de main différents (numérotés arbitrairement de 1 à 8), et 5 positions de main différentes par rapport à la tête (numérotés arbitrairement de 1 à 5)  

**4) Fonctionnement du code**  
Le but est de déduire le signe et la position de la main sur une image.  
Le programme utilise alors la bibliothèque MediaPipe afin de détecter les points de la main, et la bibliothèque dlib pour détecter les points de la tête.  
On considère alors toutes les coordonnées de ces points, et le programme utilise un algorithme des plus proches voisins pour prédire séparément le signe et la position de la main.  

**5) Fonctionnalités pour l'utilisation**  
Pour détecter les signes et positions de mains sur une vidéo, il y a deux possibilités :  
- télécharger la vidéo et utiliser le programme avec le chemin de la vidéo  
- ou utiliser le programme en direct avec la vidéo de la webcam  


