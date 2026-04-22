# MLOps
Projet réalisé au sein du cours Mise en oeuvre du big data.
Ce projet est un projet de classification d'image de visage sur 5 atributs: Barbe, Moustache, Couleur, Lunette, Taille Cheveux. Ce projet avait pour but de nous faire utiliser différentes technologies lié au MLOps. Nous avons donc pu désigner notre propre CNN et l'entrainer afin d'avoir les meilleurs résultats possible. 
Je me suis chargé du design du réseau et des premiers test, j'ai ensuite mis en place l'app MLflow afin de garder en mémoire les différentes versions du modèle et de suivre l'entrainement. 
J'ai également pu gérer les différentes tests (des lots d'images nous était données le long du projet ) afin de pouvoir réajuster les différents hyperparamètres du réseau pour tenter d'obtenir les meilleurs socres (accuracy, F1-score). Enfin j'ai mis en place Hyperopt afin de trouver les meilleurs hyperparamètres. 
Ce projet m'a donc permis de mettre en ouevre concrétements une grane partie des connaissances de mon masters 


## Problème de classification : 

- Barbe : oui/non
- Moustache : oui/non
- Couleur : 5 classes
- Lunette : 2 classes 
- Taille Cheveux : 3 classes

## Vecteur finale : 
    ### 11 dimensions :
    barbe 1
    moustache 1
    Couleur 5
    Lunette 1
    taille cheveux 3

## Start MLFLowUI

mlflow ui --backend-store-uri sqlite:///mlflow.db