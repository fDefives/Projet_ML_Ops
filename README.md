# MLOps
## Projet – Classification d’images de visages (MLOps)
Projet réalisé en équipe dans le cadre de l’unité d’enseignement Mise en œuvre du Big Data.
## Objectif :
Développement d’un modèle de classification d’images capable de prédire plusieurs attributs faciaux (barbe, moustache, couleur des cheveux, lunettes, longueur des cheveux) à partir d’un jeu de données de 20 000 images.
Moyens mis en oeuvre : 
La conception et l’entraînement d’un réseau de neurones convolutif (CNN)
Le suivi des expérimentations et versioning des modèles avec MLflow
L’optimisation des performances via recherche d’hyperparamètres avec Hyperopt

## Moyens mis en œuvre :
Conception du modèle avec la création du design de l’architecture du réseau de neurones convolutif (CNN) et la réalisation des premiers tests.
Mise en place d’un suivi d’expérimentation avec le déploiement de MLflow (traçabilité, comparaison des modèles, reproductibilité).
Phase de test et d’évaluation avec la gestion des jeux de données fournis pour l’évaluation des performances.
Optimisation des performances via l’ajustement des hyperparamètres et l’implémentation d’Hyperopt pour automatiser la recherche des meilleures configurations.

## Contribution personnelle :
J’ai participé à la conception de l’architecture du CNN et à la réalisation des premiers tests. J’ai également mis en place le suivi d’expérimentation avec MLflow, contribué aux phases de test et d’évaluation en gérant les différents jeux de données, et optimisé les performances du modèle via le tuning des hyperparamètres avec Hyperopt.
## Compétences acquises :
Renforcement des compétences en deep learning (CNN)
Mise en œuvre de pratiques MLOps (suivi, versioning, reproductibilité)


## Problème de classification : 

- Barbe : oui/non
- Moustache : oui/non
- Couleur : 5 classes
- Lunette : 2 classes 
- Longueur Cheveux : 3 classes

## Vecteur finale : 
    ### 11 dimensions :
    barbe 1
    moustache 1
    Couleur 5
    Lunette 1
    taille cheveux 3

## Start MLFLowUI

mlflow ui --backend-store-uri sqlite:///mlflow.db