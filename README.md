# PP2-aorte-segmentation - Université de Montpellier
# Ibrahim Harcha, Ilona Lazrak, Samuel Mallet, Rosa Sabater Rojas

<!-- Add banner here -->
<img src="logo.jpeg" alt="logo" width="100"/> <img src="university.jpeg" alt="university" width="100"/>


## Abstract
Aorta segmentation on 2D images of CT-Scans is a challenging task due to the complex anatomy and variability of the aorta. We propose a method for aorta segmentation using convolutional neural networks (CNNs) and U-NET architectures. We use Python as the programming language and Tensorflow as the deep learning framework. Our method involves four stages: first, we preprocess and standardize our .nrrd CT-Scans; second  we use a CNN to detect the aorta region and crop the image accordingly; third, we use a U-NET to segment the aorta from the cropped image; finally, we reconstruct a 3D segmentation of the aorta based on our model. 

Our method is validated on a collection of 4100 2D images taken along the Z axis from 56 CT-Scan from various patients.

## Première étape du projet : Familiarisation avec les réseaux de néurones

### dataPreparation.py
    - Savoir comment ouvrir des fichier .nrrd et .nii,
    - Indiquer les paths où se trouvaient les images dans nos répertoires,
    - Créer des dossiers pour sauvegarder les images au format .png,
    - Plot les images et indiquer la couche et les coordonnées X, Y, Z.
    - Ppremier approche à la normalisation d'images, mais fait avec des valeurs -1000 et 2000 triés complètement au hasard...

### testing.py
    - Créer les fonctions pour afficher les images qu'il trouvera sur les paths indiqués,
    - Créer les fonctions generatrices des images pour parcourir les dossiers et pour les redimensionner avec les paramètres définis dans les globales,
    - Compter les images des dossiers et indiquer combien d'images nous utilisons pour le training et pour le test,
    - Créer le model U-NET avec un nombre de niveaux pour paramètre,
    - Exécuter le modèle et le sauvegarder avec un nom particulier qui défini le nombre d'epochs, le batch\_size, l'image height et l'image width utilisés,
    - Load le model s'il existe déjà afin de ne pas avoir besoin d'éxecuter le programme à nouveau,
    - Afficher les prédictions du model.


## Deuxième étape du projet : Google Collab et programme final

### U-NET_model_with_AVT_dataset.ipynb 

    - Des explications annotées tout au long du programme en expliquant ce que chaque cellule fait.
    - Le format du dataset et comment y accéder via Google Drive.
    - Les résultats au dessous des cellules une fois executées.
    -  Plots des CT-Scans
    - La normalisation des images à partir de la technique MinMax
    - La répartition des images pour le training et pour le test
    - L'implémentation de l'U-Net défini
    - L'enregistrement du modèle pour ne pas avoir besoin de relancer toutes les cellules à nouveau
    - Les graphiques avec les résultats

### predict_and_reconstruct.ipynb 

    - Charger le modèle déjà entraîné dans le notebook précédent
    - L'utiliser pour prédire un masque sur une image 2D d'un CT-Scan
    - Enregistrement des masques prédites par le modèle
    - Seuillage
    - Utiliser le modèle pour construire une segmentation 3D de l'aorte
