# Application de reconnaissance de forme
Ce projet est une application de reconnaissance de chiffres et de lettres manuscrites, qui utilise MNIST et un réseau de neurones convolutifs (CNN) pour la classification des images.

## Le type d'application
Il s'agit d'une application de reconnaissance de forme, spécifiquement conçue pour identifier des chiffres et des lettres manuscrites.

## Le type d'image traitée
L'application traite des images en niveaux de gris de 28x28 pixels, typiques des ensembles de données MNIST.

## Les données à utiliser
Les données utilisées proviennent de l'ensemble de données MNIST du NIST, qui contient des images de chiffres manuscrits (0-9) et peut être étendu pour inclure des lettres manuscrites (A-Z).
> Data source:
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
EMNIST: an extension of MNIST to handwritten letters.
Retrieved from http://arxiv.org/abs/1702.05373

Pour plus de précision, il y a plusieurs jeux de données disponibles:
- **EMNIST By Class**: 814,255 images de 62 classes (chiffres et lettres majuscules/minuscules).
- **EMNIST By Merge**: 814,255 images de 47 classes (chiffres et lettres majuscules/minuscules fusionnées).
- **EMNIST Balanced**: 131,600 images de 47 classes (chiffres et lettres majuscules/minuscules équilibrées).
- **EMNIST Letters**: 103,600 images de 37 classes (fusion majuscules/minuscules).
- **EMNIST Digits**: 280,000 images de 10 classes (chiffres uniquement).
- **EMNIST MNIST**: 70,000 images de 10 classes (chiffres uniquement, similaire à MNIST).

Le jeu de données que nous utiliserons pour l'application sera **EMNIST Balanced**, qui offre un bon compromis entre la diversité des classes et la taille du dataset.

## Les paramètres à extraire de chaque image
Les paramètres extraits de chaque image incluent les pixels individuels, qui sont utilisés comme entrées pour le modèle de réseau de neurones convolutifs.

## Les prétraitements nécessaires pour extraire les informations
Les prétraitements incluent la normalisation des pixels (mise à l'échelle entre 0 et 1) et le redimensionnement des images si nécessaire. Des techniques d'augmentation de données peuvent également être appliquées pour améliorer la robustesse du modèle.

## Le nombre de classes possibles
Le nombre de classes possibles est de 10 pour les chiffres (0-9) et peut être étendu à 36 pour inclure les lettres majuscules (A-Z).

## La méthode choisie pour la reconnaissance
La méthode choisie est un réseau de neurones convolutifs (CNN), qui est particulièrement efficace pour la reconnaissance d'images en raison de sa capacité à capturer les caractéristiques spatiales et les motifs dans les données visuelles.