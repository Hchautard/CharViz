import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(input_shape=(28, 28, 1), num_classes=47):
    """
    Crée un CNN pour reconnaître les caractères EMNIST

    Args:
        input_shape: Forme de l'image (hauteur, largeur, canaux)
        num_classes: Nombre de classes à prédire

    Returns:
        model: Modèle CNN compilé
    """
    model = keras.Sequential([
        # Détection des formes de base
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Détection de motifs plus complexes
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Caractéristiques de haut niveau
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),

        # Aplatissement : passage de 2D à 1D
        layers.Flatten(),

        # Couches de classification
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Évite l'overfitting

        # Sortie : probabilité pour chaque classe
        layers.Dense(num_classes, activation='softmax')
    ])

    return model