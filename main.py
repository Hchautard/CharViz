# Data source:
# Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
# EMNIST: an extension of MNIST to handwritten letters.
# Retrieved from http://arxiv.org/abs/1702.05373

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from preprocessing import DataPreprocessor as DP
matplotlib.use('TkAgg')

from image_loader import load_images,load_labels, load_mapping


def affichage(images, labels, mapping=None, n_images=10):
    """
    Affiche des exemples d'images avec leurs labels

    Args:
        images (np.array): Images à afficher
        labels (np.array): Labels correspondants
        mapping (dict): Dictionnaire de correspondance label -> caractère
        n_images (int): Nombre d'images à afficher
    """
    # Calculer le layout (2 lignes)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))

    # Aplatir axes si nécessaire
    if n_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n_images):
        ax = axes[i]

        # Afficher l'image
        ax.imshow(images[i], cmap='gray')

        # Titre avec le vrai caractère ou juste le label
        if mapping and labels[i] in mapping:
            char = mapping[labels[i]]
            ax.set_title(f"Label: {labels[i]} → '{char}'", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"Label: {labels[i]}", fontsize=12)

        ax.axis('off')

    # Masquer les axes vides
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images = load_images('data/gzip/emnist-balanced-train-images-idx3-ubyte.gz')
    labels = load_labels('data/gzip/emnist-balanced-train-labels-idx1-ubyte.gz')
    mapping = load_mapping('data/gzip/emnist-balanced-mapping.txt')

    # Vérifier le chargement
    print(f"✅ {images.shape[0]} images chargées")
    print(f"✅ Taille de chaque image : {images.shape[1]}x{images.shape[2]} pixels")
    print(f"✅ {labels.shape[0]} labels chargés")

    if mapping:
        print(f"✅ {len(mapping)} classes chargées")
        print(f"\n📝 Exemple de mapping :")
        for i in range(min(10, len(mapping))):
            if i in mapping:
                print(f"   Label {i} → '{mapping[i]}'")

        # Afficher les images avec les vrais caractères
    affichage(images, labels, mapping)


