# Data source:
# Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
# EMNIST: an extension of MNIST to handwritten letters.
# Retrieved from http://arxiv.org/abs/1702.05373

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from image_loader import load_images,load_labels

if __name__ == "__main__":
    images = load_images('data/gzip/emnist-balanced-train-images-idx3-ubyte.gz')
    labels = load_labels('data/gzip/emnist-balanced-train-labels-idx1-ubyte.gz')

    # Vérifier le chargement
    print(f"✅ {images.shape[0]} images chargées")
    print(f"✅ Taille de chaque image : {images.shape[1]}x{images.shape[2]} pixels")
    print(f"✅ {labels.shape[0]} labels chargés")

    # Afficher quelques exemples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
