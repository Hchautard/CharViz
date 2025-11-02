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
from model import create_cnn_model
from train import train_model, plot_training_history, save_model


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


def preprocess_data():
    """
    Charge et prétraite les données EMNIST

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("\n" + "=" * 60)
    print("📋 PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60 + "\n")

    preprocessor = DP(
        train_images_path='data/gzip/emnist-balanced-train-images-idx3-ubyte.gz',
        train_labels_path='data/gzip/emnist-balanced-train-labels-idx1-ubyte.gz',
        test_images_path='data/gzip/emnist-balanced-test-images-idx3-ubyte.gz',
        test_labels_path='data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz'
    )

    # ⚠️ ORDRE CRITIQUE : load → correct → normalize → reshape
    preprocessor.load_data() \
        .correct_emnist_orientation() \
        .normalize() \
        .reshape_for_cnn() \
        .get_data_info()

    X_train, y_train = preprocessor.get_train_data()
    X_test, y_test = preprocessor.get_test_data()

    # VÉRIFICATION DE SÉCURITÉ
    print("\n🔍 Vérification finale avant entraînement :")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_train min/max: {X_train.min():.3f} / {X_train.max():.3f}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_train min/max: {y_train.min()} / {y_train.max()}")

    # Assertions de sécurité
    assert len(X_train.shape) == 4, f"❌ Mauvaise forme : {X_train.shape}"
    assert X_train.shape[-1] == 1, f"❌ Mauvais nombre de canaux : {X_train.shape[-1]}"
    assert X_train.max() > 0, "❌ Images toutes noires !"

    print("✅ Toutes les vérifications passées !\n")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    images_raw = load_images('data/gzip/emnist-balanced-train-images-idx3-ubyte.gz')
    labels = load_labels('data/gzip/emnist-balanced-train-labels-idx1-ubyte.gz')
    mapping = load_mapping('data/gzip/emnist-balanced-mapping.txt')

    images_corrected = np.array([np.fliplr(np.rot90(np.rot90(np.rot90(img, k=1)))) for img in images_raw[:10]])

    print("\n🔍 Comparaison AVANT/APRÈS correction :")
    print("=" * 60)

    # Afficher AVANT correction
    print("\n❌ AVANT correction (images mal orientées) :")
    affichage(images_raw, labels, mapping, n_images=5)

    # Afficher APRÈS correction
    print("\n✅ APRÈS correction (images bien orientées) :")
    affichage(images_corrected, labels[:10], mapping, n_images=5)

    # Vérifier le chargement
    print(f"✅ {images_corrected.shape[0]} images chargées")
    print(f"✅ Taille de chaque image : {images_corrected.shape[1]}x{images_corrected.shape[2]} pixels")
    print(f"✅ {labels.shape[0]} labels chargés")

    if mapping:
        print(f"✅ {len(mapping)} classes chargées")
        print(f"\n📝 Exemple de mapping :")
        for i in range(min(10, len(mapping))):
            if i in mapping:
                print(f"   Label {i} → '{mapping[i]}'")

        # Afficher les images avec les vrais caractères
    affichage(images_corrected, labels, mapping)

    # Prétraiter les données pour CNN
    X_train, y_train, X_test, y_test = preprocess_data()

    # Créer le modèle CNN
    model, history = train_model(X_train, y_train, X_test, y_test,
                                epochs=10, batch_size=128)
    save_model(model, 'emnist_cnn_model.keras')
    plot_training_history(history)





