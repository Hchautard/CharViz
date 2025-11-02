import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('TkAgg')

from image_loader import load_images, load_labels, load_mapping
from preprocessing import DataPreprocessor as DP


class EMNISTPredictor:
    """
    Classe pour faire des prédictions avec le modèle CNN entraîné
    """

    def __init__(self, model_path='emnist_cnn_model.keras',
                 mapping_path='data/gzip/emnist-balanced-mapping.txt'):
        """
        Initialise le prédicteur
        """
        print("\n" + "=" * 60)
        print("🔮 CHARGEMENT DU PRÉDICTEUR")
        print("=" * 60 + "\n")

        print(f"📂 Chargement du modèle : {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✅ Modèle chargé avec succès\n")

        print(f"📂 Chargement du mapping : {mapping_path}")
        self.mapping = load_mapping(mapping_path)
        print(f"✅ {len(self.mapping)} classes chargées\n")

    def preprocess_emnist_image(self, image):
        """
        Prétraite une image du DATASET EMNIST (avec correction d'orientation)

        Args:
            image: Image EMNIST (28, 28) ou (28, 28, 1)

        Returns:
            Image prétraitée (1, 28, 28, 1)
        """
        if len(image.shape) == 3 and image.shape[-1] == 1:
            img = image.squeeze()
        else:
            img = image

        # ✅ Correction d'orientation EMNIST (UNIQUEMENT pour dataset EMNIST)
        img = np.rot90(np.fliplr(img))

        # Normaliser si nécessaire
        if img.max() > 1.0:
            img = img.astype('float32') / 255.0

        # Reshape pour le modèle
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        return img

    def preprocess_custom_image(self, image):
        """
        Prétraite une IMAGE CUSTOM (SANS correction d'orientation EMNIST)

        Args:
            image: Image custom (28, 28)

        Returns:
            Image prétraitée (1, 28, 28, 1)
        """
        if len(image.shape) == 3:
            img = image.squeeze()
        else:
            img = image

        # ❌ PAS de correction d'orientation pour images custom !

        # Normaliser si nécessaire
        if img.max() > 1.0:
            img = img.astype('float32') / 255.0

        # Reshape pour le modèle
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        return img

    def predict_single(self, image, return_confidence=True, is_emnist=True):
        """
        Prédit la classe d'une seule image

        Args:
            image: Image (28, 28) ou (28, 28, 1)
            return_confidence: Si True, retourne aussi la confiance
            is_emnist: Si True, applique la correction EMNIST (dataset)
                      Si False, pas de correction (image custom)

        Returns:
            (classe_predite, caractere, confiance) ou (classe_predite, caractere)
        """
        # Choisir le bon preprocessing
        if is_emnist:
            img_processed = self.preprocess_emnist_image(image)
        else:
            img_processed = self.preprocess_custom_image(image)

        # Faire la prédiction
        predictions = self.model.predict(img_processed, verbose=0)

        # Obtenir la classe prédite et la confiance
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Obtenir le caractère correspondant
        predicted_char = self.mapping.get(predicted_class, '?')

        if return_confidence:
            return predicted_class, predicted_char, confidence
        else:
            return predicted_class, predicted_char

    def predict_batch(self, images, is_emnist=True):
        """
        Prédit sur un batch d'images
        """
        results = []
        for image in images:
            result = self.predict_single(image, is_emnist=is_emnist)
            results.append(result)
        return results

    def visualize_prediction(self, image, true_label=None, is_emnist=True):
        """
        Visualise une prédiction avec l'image et les probabilités

        Args:
            image: Image à prédire (28, 28)
            true_label: Label réel (optionnel)
            is_emnist: True si image du dataset EMNIST, False si custom
        """
        # Faire la prédiction avec le bon preprocessing
        predicted_class, predicted_char, confidence = self.predict_single(
            image, is_emnist=is_emnist
        )

        # Obtenir toutes les probabilités
        if is_emnist:
            img_processed = self.preprocess_emnist_image(image)
        else:
            img_processed = self.preprocess_custom_image(image)

        predictions = self.model.predict(img_processed, verbose=0)[0]

        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Afficher l'image
        ax1.imshow(np.rot90(np.fliplr(image.squeeze())), cmap='gray')

        # Titre avec prédiction
        title = f"Prédiction: '{predicted_char}' (classe {predicted_class})\n"
        title += f"Confiance: {confidence * 100:.2f}%"

        if true_label is not None:
            true_char = self.mapping.get(true_label, '?')
            is_correct = (predicted_class == true_label)
            color = 'green' if is_correct else 'red'
            title += f"\nVérité: '{true_char}' (classe {true_label})"
            ax1.set_title(title, fontsize=12, fontweight='bold', color=color)
        else:
            ax1.set_title(title, fontsize=12, fontweight='bold')

        ax1.axis('off')

        # Afficher les top 5 prédictions
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_probs = predictions[top5_indices]
        top5_chars = [self.mapping.get(i, '?') for i in top5_indices]

        colors = ['green' if i == predicted_class else 'skyblue' for i in top5_indices]

        ax2.barh(range(5), top5_probs, color=colors)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([f"'{c}' ({i})" for c, i in zip(top5_chars, top5_indices)])
        ax2.set_xlabel('Probabilité', fontsize=11)
        ax2.set_title('Top 5 prédictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1)

        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            ax2.text(prob + 0.02, i, f'{prob * 100:.1f}%',
                     va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    def test_on_test_set(self, n_samples=20, show_errors_only=False):
        """
        Teste le modèle sur des échantillons du test set
        """
        print("\n" + "=" * 60)
        print("TEST SUR LE TEST SET")
        print("=" * 60 + "\n")

        print("Chargement des données de test...")
        images = load_images('data/gzip/emnist-balanced-test-images-idx3-ubyte.gz')
        labels = load_labels('data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz')

        print("Prétraitement...")
        images_normalized = images.astype('float32') / 255.0

        indices = np.random.choice(len(images), n_samples, replace=False)

        print(f"\n{n_samples} échantillons sélectionnés aléatoirement\n")

        n_correct = 0
        n_displayed = 0

        for idx in indices:
            img = images_normalized[idx]
            true_label = labels[idx]

            predicted_class, predicted_char, confidence = self.predict_single(
                img, is_emnist=True
            )
            is_correct = (predicted_class == true_label)

            if is_correct:
                n_correct += 1

            if not show_errors_only or not is_correct:
                true_char = self.mapping.get(true_label, '?')
                status = "✅" if is_correct else "❌"

                print(f"{status} Image #{idx}: "
                      f"Prédit '{predicted_char}' ({confidence * 100:.1f}%) | "
                      f"Vérité '{true_char}'")

                n_displayed += 1

        accuracy = (n_correct / n_samples) * 100
        print(f"\nAccuracy sur cet échantillon : {n_correct}/{n_samples} = {accuracy:.2f}%")

        return images_normalized, labels, indices


def visualize_multiple_predictions(predictor, images, labels, indices, n_display=10):
    """
    Affiche plusieurs prédictions dans une grille

    Args:
        predictor: Instance de EMNISTPredictor
        images: Images du test set
        labels: Labels du test set
        indices: Indices à afficher
        n_display: Nombre d'images à afficher
    """
    n_cols = 5
    n_rows = (n_display + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten() if n_display > 1 else [axes]

    for i, idx in enumerate(indices[:n_display]):
        ax = axes[i]
        img = images[idx]
        true_label = labels[idx]

        # Prédiction
        predicted_class, predicted_char, confidence = predictor.predict_single(img)
        true_char = predictor.mapping.get(true_label, '?')
        is_correct = (predicted_class == true_label)

        # Afficher l'image
        ax.imshow(np.rot90(np.fliplr(img)), cmap='gray')

        # Titre avec couleur
        color = 'green' if is_correct else 'red'
        title = f"Prédit: '{predicted_char}' ({confidence * 100:.1f}%)\n"
        title += f"Vérité: '{true_char}'"
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')

    # Masquer les axes vides
    for i in range(n_display, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def load_custom_image(image_path):
    """
    Charge une image custom depuis un fichier

    Args:
        image_path: Chemin vers l'image

    Returns:
        Image prétraitée (28, 28) en niveaux de gris
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow n'est pas installé")
        print("Installez-le avec : pip install Pillow")
        return None

    try:
        # Charger l'image
        img = Image.open(image_path)

        # Convertir en niveaux de gris
        img = img.convert('L')

        # Redimensionner à 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Convertir en array numpy
        img_array = np.array(img)

        # IMPORTANT : Inverser les couleurs si nécessaire
        # EMNIST attend du blanc sur fond noir
        # Si ton image est noire sur fond blanc, décommenter :
        # img_array = 255 - img_array

        print(f"Image chargée : {image_path}")
        print(f"Taille : {img_array.shape}")
        print(f"Min/Max : {img_array.min()}/{img_array.max()}")

        return img_array

    except FileNotFoundError:
        print(f"Fichier non trouvé : {image_path}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None


def predict_custom_image_interactive(predictor):
    """
    Interface pour prédire sur une image custom
    """
    print("\n" + "=" * 60)
    print("PRÉDICTION SUR IMAGE CUSTOM")
    print("=" * 60)
    print("\nFormats acceptés : .png, .jpg, .jpeg, .bmp")
    print("Recommandations :")
    print("  - Image en niveaux de gris ou noir et blanc")
    print("  - Caractère bien centré")
    print("  - Fond blanc, caractère noir (ou inversé)")
    print("=" * 60 + "\n")

    image_path = input("Chemin de l'image : ").strip()
    image_path = image_path.strip('"').strip("'")

    img = load_custom_image(image_path)

    if img is None:
        return

    # Prévisualisation
    print("\n Prévisualisation de l'image...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Image originale', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(255 - img, cmap='gray')
    ax2.set_title('Image inversée', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    invert = input("\n Inverser les couleurs ? (o/n) : ").strip().lower()

    if invert == 'o':
        img = 255 - img

    print("\nPrédiction en cours...")
    predictor.visualize_prediction(img, is_emnist=False)


# ============================================
# Programme principal
# ============================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SYSTÈME DE PRÉDICTION DE CARACTÈRES EMNIST")
    print("=" * 60)

    # Prédicteur
    predictor = EMNISTPredictor(
        model_path='emnist_cnn_model.keras',
        mapping_path='data/gzip/emnist-balanced-mapping.txt'
    )

    # Menu interactif
    while True:
        print("\n" + "=" * 60)
        print("MENU")
        print("=" * 60)
        print("1. Tester sur des échantillons aléatoires du test set")
        print("2. Prédire sur une image spécifique (par index)")
        print("3. Charger et prédire sur une image custom")
        print("4. Quitter")
        print("=" * 60)

        choice = input("\n Choix (1-4) : ").strip()

        if choice == '1':
            n = int(input("Combien d'échantillons ? (défaut: 20) : ") or "20")
            images, labels, indices = predictor.test_on_test_set(n_samples=n)

            show = input("\nAfficher les images ? (o/n) : ").strip().lower()
            if show == 'o':

                # Correction des images pour l'affichage
                n_display = int(input("Combien d'images afficher ? (défaut: 20) : ") or "20")
                visualize_multiple_predictions(predictor, images, labels, indices, n_display)

        elif choice == '2':
            idx = int(input("Index de l'image (0-18799) : "))

            images = load_images('data/gzip/emnist-balanced-test-images-idx3-ubyte.gz')
            labels = load_labels('data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz')

            images_normalized = images.astype('float32') / 255.0

            if 0 <= idx < len(images):
                predictor.visualize_prediction(images_normalized[idx], labels[idx], is_emnist=True)
            else:
                print(f"Index invalide. Doit être entre 0 et {len(images) - 1}")

        elif choice == '3':
            predict_custom_image_interactive(predictor)

        elif choice == '4':
            break

        else:
            print("Choix invalide")