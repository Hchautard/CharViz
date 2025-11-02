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

        Args:
            model_path: Chemin vers le modèle sauvegardé
            mapping_path: Chemin vers le fichier de mapping
        """
        print("\n" + "=" * 60)
        print("🔮 CHARGEMENT DU PRÉDICTEUR")
        print("=" * 60 + "\n")

        # Charger le modèle
        print(f"📂 Chargement du modèle : {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✅ Modèle chargé avec succès\n")

        # Charger le mapping
        print(f"📂 Chargement du mapping : {mapping_path}")
        self.mapping = load_mapping(mapping_path)
        print(f"✅ {len(self.mapping)} classes chargées\n")

    def preprocess_image(self, image):
        """
        Prétraite une image pour la prédiction

        Args:
            image: Image brute (28, 28) ou (28, 28, 1)

        Returns:
            Image prétraitée (1, 28, 28, 1)
        """
        # Si l'image est déjà (28, 28, 1), la garder telle quelle
        if len(image.shape) == 3 and image.shape[-1] == 1:
            img = image.squeeze()  # (28, 28, 1) → (28, 28)
        else:
            img = image

        # Correction d'orientation EMNIST
        img = np.rot90(np.fliplr(img))

        # Normaliser si nécessaire
        if img.max() > 1.0:
            img = img.astype('float32') / 255.0

        # Reshape pour le modèle : (28, 28) → (1, 28, 28, 1)
        img = np.expand_dims(img, axis=-1)  # (28, 28) → (28, 28, 1)
        img = np.expand_dims(img, axis=0)  # (28, 28, 1) → (1, 28, 28, 1)

        return img

    def predict_single(self, image, return_confidence=True):
        """
        Prédit la classe d'une seule image

        Args:
            image: Image (28, 28) ou (28, 28, 1)
            return_confidence: Si True, retourne aussi la confiance

        Returns:
            Si return_confidence: (classe_predite, caractere, confiance)
            Sinon: (classe_predite, caractere)
        """
        # Prétraiter l'image
        img_processed = self.preprocess_image(image)

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

    def predict_batch(self, images):
        """
        Prédit sur un batch d'images

        Args:
            images: Tableau d'images (n, 28, 28) ou (n, 28, 28, 1)

        Returns:
            Liste de tuples (classe, caractere, confiance)
        """
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        return results

    def visualize_prediction(self, image, true_label=None):
        """
        Visualise une prédiction avec l'image et les probabilités

        Args:
            image: Image à prédire (28, 28)
            true_label: Label réel (optionnel)
        """
        # Faire la prédiction
        predicted_class, predicted_char, confidence = self.predict_single(image)

        # Obtenir toutes les probabilités
        img_processed = self.preprocess_image(image)
        predictions = self.model.predict(img_processed, verbose=0)[0]

        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Afficher l'image
        ax1.imshow(image.squeeze(), cmap='gray')

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

        # Ajouter les valeurs sur les barres
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            ax2.text(prob + 0.02, i, f'{prob * 100:.1f}%',
                     va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    def test_on_test_set(self, n_samples=20, show_errors_only=False):
        """
        Teste le modèle sur des échantillons du test set

        Args:
            n_samples: Nombre d'échantillons à tester
            show_errors_only: Si True, affiche seulement les erreurs
        """
        print("\n" + "=" * 60)
        print("🧪 TEST SUR LE TEST SET")
        print("=" * 60 + "\n")

        # Charger les données de test
        print("📂 Chargement des données de test...")
        images = load_images('data/gzip/emnist-balanced-test-images-idx3-ubyte.gz')
        labels = load_labels('data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz')

        # Prétraiter
        print("🔧 Prétraitement...")
        images_corrected = np.array([np.rot90(np.fliplr(img)) for img in images])
        images_normalized = images_corrected.astype('float32') / 255.0

        # Sélectionner des échantillons aléatoires
        indices = np.random.choice(len(images), n_samples, replace=False)

        print(f"\n🎲 {n_samples} échantillons sélectionnés aléatoirement\n")

        # Prédire et afficher
        n_correct = 0
        n_displayed = 0

        for idx in indices:
            img = images_normalized[idx]
            true_label = labels[idx]

            predicted_class, predicted_char, confidence = self.predict_single(img)
            is_correct = (predicted_class == true_label)

            if is_correct:
                n_correct += 1

            # Afficher selon le mode
            if not show_errors_only or not is_correct:
                true_char = self.mapping.get(true_label, '?')
                status = "✅" if is_correct else "❌"

                print(f"{status} Image #{idx}: "
                      f"Prédit '{predicted_char}' ({confidence * 100:.1f}%) | "
                      f"Vérité '{true_char}'")

                n_displayed += 1

        accuracy = (n_correct / n_samples) * 100
        print(f"\n📊 Accuracy sur cet échantillon : {n_correct}/{n_samples} = {accuracy:.2f}%")

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
        ax.imshow(img, cmap='gray')

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
        print("❌ PIL/Pillow n'est pas installé")
        print("   Installez-le avec : pip install Pillow")
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

        print(f"✅ Image chargée : {image_path}")
        print(f"   Taille : {img_array.shape}")
        print(f"   Min/Max : {img_array.min()}/{img_array.max()}")

        return img_array

    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {image_path}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


def predict_custom_image_interactive(predictor):
    """
    Interface pour prédire sur une image custom
    """
    print("\n" + "=" * 60)
    print("📸 PRÉDICTION SUR IMAGE CUSTOM")
    print("=" * 60)
    print("\nFormats acceptés : .png, .jpg, .jpeg, .bmp")
    print("Recommandations :")
    print("  - Image en niveaux de gris ou noir et blanc")
    print("  - Caractère bien centré")
    print("  - Fond blanc, caractère noir (ou inversé)")
    print("=" * 60 + "\n")

    image_path = input("📂 Chemin de l'image : ").strip()

    # Enlever les guillemets si présents
    image_path = image_path.strip('"').strip("'")

    # Charger l'image
    img = load_custom_image(image_path)

    if img is None:
        return

    # Demander si on doit inverser les couleurs
    print("\n🎨 Prévisualisation de l'image...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Image originale', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(255 - img, cmap='gray')
    ax2.set_title('Image inversée', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    invert = input("\n🔄 Inverser les couleurs ? (o/n) : ").strip().lower()

    if invert == 'o':
        img = 255 - img
        print("✅ Couleurs inversées")

    # Faire la prédiction
    print("\n🔮 Prédiction en cours...")
    predictor.visualize_prediction(img)


# ============================================
# Programme principal
# ============================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🎯 SYSTÈME DE PRÉDICTION EMNIST")
    print("=" * 60)

    # Créer le prédicteur
    predictor = EMNISTPredictor(
        model_path='emnist_cnn_model.keras',
        mapping_path='data/gzip/emnist-balanced-mapping.txt'
    )

    # Menu interactif
    while True:
        print("\n" + "=" * 60)
        print("📋 MENU")
        print("=" * 60)
        print("1. Tester sur des échantillons aléatoires du test set")
        print("2. Visualiser des prédictions détaillées")
        print("3. Afficher seulement les erreurs")
        print("4. Prédire sur une image spécifique (par index)")
        print("5. 📸 Charger et prédire sur une image custom")  # ← NOUVEAU
        print("6. Quitter")
        print("=" * 60)

        choice = input("\n👉 Choix (1-6) : ").strip()

        if choice == '1':
            n = int(input("Combien d'échantillons ? (défaut: 20) : ") or "20")
            images, labels, indices = predictor.test_on_test_set(n_samples=n)

            show = input("\nAfficher les images ? (o/n) : ").strip().lower()
            if show == 'o':
                n_display = int(input("Combien d'images afficher ? (défaut: 10) : ") or "10")
                visualize_multiple_predictions(predictor, images, labels, indices, n_display)

        elif choice == '2':
            n = int(input("Combien de prédictions détaillées ? (défaut: 5) : ") or "5")

            images = load_images('data/gzip/emnist-balanced-test-images-idx3-ubyte.gz')
            labels = load_labels('data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz')

            images_corrected = np.array([np.rot90(np.fliplr(img)) for img in images])
            images_normalized = images_corrected.astype('float32') / 255.0

            indices = np.random.choice(len(images), n, replace=False)

            for idx in indices:
                predictor.visualize_prediction(images_normalized[idx], labels[idx])

        elif choice == '3':
            n = int(input("Combien d'échantillons tester ? (défaut: 100) : ") or "100")
            images, labels, indices = predictor.test_on_test_set(n_samples=n, show_errors_only=True)

        elif choice == '4':
            idx = int(input("Index de l'image (0-18799) : "))

            images = load_images('data/gzip/emnist-balanced-test-images-idx3-ubyte.gz')
            labels = load_labels('data/gzip/emnist-balanced-test-labels-idx1-ubyte.gz')

            images_corrected = np.array([np.rot90(np.fliplr(img)) for img in images])
            images_normalized = images_corrected.astype('float32') / 255.0

            if 0 <= idx < len(images):
                predictor.visualize_prediction(images_normalized[idx], labels[idx])
            else:
                print(f"❌ Index invalide. Doit être entre 0 et {len(images) - 1}")

        elif choice == '5':  # ← NOUVEAU
            predict_custom_image_interactive(predictor)

        elif choice == '6':
            print("\n👋 Au revoir !")
            break

        else:
            print("❌ Choix invalide")