import numpy as np
from image_loader import load_images, load_labels


class DataPreprocessor:
    """
    Classe pour gérer le prétraitement des données EMNIST
    """

    def __init__(self, train_images_path, train_labels_path,
                 test_images_path, test_labels_path):
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Charge les données depuis les fichiers .gz"""
        print("📂 Chargement des données...")

        self.X_train = load_images(self.train_images_path)
        self.y_train = load_labels(self.train_labels_path)
        self.X_test = load_images(self.test_images_path)
        self.y_test = load_labels(self.test_labels_path)

        print(f"✅ Images d'entraînement : {self.X_train.shape}")
        print(f"✅ Labels d'entraînement : {self.y_train.shape}")
        print(f"✅ Images de test : {self.X_test.shape}")
        print(f"✅ Labels de test : {self.y_test.shape}")

        return self

    def correct_emnist_orientation(self):
        """
        Corrige l'orientation des images EMNIST
        IMPORTANT : À faire AVANT normalize et reshape
        """
        print("🔧 Correction de l'orientation EMNIST...")

        # Vérifier qu'on a bien des images 2D (pas encore reshape)
        if len(self.X_train.shape) != 3:
            print(f"⚠️ Forme inattendue : {self.X_train.shape}")
            return self

        # Corriger l'orientation pour chaque image
        corrected_train = []
        for img in self.X_train:
            corrected_train.append(np.rot90(np.fliplr(img)))

        corrected_test = []
        for img in self.X_test:
            corrected_test.append(np.rot90(np.fliplr(img)))

        self.X_train = np.array(corrected_train)
        self.X_test = np.array(corrected_test)

        print(f"✅ Orientation corrigée : {self.X_train.shape}")

        return self

    def normalize(self):
        """
        Normalise les valeurs des pixels entre 0 et 1
        IMPORTANT : À faire APRÈS correction d'orientation, AVANT reshape
        """
        print("🔧 Normalisation des pixels...")

        # Convertir en float32 et diviser par 255
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        print(f"✅ Plage de valeurs : [{self.X_train.min():.3f}, {self.X_train.max():.3f}]")

        # Vérification de sécurité
        if self.X_train.max() == 0.0:
            print("⚠️ ATTENTION : Toutes les images sont noires !")
            print("   → Vérifier le chargement des données")

        return self

    def reshape_for_cnn(self):
        """
        Ajoute une dimension pour le canal (nécessaire pour les CNN)
        IMPORTANT : À faire EN DERNIER
        """
        print("🔧 Ajout de la dimension du canal...")

        # Vérifier qu'on a bien 3 dimensions avant reshape
        if len(self.X_train.shape) == 3:
            self.X_train = np.expand_dims(self.X_train, axis=-1)
            self.X_test = np.expand_dims(self.X_test, axis=-1)
            print(f"✅ Nouvelle forme : {self.X_train.shape}")
        else:
            print(f"⚠️ Forme déjà modifiée : {self.X_train.shape}")

        return self

    def get_data_info(self):
        """Affiche des informations sur les données"""
        print("\n" + "=" * 50)
        print("📊 INFORMATIONS SUR LES DONNÉES")
        print("=" * 50)

        unique_train, counts_train = np.unique(self.y_train, return_counts=True)

        print(f"\n🔢 Nombre de classes : {len(unique_train)}")
        print(f"📈 Taille du dataset d'entraînement : {len(self.y_train)}")
        print(f"📉 Taille du dataset de test : {len(self.y_test)}")

        print(f"\n📊 Distribution des classes (train) :")
        print(f"   - Min : {counts_train.min()} exemples")
        print(f"   - Max : {counts_train.max()} exemples")
        print(f"   - Moyenne : {counts_train.mean():.1f} exemples")

        print(f"\n🖼️  Statistiques des pixels :")
        print(f"   - Forme : {self.X_train.shape}")
        print(f"   - Type : {self.X_train.dtype}")
        print(f"   - Min : {self.X_train.min():.3f}")
        print(f"   - Max : {self.X_train.max():.3f}")
        print(f"   - Moyenne : {self.X_train.mean():.3f}")

        print("=" * 50 + "\n")

        return self

    def get_train_data(self):
        """Retourne les données d'entraînement"""
        return self.X_train, self.y_train

    def get_test_data(self):
        """Retourne les données de test"""
        return self.X_test, self.y_test