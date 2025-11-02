import numpy as np
from image_loader import load_images, load_labels


class DataPreprocessor:

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
        print("Chargement des données...")

        self.X_train = load_images(self.train_images_path)
        self.y_train = load_labels(self.train_labels_path)
        self.X_test = load_images(self.test_images_path)
        self.y_test = load_labels(self.test_labels_path)

        print(f"Images d'entraînement : {self.X_train.shape}")  # DOIT afficher (n, 28, 28)
        print(f"Labels d'entraînement : {self.y_train.shape}")
        print(f"Images de test : {self.X_test.shape}")
        print(f"Labels de test : {self.y_test.shape}")

        return self

    def correct_emnist_orientation(self):
        """Corrige l'orientation des images EMNIST"""
        print("Correction de l'orientation...")

        # Corriger l'orientation
        corrected_train = []
        for img in self.X_train:
            corrected_train.append(np.rot90(np.fliplr(img)))

        corrected_test = []
        for img in self.X_test:
            corrected_test.append(np.rot90(np.fliplr(img)))

        self.X_train = np.array(corrected_train)
        self.X_test = np.array(corrected_test)

        return self

    def normalize(self):
        """Normalise les valeurs des pixels entre 0 et 1"""
        print("Normalisation des pixels...")

        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        print(f"Plage de valeurs : [{self.X_train.min():.3f}, {self.X_train.max():.3f}]")

        # VÉRIFICATION
        assert self.X_train.max() > 0, "Images toutes noires !"
        assert len(self.X_train.shape) == 3, f"Après normalisation : {self.X_train.shape}"

        return self

    def reshape_for_cnn(self):
        """Ajoute une dimension pour le canal"""
        print("Ajout de la dimension du canal...")

        # VÉRIFICATION : doit être 3D avant reshape
        assert len(self.X_train.shape) == 3, f"Avant reshape : {self.X_train.shape}"

        # Ajouter la dimension du canal
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

        print(f"Nouvelle forme : {self.X_train.shape}")

        # VÉRIFICATION FINALE
        assert len(self.X_train.shape) == 4, f"Après reshape : {self.X_train.shape}"
        assert self.X_train.shape[-1] == 1, f"Mauvais nombre de canaux : {self.X_train.shape[-1]}"

        return self

    def get_data_info(self):
        """Affiche des informations sur les données"""
        print("\n" + "=" * 50)
        print("INFORMATIONS SUR LES DONNÉES")
        print("=" * 50)

        unique_train, counts_train = np.unique(self.y_train, return_counts=True)

        print(f"\nNombre de classes : {len(unique_train)}")
        print(f"Taille train : {len(self.y_train)}")
        print(f"Taille test : {len(self.y_test)}")
        print(f"\nForme finale : {self.X_train.shape}")
        print(f"Min/Max pixels : {self.X_train.min():.3f} / {self.X_train.max():.3f}")

        print("=" * 50 + "\n")

        return self

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
