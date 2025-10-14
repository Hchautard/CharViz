import numpy as np
from image_loader import load_images, load_labels

class DataPreprocessor:
    """
    Classe pour gérer le prétraitement des données EMNIST
    """

    def __init__(self, train_images_path, train_labels_path,
                 test_images_path, test_labels_path):
        """
        Initialise le préprocesseur avec les chemins des fichiers

        Args:
            train_images_path (str): Chemin vers les images d'entraînement
            train_labels_path (str): Chemin vers les labels d'entraînement
            test_images_path (str): Chemin vers les images de test
            test_labels_path (str): Chemin vers les labels de test
        """
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

        # Données chargées (initialement None)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """
        Charge les données depuis les fichiers .gz
        """
        print("Chargement des données...")

        # Charger les images et labels
        self.X_train = load_images(self.train_images_path)
        self.y_train = load_labels(self.train_labels_path)
        self.X_test = load_images(self.test_images_path)
        self.y_test = load_labels(self.test_labels_path)

        print(f"Images d'entraînement : {self.X_train.shape}")
        print(f"Labels d'entraînement : {self.y_train.shape}")
        print(f"Images de test : {self.X_test.shape}")
        print(f"Labels de test : {self.y_test.shape}")

        return self

    def normalize(self):
        """
        Normalise les valeurs des pixels entre 0 et 1
        (passage de [0, 255] à [0.0, 1.0])
        """
        print("Normalisation...")

        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0

        print(f"Plage de valeurs : [{self.X_train.min():.2f}, {self.X_train.max():.2f}]")

        return self

    def reshape_for_cnn(self):
        """
        Ajoute une dimension pour le canal (nécessaire pour les CNN)
        Shape: (n, 28, 28) -> (n, 28, 28, 1)
        """
        print("Reshape for CNN...")

        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

        print(f"Nouvelle forme : {self.X_train.shape}")

        return self

    def get_train_data(self):
        """
        Retourne les données d'entraînement
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        Retourne les données de test
        """
        return self.X_test, self.y_test

def create_validation_split(X_train, y_train, validation_ratio=0.1):
    """
    Crée un ensemble de validation à partir des données d'entraînement

    Args:
        X_train (np.array): Images d'entraînement
        y_train (np.array): Labels d'entraînement
        validation_ratio (float): Proportion de données pour la validation

    Returns:
        tuple: (X_train_new, y_train_new, X_val, y_val)
    """
    n_samples = len(X_train)
    n_validation = int(n_samples * validation_ratio)

    # Mélanger les indices
    indices = np.random.permutation(n_samples)

    # Séparer les indices
    val_indices = indices[:n_validation]
    train_indices = indices[n_validation:]

    # Créer les nouveaux ensembles
    X_train_new = X_train[train_indices]
    y_train_new = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"✅ Split validation : {len(X_train_new)} train / {len(X_val)} validation")

    return X_train_new, y_train_new, X_val, y_val