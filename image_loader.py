import gzip
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def load_images(path):
    """
    Charge les images depuis un fichier MNIST/EMNIST (.gz)

    Returns:
        numpy.ndarray: Images de forme (n_images, 28, 28)
    """
    with gzip.open(path, 'rb') as f:
        # Ignorer l'en-tête (16 premiers octets)
        f.read(16)

        # Lire les données sous forme de tableau numpy
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)

        # Reshape selon le format MNIST : (nombre_images, 28, 28)
        images = data.reshape(-1, 28, 28)

    return images  # PAS de expand_dims ici !


def load_labels(path):
    """
    Charge les labels depuis un fichier MNIST/EMNIST (.gz)

    Args:
        path (str): Chemin vers le fichier .gz des labels

    Returns:
        numpy.ndarray: Labels de forme (n_labels,)
    """
    with gzip.open(path, 'rb') as f:
        # Ignorer l'en-tête (8 premiers octets)
        f.read(8)

        # Lire les données
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels


def load_mapping(path):
    """
    Charge la correspondance label numérique -> caractère

    Args:
        path (str): Chemin vers le fichier de mapping

    Returns:
        dict: Dictionnaire {label: caractère}
    """
    mapping = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    label = int(parts[0])  # Label numérique (0-46)
                    char_code = int(parts[1])  # Code ASCII du caractère
                    mapping[label] = chr(char_code)  # Conversion en caractère
    except FileNotFoundError:
        print(f"⚠️  Fichier de mapping non trouvé : {path}")
        return None

    return mapping
