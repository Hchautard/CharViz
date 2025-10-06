import gzip
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

def load_images(path):
    """
    Charge les images depuis un fichier MNIST/EMNIST (.gz)

    Args:
        path (str): Chemin vers le fichier .gz des images

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

    return images


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
