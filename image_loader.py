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
        f.read(16)
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        images = data.reshape(-1, 28, 28)  # (n, 28, 28) - 3D
    return images


def load_labels(path):
    """
    Charge les labels depuis un fichier MNIST/EMNIST (.gz)

    Returns:
        numpy.ndarray: Labels de forme (n_labels,)
    """
    with gzip.open(path, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


def load_mapping(path='data/gzip/emnist-balanced-mapping.txt'):
    """
    Charge la correspondance label numérique -> caractère

    Returns:
        dict: {label: caractère}
    """
    mapping = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    label = int(parts[0])
                    char_code = int(parts[1])
                    mapping[label] = chr(char_code)
    except FileNotFoundError:
        print(f"Fichier de mapping non trouvé : {path}")
        return None
    return mapping
