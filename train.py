import matplotlib.pyplot as plt
from model import create_cnn_model

def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    """
    Crée et entraîne le modèle CNN

    Args:
        X_train: Images d'entraînement
        y_train: Labels d'entraînement
        X_test: Images de test
        y_test: Labels de test
        epochs: Nombre d'époques
        batch_size: Taille des batchs

    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 60)
    print("CRÉATION ET ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60 + "\n")

    # Créer le modèle
    model = create_cnn_model(input_shape=(28, 28, 1), num_classes=47)

    # Afficher l'architecture
    print("Architecture du modèle :\n")
    model.summary()

    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement
    print(f"Début de l'entraînement ({epochs} epochs, batch_size={batch_size})...\n")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # # Évaluer sur le test set
    # print("\n Évaluation sur le test set...")
    # test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    #
    # print(f"\n Résultats finaux :")
    # print(f"   Loss (test): {test_loss:.4f}")
    # print(f"   Accuracy (test): {test_accuracy * 100:.2f}%")

    return model, history


def plot_training_history(history):
    """
    Affiche les courbes de loss et accuracy pendant l'entraînement

    Args:
        history: Historique retourné par model.fit()
    """
    print("\n" + "=" * 60)
    print("VISUALISATION DES COURBES D'APPRENTISSAGE")
    print("=" * 60 + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Courbe de Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Courbe d'Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, marker='o')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_title("Évolution de l'Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_model(model, filepath='emnist_cnn_model.keras'):
    """
    Sauvegarde le modèle entraîné

    Args:
        model: Modèle à sauvegarder
        filepath: Chemin du fichier de sauvegarde
    """
    print("\n" + "=" * 60)
    print("SAUVEGARDE DU MODÈLE")
    print("=" * 60 + "\n")

    model.save(filepath)
    print(f"Modèle sauvegardé : {filepath}\n")