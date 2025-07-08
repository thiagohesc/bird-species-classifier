import os
import traceback
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from pathlib import Path
from utils.constants import EnvVars, PathFiles


class Classifier:
    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (224, 224),
        batch_size: int = 32,
        num_classes: int = 200,
        model_path: str = "model.keras",
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_path = model_path
        self.history = None
        self.model = None

        self._load_datasets()
        self._build_model()

    def _load_datasets(self):
        self.train_ds = image_dataset_from_directory(
            self.data_dir / "train",
            image_size=self.img_size,
            batch_size=self.batch_size,
        )
        self.val_ds = image_dataset_from_directory(
            self.data_dir / "test",
            image_size=self.img_size,
            batch_size=self.batch_size,
        )

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = (
            self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def _build_model(self):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
                layers.RandomContrast(0.2),  # novo
                layers.RandomBrightness(0.2),  # novo
            ]
        )

        conv_base = keras.applications.EfficientNetB7(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        conv_base.trainable = False

        inputs = keras.Input(shape=(*self.img_size, 3))
        x = data_augmentation(inputs)
        x = keras.applications.efficientnet.preprocess_input(x)
        x = conv_base(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, epochs: int = 50, epochs_fine: int = 20):
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_path, save_best_only=True, monitor="val_loss"
        )
        early_stop_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        print("Iniciando pré-treinamento com EfficientNetB0 congelado...")
        history1 = self.model.fit(
            self.train_ds,
            epochs=epochs,
            validation_data=self.val_ds,
            callbacks=[checkpoint_cb, early_stop_cb],
        )

        print("Descongelando camadas da EfficientNetB0 para fine-tuning...")
        conv_base = self.model.get_layer(index=3)
        conv_base.trainable = True
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history2 = self.model.fit(
            self.train_ds,
            epochs=epochs_fine,
            validation_data=self.val_ds,
            callbacks=[checkpoint_cb],
        )

        # Combina os históricos
        self.history = keras.callbacks.History()
        self.history.history = {
            key: history1.history.get(key, []) + history2.history.get(key, [])
            for key in set(history1.history) | set(history2.history)
        }

    def evaluate(self):
        self.model = keras.models.load_model(self.model_path)
        test_loss, test_acc = self.model.evaluate(self.val_ds)
        print(f"\nTest accuracy: {test_acc:.4f}")
        return test_loss, test_acc

    def plot_history(self, save_path):
        if not self.history:
            print("Modelo ainda não foi treinado.")
            return

        acc = self.history.history.get("accuracy", [])
        val_acc = self.history.history.get("val_accuracy", [])
        loss = self.history.history.get("loss", [])
        val_loss = self.history.history.get("val_loss", [])
        epochs = range(1, len(acc) + 1)
        xticks_range = list(range(1, len(acc) + 1, 5))

        print(f"Salvando gráfico em: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.style.use("ggplot")
        plt.figure(figsize=(14, 6))

        # Gráfico de Acurácia
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label="Treinamento", linewidth=2)
        plt.plot(epochs, val_acc, linestyle="--", label="Validação", linewidth=2)
        plt.title("Acurácia por Época", fontsize=14)
        plt.xlabel("Época", fontsize=12)
        plt.ylabel("Acurácia", fontsize=12)
        plt.xticks(xticks_range)
        plt.legend()
        plt.grid(True)

        # Acurácia máxima
        if val_acc:
            max_val_epoch = val_acc.index(max(val_acc)) + 1
            plt.annotate(
                f"{max(val_acc):.2f}",
                xy=(max_val_epoch, max(val_acc)),
                xytext=(max_val_epoch, max(val_acc) + 0.02),
                arrowprops=dict(arrowstyle="->", color="black"),
            )

        # Gráfico de Perda
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label="Treinamento", linewidth=2)
        plt.plot(epochs, val_loss, linestyle="--", label="Validação", linewidth=2)
        plt.title("Perda (Loss) por Época", fontsize=14)
        plt.xlabel("Época", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.xticks(xticks_range)
        plt.legend()
        plt.grid(True)

        # Perda mínima
        if val_loss:
            min_val_epoch = val_loss.index(min(val_loss)) + 1
            plt.annotate(
                f"{min(val_loss):.2f}",
                xy=(min_val_epoch, min(val_loss)),
                xytext=(min_val_epoch, min(val_loss) + 0.2),
                arrowprops=dict(arrowstyle="->", color="black"),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def train_model(train: bool = False):
    if not train:
        print("Treinamento desativado por configuração.")
        return

    try:
        print("Iniciando treinamento do classificador...")
        classifier = Classifier(
            data_dir=PathFiles.DATA,
            img_size=(224, 224),
            batch_size=8,
            num_classes=200,
            model_path=PathFiles.MODEL,
        )

        classifier.train(epochs=50, epochs_fine=10)
        classifier.evaluate()
        classifier.plot_history(save_path=PathFiles.RESULTS)

        print("Treinamento e avaliação concluídos com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro ao gerar o modelo: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    train_model(train=EnvVars.TRAIN)
