import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input

# === CONFIGURAÇÕES ===
MODEL_PATH = "./datasets/data/model.keras"           # Caminho para seu modelo salvo
CLASSES_PATH = "./datasets/CUB_200_2011/classes.txt"         # Arquivo de classes (opcional)
IMG_PATH = "./img2.png"        # Caminho da imagem para teste
IMG_SIZE = (224, 224)                # Tamanho que seu modelo espera

# === CARREGAR MODELO ===
if Path(MODEL_PATH).exists():
    model = load_model(MODEL_PATH)
    print("✅ Modelo carregado.")
else:
    raise FileNotFoundError(f"❌ Modelo não encontrado: {MODEL_PATH}")

# === CARREGAR CLASSES (opcional) ===
if Path(CLASSES_PATH).exists():
    classes_df = pd.read_csv(CLASSES_PATH, sep=" ", names=["class_id", "class_name"])
    CLASS_NAMES = classes_df.sort_values("class_id")["class_name"].tolist()
else:
    CLASS_NAMES = [f"Class_{i}" for i in range(model.output_shape[-1])]
    print("⚠️ Arquivo de classes ausente. Usando nomes genéricos.")

# === CARREGAR IMAGEM ===
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_batch = np.expand_dims(preprocess_input(img_array), axis=0)

# === PREDIÇÃO ===
predictions = model.predict(img_batch)[0]
pred_index = np.argmax(predictions)
confidence = float(predictions[pred_index])

predicted_class = {
    "class_id": int(pred_index),
    "class_name": CLASS_NAMES[pred_index],
    "confidence": round(confidence, 4)
}

# === RESULTADO ===
print("\n🎯 Resultado:")
print(predicted_class)

# === VISUALIZAÇÃO ===
plt.imshow(img)
plt.title(f"{predicted_class['class_name']} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()
