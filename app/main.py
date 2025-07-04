import numpy as np
import io
import pandas as pd

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.constants import EnvVars, Urls, PathFiles
from utils.schemas import (
    DownloadRequest,
    ClearRequest,
    PreparerRequest,
)
from services.dataset_downloader import DatasetDownloader
from services.dataset_preparer import DatasetPreparer
from utils.utils import clear_directory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.dattaflow.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


classes_path = PathFiles.CUB + "/classes.txt"

classes_df = pd.read_csv(classes_path, sep=" ", names=["class_id", "class_name"])
CLASS_NAMES = classes_df.sort_values("class_id")["class_name"].tolist()

model = load_model(PathFiles.MODEL)
img_size = (224, 224)


@app.post("/download")
def download(request: DownloadRequest):
    if request.code != EnvVars.AUTH_CODE:
        raise HTTPException(status_code=403, detail="Código de autorização inválido.")
    try:
        downloader = DatasetDownloader(
            url=Urls.DATASET_URL, download_dir=PathFiles.DATASETS
        )
        downloader.run()
        return {"status": "sucesso", "mensagem": "Arquivos baixados e extraídos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
def clear(request: ClearRequest):
    if request.code != EnvVars.AUTH_CODE:
        raise HTTPException(status_code=403, detail="Código de autorização inválido.")
    try:
        clear_directory(PathFiles.DATASETS)
        return {"message": f"Limpeza {PathFiles.DATASETS} - concluída com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preparer")
def preparer(request: PreparerRequest):
    if request.code != EnvVars.AUTH_CODE:
        raise HTTPException(status_code=403, detail="Código de autorização inválido.")
    try:
        preparer = DatasetPreparer(source_dir=PathFiles.CUB, target_dir=PathFiles.DATA)
        preparer.prepare()
        return {"message": f"Dataset reorganizado com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    code: str = Query(..., description="Código de autorização"),
):
    if code != EnvVars.AUTH_CODE:
        raise HTTPException(status_code=403, detail="Código de autorização inválido.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(img_size)

        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model(PathFiles.MODEL)

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = CLASS_NAMES[predicted_class]

        return JSONResponse(
            content={
                "class_id": predicted_class,
                "class_name": predicted_label,
                "confidence": round(confidence, 4),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {e}")
