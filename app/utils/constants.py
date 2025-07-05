from dotenv import load_dotenv
import os

load_dotenv()


class EnvVars:
    AUTH_CODE = os.getenv("AUTH_CODE")
    if AUTH_CODE is None:
        raise EnvironmentError("A variável de ambiente AUTH_CODE não está definida.")

    TRAIN = True


class Urls:
    DATASET_URL = (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    )


class PathFiles:
    ROOT_PATH = "/app"
    DATASETS = f"{ROOT_PATH}/datasets"
    CUB = f"{DATASETS}/CUB_200_2011"
    DATA = f"{DATASETS}/data"
    MODEL = f"{DATA}/model.keras"
    RESULTS = f"{DATASETS}/results/plot_history.png"
