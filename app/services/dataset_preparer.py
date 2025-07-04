import pandas as pd
from pathlib import Path
import shutil


class DatasetPreparer:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

        self.images_dir = self.source_dir / "images"
        self.images_txt = self.source_dir / "images.txt"
        self.labels_txt = self.source_dir / "image_class_labels.txt"
        self.splits_txt = self.source_dir / "train_test_split.txt"
        self.classes_txt = self.source_dir / "classes.txt"

    def prepare(self):
        images = pd.read_csv(self.images_txt, sep=" ", names=["img_id", "filepath"])
        labels = pd.read_csv(self.labels_txt, sep=" ", names=["img_id", "label"])
        labels["label"] = labels["label"] - 1
        splits = pd.read_csv(self.splits_txt, sep=" ", names=["img_id", "is_train"])
        classes = pd.read_csv(
            self.classes_txt, sep=" ", names=["class_id", "class_name"]
        )
        classes["class_id"] = classes["class_id"] - 1

        df = (
            images.merge(labels, on="img_id")
            .merge(splits, on="img_id")
            .merge(classes, left_on="label", right_on="class_id")
        )

        for _, row in df.iterrows():
            split = "train" if row.is_train == 1 else "test"
            class_name = row.class_name
            src_path = self.images_dir / row.filepath
            dst_path = self.target_dir / split / class_name / src_path.name

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                print(f"Erro ao copiar {src_path}: {e}")

        print(f"Dataset reorganizado com sucesso em: {self.target_dir}")
