from pathlib import Path
import urllib.request
import tarfile


class DatasetDownloader:
    def __init__(self, url, download_dir="/app", filename="data.tgz", extract_to=None):
        self.url = url
        self.download_dir = Path(download_dir)
        self.filename = filename
        self.filepath = self.download_dir / filename

        self.extract_to = Path(extract_to) if extract_to else self.download_dir
        self.extracted_flag = self.extract_to / ".extracted"

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extract_to.mkdir(parents=True, exist_ok=True)

    def download(self):
        if not self.filepath.exists():
            print(f"Baixando dataset de: {self.url}")
            urllib.request.urlretrieve(self.url, self.filepath)
            print(f"Download concluído: {self.filepath}")
        else:
            print(f"Arquivo já existe: {self.filepath}")

    def extract(self):
        if self.extracted_flag.exists():
            print(f"Dataset já extraído em: {self.extract_to}")
            return

        print(f"Extraindo {self.filename} para {self.extract_to}...")
        with tarfile.open(self.filepath, "r:gz") as tar:
            tar.extractall(path=self.extract_to)
        self.extracted_flag.write_text("done")
        print("Extração concluída.")

    def run(self):
        self.download()
        self.extract()
