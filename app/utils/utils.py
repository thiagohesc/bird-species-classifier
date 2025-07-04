import shutil
from pathlib import Path


def clear_directory(path: str):
    path = Path(path)
    if path.exists():
        for item in path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Erro ao remover {item}: {e}")
    else:
        path.mkdir(parents=True, exist_ok=True)
