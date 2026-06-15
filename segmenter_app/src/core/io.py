from abc import ABC, abstractmethod
from typing import Generator, Any, List, Dict
import numpy as np
import os
import imageio.v3 as imageio
import pandas as pd
import json
import glob

# --- INTERFACES ---

class DataSource(ABC):
    @abstractmethod
    def load_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def stream_video(self) -> Generator[np.ndarray, None, None]:
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

    @abstractmethod
    def list_files(self, extensions: List[str]) -> List[str]:
        pass

class DataSink(ABC):
    @abstractmethod
    def save_image(self, image: np.ndarray, name: str):
        pass

    @abstractmethod
    def save_video(self, frames: List[np.ndarray], name: str, fps: int):
        pass

    @abstractmethod
    def save_stats(self, data: List[Dict], name: str, format: str):
        pass

# --- LOCAL IMPLEMENTATIONS ---

class LocalFileSource(DataSource):
    def __init__(self, path: str):
        self.path = path

    def load_image(self) -> np.ndarray:
        return imageio.imread(self.path)

    def stream_video(self) -> Generator[np.ndarray, None, None]:
        return imageio.imiter(self.path)

    def get_metadata(self) -> dict:
        try:
            return imageio.immeta(self.path)
        except:
            return {}

    def list_files(self, extensions: List[str]) -> List[str]:
        if not os.path.isdir(self.path):
            return [self.path]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.path, ext)))
        return sorted(files)

class LocalFileSink(DataSink):
    def __init__(self, base_path: str):
        self.base_path = base_path
        # If base_path has an extension, treat parent as dir
        if os.path.splitext(base_path)[1]:
            self.output_dir = os.path.dirname(base_path)
        else:
            self.output_dir = base_path
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, image: np.ndarray, name: str):
        # Handle full paths or relative names
        if os.path.isabs(name):
            path = name
        else:
            path = os.path.join(self.output_dir, name)
        imageio.imwrite(path, image)

    def save_video(self, frames: List[np.ndarray], name: str, fps: int):
        path = os.path.join(self.output_dir, name)
        imageio.imwrite(path, frames, fps=fps, codec='libx264')

    def save_stats(self, data: List[Dict], name: str, format: str):
        if not data: return
        filename = f"{name}.{format}"
        path = os.path.join(self.output_dir, filename)

        if format == 'csv':
            pd.DataFrame(data).to_csv(path, index=False)
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
        elif format == 'txt':
            with open(path, 'w') as f:
                headers = list(data[0].keys())
                f.write("\t".join(headers) + "\n")
                for entry in data:
                    row = [str(entry.get(k, "")) for k in headers]
                    f.write("\t".join(row) + "\n")

# --- FACTORY ---

class IOFactory:
    @staticmethod
    def get_source(uri: str) -> DataSource:
        # In future, check for 's3://' or 'http://'
        return LocalFileSource(uri)

    @staticmethod
    def get_sink(uri: str) -> DataSink:
        return LocalFileSink(uri)
