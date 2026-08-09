import csv

import numpy as np
from scipy.spatial import KDTree
import matplotlib.cm as cm

from pathlib import Path
from typing import Dict, Tuple


class Annotations:
    def __init__(self, seedpoints_images_path: Path, seedpoints_3d_path: Path):
        self.prompt_data: Dict[str, KDTree] = {}
        self.image_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._load(seedpoints_images_path, seedpoints_3d_path)

    def _load(self, images_path: Path, points_3d_path: Path):
        if not images_path.exists() or not points_3d_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {images_path} or {points_3d_path}")

        temp_data = {}
        with open(points_3d_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4: continue

                x, y, z, cls = row
                if cls not in temp_data:
                    temp_data[cls] = []  # 3D points

                temp_data[cls].append((round(float(x), 3), round(float(y), 3), round(float(z), 3)))

        # Convert to KDTree objects immediately
        while temp_data:
            k, v = temp_data.popitem()
            self.prompt_data[k] = KDTree(np.array(v))

        temp_data.clear()
        with open(images_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4: continue

                img_name, x, y, cls = row
                if img_name not in temp_data:
                    temp_data[img_name] = ([], [])  # classes, points

                temp_data[img_name][0].append(cls)
                temp_data[img_name][1].append((round(float(x), 3), round(float(y), 3)))

        while temp_data:
            k, v = temp_data.popitem()
            self.image_data[k] = (np.array(v[0]), np.array(v[1]))


class IDMap:
    def __init__(self, dir_path: str):
        self.file_path = Path(dir_path) / "ID_map.csv"
        self._ids: Dict[str, int] = {}
        self._reverse_ids: Dict[str, int] = {}
        self._load()

    def _load(self):
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) == 2:
                        self._ids[row[0]] = int(row[1])
                        self._reverse_ids[int(row[1])] = row[0]

    def _save(self):
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for label, id in self._ids.items():
                writer.writerow((label, id))

    def get_id(self, label: str) -> Tuple[int, np.ndarray]:
        """Returns the object ID and encoded RGB color for a given label."""
        cls, grp, inst = label.strip().split('_')

        # Create a unique numerical identifier based on the class
        if cls not in self._ids:
            self._ids[cls] = len(self._ids) + 1
            self._reverse_ids[self._ids[cls]] = cls
            self._save()

        object_id = self.encode_object_id(self._ids[cls], int(inst))
        rgb_color = self.object_id_to_rgb(object_id)

        return object_id, rgb_color

    @staticmethod
    def encode_object_id(class_id: int, instance_id: int) -> int:
        """Returns a unique object ID based on class ID and instance ID."""
        return class_id * 1000 + instance_id

    @staticmethod
    def decode_object_id(object_id: int) -> Tuple[int, int]:
        """Returns the class ID and instance ID from a unique object ID."""
        class_id = object_id // 1000
        instance_id = object_id % 1000
        return class_id, instance_id

    @staticmethod
    def object_id_to_rgb(object_id: int) -> np.ndarray:
        """Encodes a class ID and instance ID into a single RGB color."""

        # Reference: https://arxiv.org/abs/1801.00868
        r = (object_id >> 16) & 255
        g = (object_id >> 8) & 255
        b = object_id & 255

        return np.array([r, g, b], dtype=np.uint8)

    @staticmethod
    def rgb_to_object_id(encoded_id: np.ndarray) -> Tuple[int, int]:
        object_id = (encoded_id[0] << 16) + (encoded_id[1] << 8) + encoded_id[2]
        return IDMap.decode_object_id(object_id)

    @staticmethod
    def get_colors(n: int) -> np.ndarray:
        """Returns a list of n distinct colors from the gist_rainbow colormap in RGB format."""
        return np.array([
            255 * np.array(cm.gist_rainbow(c)[:3])
            for c in np.arange(0, 1, 1/n)
        ], dtype=np.uint8)
