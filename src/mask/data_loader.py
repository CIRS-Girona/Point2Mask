import csv

import numpy as np
from scipy.spatial import KDTree
import matplotlib.cm as cm

from pathlib import Path
from typing import Dict, Tuple


class Annotations:
    def __init__(self, seedpoints_images_path: Path):
        self.image_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._load(seedpoints_images_path)

    def _load(self, images_path: Path):
        if not images_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {images_path}")

        temp_data = {}
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
    def __init__(self, dir_path: str, superclasses: Dict[int, set], label_correction: Dict[str, str]):
        self.file_path = Path(dir_path) / "ID_map.csv"
        self._ids: Dict[str, int] = {}
        self._reverse_ids: Dict[int, str] = {}

        self.superclasses = superclasses
        self.label_correction = label_correction
        self.superclass_count = {k: 0 for k in self.superclasses.keys()}

        self._load()

    def _load(self):
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) == 2:
                        supercls = self.get_superclass(row[0].split('_')[0])
                        self.superclass_count[supercls] += 1

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

        if cls in self.label_correction:
            cls = self.label_correction[cls]

        supercls = self.get_superclass(cls)

        # Preserve group information
        label_id = f"{cls}_{grp}"

        # Create a unique numerical identifier based on the class
        if label_id not in self._ids:
            self._ids[label_id] = self.superclass_count[supercls]
            self._reverse_ids[self._ids[label_id]] = label_id
            self.superclass_count[supercls] += 1
            self._save()

        object_id = self.encode_object_id(supercls, self._ids[label_id], int(inst))
        rgb_color = self.object_id_to_rgb(object_id)

        return object_id, rgb_color

    def get_label(self, object_id: int) -> str | None:
        superclass_id, class_id, instance_id = self.decode_object_id(object_id)
        return self._reverse_ids.get(class_id, None)

    def get_superclass(self, class_id: str) -> int:
        for superclass, classes in self.superclasses.items():
            if class_id in classes:
                return superclass

        return 0  # 0 is always the default class

    @staticmethod
    def encode_object_id(superclass_id: int, class_id: int, instance_id: int) -> int:
        """
        Returns a unique object ID based on class ID and instance ID.
        
        Allows for 1,000 instances per class
        Allows for 10,000 classes per superclass
        Allows for a total of 167 superclasses
        """
        return superclass_id * 100_000 + class_id * 1_000 + instance_id

    @staticmethod
    def decode_object_id(object_id: int) -> Tuple[int, int, int]:
        """Returns the superclass ID, class ID, and instance ID from a unique object ID."""
        superclass_id = object_id // 100_000
        class_id = (object_id % 100_000) // 1_000
        instance_id = object_id % 1_000
        return superclass_id, class_id, instance_id


    @staticmethod
    def object_id_to_rgb(object_id: int) -> np.ndarray:
        """Encodes a class ID and instance ID into a single RGB color."""

        # Reference: https://arxiv.org/abs/1801.00868
        r = (object_id >> 16) & 255
        g = (object_id >> 8) & 255
        b = object_id & 255

        return np.array([r, g, b], dtype=np.uint8)

    @staticmethod
    def rgb_to_object_id(encoded_id: np.ndarray) -> Tuple[int, int, int]:
        object_id = (encoded_id[0] << 16) + (encoded_id[1] << 8) + encoded_id[2]
        return IDMap.decode_object_id(object_id)

    @staticmethod
    def get_colors(n: int) -> np.ndarray:
        """Returns a list of n distinct colors from the gist_rainbow colormap in RGB format."""
        if n == 0:
            return np.array([])

        return np.array([
            255 * np.array(cm.gist_rainbow(c)[:3])
            for c in np.arange(0, 1, 1/n)
        ], dtype=np.uint8)
