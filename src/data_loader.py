import csv, gc
import numpy as np
import matplotlib.cm as cm

from pathlib import Path
from typing import Dict, Tuple


class Annotations:
    def __init__(self, csv_path: Path):
        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._load(csv_path)

    def _load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        temp_data = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4: continue

                img_name, x, y, cl = row
                if img_name not in temp_data:
                    temp_data[img_name] = {'cls': [], 'pts': []}

                temp_data[img_name]['cls'].append(cl)
                temp_data[img_name]['pts'].append((round(float(x), 3), round(float(y), 3)))

        # Convert to numpy arrays immediately
        while temp_data:
            k, v = temp_data.popitem()
            self.data[k] = (np.array(v['cls']), np.array(v['pts']))

            del k, v
            gc.collect()


class Colormap:
    def __init__(self, csv_path: str):
        self.file_path = Path(csv_path)
        self.colors: Dict[str, Tuple[int, int, int]] = {}
        self._load()

    def _hex_to_rgb(self, h: str) -> Tuple[int, ...]:
        return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

    def _rgb_to_hex(self, c: Tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*c)

    def _load(self):
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) == 2:
                        self.colors[row[0]] = self._hex_to_rgb(row[1])

    def save(self):
        """Call this manually at the end of processing, not every loop."""
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for label, color in self.colors.items():
                writer.writerow((label, self._rgb_to_hex(color)))

    def get_color(self, label: str) -> Tuple[int, int, int]:
        if label not in self.colors:
            while True:
                rgb = tuple(map(int, 255 * np.array(cm.gist_ncar(np.random.random())[:3])))
                if rgb not in self.colors.values():
                    self.colors[label] = rgb
                    break

        return self.colors[label]
