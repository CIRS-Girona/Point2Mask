import os
import yaml

from pathlib import Path
from typing import Dict


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self._cfg = yaml.safe_load(f)

    @property
    def clip_limit(self) -> float: return self._cfg.get('clip_limit', 2.0)
    
    @property
    def tile_grid(self) -> int: return self._cfg.get('tile_grid_size', 8)

    @property
    def min_area(self) -> int: return self._cfg.get('min_mask_area_default', 2000)

    @property
    def prompt_type(self) -> str: return self._cfg.get('prompt_type', 'bb')

    @property
    def sampling_mode(self) -> str: return self._cfg.get('sampling_mode', 'hr')

    @property
    def mapping(self) -> Dict[str, int]: return self._cfg.get('indexed_mapping', {})

    @property
    def directories(self) -> list[Path]:
        root_dir = self._cfg.get('data_dir', None)
        if root_dir is None:
            return []
        dirs = []
        for day in os.listdir(root_dir):
            for plot in os.listdir(f"{root_dir}/{day}"):
                for camera in os.listdir(f"{root_dir}/{day}/{plot}"):
                    if not os.path.exists(f"{root_dir}/{day}/{plot}/{camera}/seedpoints_on_images.csv"):
                        continue
                    elif os.path.exists(f"{root_dir}/{day}/{plot}/{camera}/masks/"):
                        num_images = len(os.listdir(f"{root_dir}/{day}/{plot}/{camera}/images/"))
                        num_masks = len(os.listdir(f"{root_dir}/{day}/{plot}/{camera}/masks/"))
                        if 3 * num_images == num_masks:
                            continue

                    dirs.append(Path(f"{root_dir}/{day}/{plot}/{camera}/"))

        return dirs

    @property
    def colormap_path(self) -> str:
        return self._cfg.get('colormap_path', 'colormap.csv')

    def get_paths(self, base_dir: Path) -> Dict[str, Path]:
        """Returns resolved paths for input/output."""
        return {
            'images': base_dir / self._cfg.get('images_dir', 'images/'),
            'output': base_dir / self._cfg.get('output_dir', 'masks/'),
            'annot': base_dir / self._cfg.get('annotations_file', 'annotations.csv')
        }
