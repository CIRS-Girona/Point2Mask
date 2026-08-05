import json

import cv2
import numpy as np

from datetime import datetime
from pathlib import Path

from typing import List, Dict, Any


class CocoExporter:
    def __init__(self):
        self.info = {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": datetime.now().isoformat()
        }
        self.licenses = [{"id": 1, "name": "Unknown", "url": ""}]
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: List[Dict[str, Any]] = []

        # Internal tracking
        self._cat_name_to_id: Dict[str, int] = {}
        self._image_path_to_id: Dict[str, int] = {}
        self._current_ann_id = 1
        self._current_img_id = 1
        self._current_cat_id = 1

    def add_category(self, name: str) -> int:
        """Ensures category exists and returns its ID."""
        if name not in self._cat_name_to_id:
            cat_id = self._current_cat_id
            self.categories.append({
                "id": cat_id,
                "name": name,
                "supercategory": "object"
            })
            self._cat_name_to_id[name] = cat_id
            self._current_cat_id += 1
        return self._cat_name_to_id[name]

    def add_image(self, filename: str, height: int, width: int, creation_time: float) -> int:
        """Registers an image and returns its ID."""
        if filename not in self._image_path_to_id:
            img_id = self._current_img_id
            self.images.append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": filename,
                "license": 1,
                "date_captured": datetime.fromtimestamp(creation_time).isoformat()
            })
            self._image_path_to_id[filename] = img_id
            self._current_img_id += 1
        return self._image_path_to_id[filename]

    def add_annotation(self, image_id: int, category_id: int, binary_mask: np.ndarray, tolerance: float = 0.005) -> List[List[float]]:
        """Converts binary mask to COCO polygon and adds annotation."""
        # Find contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        area = 0.0

        all_points_x = []
        all_points_y = []

        for contour in contours:
            if cv2.contourArea(contour) < 10:  # Filter tiny noise
                continue

            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            # Simplify: epsilon is the max distance from the original curve
            epsilon = tolerance * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # COCO expects [x1, y1, x2, y2, ...]
            poly = approx.flatten().tolist()
            if len(poly) >= 6:  # Need at least 3 points (6 coords)
                segmentation.append(poly)
                area += cv2.contourArea(contour)

                # Collect points for global bbox calculation
                all_points_x.extend(poly[0::2])
                all_points_y.extend(poly[1::2])

        if not segmentation:
            return segmentation

        # Calculate bounding box [x, y, width, height]
        if all_points_x and all_points_y:
            x_min = min(all_points_x)
            y_min = min(all_points_y)
            width = max(all_points_x) - x_min
            height = max(all_points_y) - y_min
            bbox = [x_min, y_min, width, height]
        else:
            bbox = [0, 0, 0, 0]

        annotation = {
            "id": self._current_ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": float(area),
            "bbox": bbox,
            "iscrowd": 0
        }
        self.annotations.append(annotation)
        self._current_ann_id += 1

        return segmentation

    def save(self, output_path: Path):
        """Writes the accumulated data to a JSON file."""
        coco_output = {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"COCO annotations saved to {output_path}")