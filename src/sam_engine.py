import torch
import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, KDTree
from skimage.draw import polygon as draw_polygon
from transformers import SamModel, SamProcessor
from typing import List, Tuple, Optional, Union

class SAMEngine:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM on {self.device}...")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    def _get_packing_params(self, area: float) -> Tuple[int, int]:
        """Dynamic radius and count based on polygon area."""
        if area > 1_000_000: return 8, 150
        elif area > 500_000: return 5, 150
        elif area > 100_000: return 3, 100
        return 2, 90

    def generate_prompts(self, poly: Polygon, points: np.ndarray, mode: str, candidates_count=2000) -> List[List[List[float]]]:
        """Generates prompt points using circle packing or point density."""
        n_circles, radius = self._get_packing_params(poly.area)
        min_x, min_y, max_x, max_y = poly.bounds
        
        # vectorizing random generation might speed this up, but loop is okay for 2000
        candidates = []
        for _ in range(candidates_count):
            p = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            if poly.contains(p):
                candidates.append(p)

        if not candidates: return []

        scored = []
        if mode == 'p' and points.size > 0:
            tree = KDTree(points)
            for p in candidates:
                count = len(tree.query_ball_point([p.x, p.y], radius))
                scored.append((count, p))
        else:
            for p in candidates:
                inter_area = poly.intersection(p.buffer(radius)).area
                scored.append((inter_area, p))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for _, p in scored:
            if all(p.distance(s) >= 2 * radius for s in selected):
                selected.append(p)
            if len(selected) >= n_circles: break

        return [[[list(p.coords[0]) for p in selected]]]

    def infer(self, image: np.ndarray, points: np.ndarray, label: str) -> Optional[np.ndarray]:
        """Returns the binary mask for the object group."""
        unique_pts = np.unique(points, axis=0)
        if len(unique_pts) < 3: return None

        try:
            hull = ConvexHull(unique_pts)
            poly = Polygon(unique_pts[hull.vertices])
            
            # Mode determination from label name (logic preserved from original)
            mode = 'p' if 'chain' in label or 'scale' in label else 'l'
            prompts = self.generate_prompts(poly, unique_pts, mode)

            if not prompts: return None

            # SAM Inference
            inputs = self.processor(images=image, input_points=prompts, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )

            # Combine masks
            final_mask = masks[0].squeeze(0).permute(1, 2, 0).numpy()
            for m in masks[1:]:
                final_mask |= m.squeeze(0).permute(1, 2, 0).numpy()
            final_mask = np.all(final_mask, axis=2)

            # IOU Check
            h, w = image.shape[:2]
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            rr, cc = draw_polygon(np.array(poly.exterior.coords)[:, 1], np.array(poly.exterior.coords)[:, 0], shape=(h, w))
            hull_mask[rr, cc] = 1

            intersection = np.logical_and(final_mask, hull_mask)
            union = np.logical_or(final_mask, hull_mask)

            # Avoid division by zero
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)

            if iou > 0.2:
                dilated = cv2.dilate(hull_mask, np.ones((5, 5), np.uint8), iterations=50)
                final_mask = np.logical_and(final_mask, dilated)

            return final_mask

        except Exception as e:
            print(f"Failed on {label}: {e}")
            return None