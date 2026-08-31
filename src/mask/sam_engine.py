from typing import List

import torch
import numpy as np
import cv2

from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from skimage.draw import polygon as draw_polygon

from transformers import Sam2Model, Sam2Processor


class SAMEngine:
    def __init__(self,
        occlusion_th: float,
        distance_th: float,
        bb_length_th: float,
        point_sample_th: int,
        clip_limit: float,
        tile_grid_size: int
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.occlusion_th = occlusion_th
        self.distance_th = distance_th
        self.bb_length_th = bb_length_th
        self.point_sample_th = point_sample_th

        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size)
        )

        print(f"Loading SAM on {self.device}...")
        self.model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(self.device)
        self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")


    def enhance_image(
        self,
        image: np.ndarray, 
        brightness: int = 10, 
        saturation: float = 1.25
    ) -> np.ndarray:
        """Applies bilateral filtering, color correction, and CLAHE."""
        # Denoise
        image = cv2.bilateralFilter(image, 5, 50.0, 50.0).astype(np.float32)

        max_ch = np.max(image, axis=(0, 1))
        mean_ch = np.mean(image, axis=(0, 1))

        # Log power per channel color correction
        gain = np.mean(mean_ch) / 255
        power_ch = np.log(gain) / np.log(mean_ch / max_ch)

        for i in range(3):
            image[:, :, i] = np.power(image[:, :, i] / max_ch[i], power_ch[i])

        image = np.clip(255 * image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        image[:, :, 2] += brightness
        image[:, :, 1] *= saturation

        image = np.clip(image, 0, 255).astype(np.uint8)
        image[:, :, 2] = self.clahe.apply(image[:, :, 2])

        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def get_prompt(self, points, depth, positions, valid_mask, tree):
        if not np.any(valid_mask) or tree is None:
            return None

        group_points = points[valid_mask]
        group_depths = depth[group_points[:, 1].astype(int), group_points[:, 0].astype(int)] / 1000.0

        # Filter points based on distance from the camera
        group_points = group_points[group_depths <= self.distance_th]
        if len(group_points) == 0:
            return None

        # Filter occluded points using KDTree
        pixel_positions = positions[group_points[:, 1].astype(int), group_points[:, 0].astype(int)]
        distances, _ = tree.query(pixel_positions, k=1, distance_upper_bound=self.occlusion_th)

        group_points = group_points[distances != np.inf]
        if len(group_points) == 0:
            return None

        # Filter based on bb size and num of samples
        unique_pts = np.unique(group_points, axis=0)
        if len(unique_pts) < self.point_sample_th:
            return None

        hull = ConvexHull(unique_pts)
        vertices = unique_pts[hull.vertices]
        poly = Polygon(vertices)

        box_area = (poly.bounds[2] - poly.bounds[0]) * (poly.bounds[3] - poly.bounds[1])
        if np.sqrt(box_area) < self.bb_length_th:
            return None

        return list(poly.bounds)  # [x_min, y_min, x_max, y_max]

    def infer(self,
        image: np.ndarray,
        box_prompts: List[List[int]],
        object_ids: List[int],
    ) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(
                images=image,
                input_boxes=[box_prompts],
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

            mask = masks.squeeze(1).permute(1, 2, 0).cpu().numpy()

            del inputs, outputs, masks
            torch.cuda.empty_cache()

        return np.max(mask * np.array(object_ids), axis=2)