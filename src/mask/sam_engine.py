from typing import List

import torch
import numpy as np
import cv2

from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from skimage.draw import polygon as draw_polygon

from transformers import Sam2VideoModel, Sam2VideoProcessor


class SAMEngine:
    def __init__(self,
        occlusion_th: float,
        distance_th: float,
        bb_length_th: float,
        point_sample_th: int,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.occlusion_th = occlusion_th
        self.distance_th = distance_th
        self.bb_length_th = bb_length_th
        self.point_sample_th = point_sample_th

        print(f"Loading SAM on {self.device}...")
        self.model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large").to(self.device)
        self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large")

        self._session = self.processor.init_video_session(
            inference_device=self.device,
        )

    def reset(self):
        self._session.reset_inference_session()

    def get_prompt(self,
        points,
        depth,
        positions,
        valid_mask,
        tree
    ):
        if not np.any(valid_mask) or tree is None:
            return None, None

        group_points = points[valid_mask]
        group_depths = depth[group_points[:, 1].astype(int), group_points[:, 0].astype(int)] / 1000.0

        # Filter points based on distance from the camera
        group_points = group_points[group_depths <= self.distance_th]

        # Filter occluded points using KDTree
        pixel_positions = positions[group_points[:, 1].astype(int), group_points[:, 0].astype(int)]
        distances, _ = tree.query(pixel_positions, k=1, distance_upper_bound=self.occlusion_th)

        group_points = group_points[distances != np.inf]
        unique_pts = np.unique(group_points, axis=0)

        hull = ConvexHull(unique_pts)
        vertices = unique_pts[hull.vertices]
        poly = Polygon(vertices)

        box_prompt = list(poly.bounds)  # [x_min, y_min, x_max, y_max]

        box_area = (poly.bounds[2] - poly.bounds[0]) * (poly.bounds[3] - poly.bounds[1])
        if np.sqrt(box_area) < self.bb_length_th or len(unique_pts) < self.point_sample_th:
            return None, None

        return box_prompt, poly

    def infer(self,
        image: np.ndarray,
        box_prompt: List[List[int]],
        frame_idx: int,
        object_ids: List[int]
    ) -> np.ndarray:
        # Reference: https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/sam2_video#streaming-video-inference
        inputs = self.processor(images=image, device=self.device, return_tensors="pt").to(self.device)

        if box_prompt is not None:
            self.processor.add_inputs_to_inference_session(
                inference_session=self._session,
                frame_idx=frame_idx,
                obj_ids=object_ids,
                input_boxes=[box_prompt],
                original_size=inputs.original_sizes[0],
            )

        outputs = self.model(inference_session=self._session, frame=inputs.pixel_values[0])
        if box_prompt is None:
            return None

        masks = self.processor.post_process_masks([outputs.pred_masks], original_sizes=inputs.original_sizes, binarize=False)
        mask = masks[0].squeeze(1).permute(1, 2, 0).max(dim=2).cpu().numpy()

        return mask


    def process_mask(self, mask: np.ndarray, poly: Polygon):
        # IOU Check
        h, w = mask.shape[:2]
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        rr, cc = draw_polygon(
            np.array(poly.exterior.coords)[:, 1],
            np.array(poly.exterior.coords)[:, 0],
            shape=(h, w))
        hull_mask[rr, cc] = 1

        intersection = np.logical_and(mask, hull_mask)
        union = np.logical_or(mask, hull_mask)

        # Check if the IOU is above threshold
        if np.sum(intersection) / (np.sum(union) + 1e-6) > 0.2:
            dilated = cv2.dilate(hull_mask, np.ones((5, 5), np.uint8), iterations=50)
            mask = np.logical_and(mask, dilated)

        return mask