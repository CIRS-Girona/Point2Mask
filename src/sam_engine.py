import torch
import numpy as np
import cv2

from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, KDTree, distance_matrix
from sklearn.decomposition import PCA
from skimage.graph import route_through_array
from skimage.morphology import closing, disk, medial_axis
from skimage.transform import resize
from skimage.draw import polygon as draw_polygon

from transformers import SamModel, SamProcessor

from typing import List, Optional


class SAMEngine:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM on {self.device}...")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    def infer(self,
        image: np.ndarray,
        points: np.ndarray,
        label: str,
        prompt_type: str = "bb",
        sampling_mode: str = "hr"
    ) -> Optional[np.ndarray]:
        """
        'prompt_type' can be 'bb' or 'pt' or 'both' (default)
        to switch between bounding box or point-based prompting or both.
        'sampling_mode' can be 'hr' (default) or 'yc'
        to switch between different prompt sampling strategies.
        """

        unique_pts = np.unique(points, axis=0)
        if len(unique_pts) < 3: return None

        try:
            hull = ConvexHull(unique_pts)
            vertices = unique_pts[hull.vertices]
            poly = Polygon(vertices)

            point_prompts = None
            box_prompt = None

            if prompt_type in ('pt', 'both'):
                point_prompts = self.medial_axis_sampling(unique_pts, vertices, offset=0.2) \
                    if 'chain' not in label and sampling_mode != 'yc' else \
                        self.curvature_spline_sampling(unique_pts, image.shape[:2]) \
                            if sampling_mode != 'yc' else \
                                self.local_coverage_sampling(poly, unique_pts,
                                    mode='p' if 'chain' in label or 'scale' in label else 'l')

                if not point_prompts:
                    print(f"No valid prompts for {label} with {len(unique_pts)} unique points."
                          " Falling back to box prompts.")
                    prompt_type = 'bb'

            if prompt_type in ('bb', 'both'):
                box_prompt = [[list(poly.bounds)]]  # [x_min, y_min, x_max, y_max]

            # SAM Inference
            inputs = self.processor(images=image, return_tensors="pt",
                        input_points=point_prompts, input_boxes=box_prompt
                    ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            # Combine masks
            final_mask = masks[0].squeeze(0).permute(1, 2, 0).numpy()
            for m in masks[1:]:
                final_mask |= m.squeeze(0).permute(1, 2, 0).numpy()
            final_mask = np.all(final_mask, axis=2)

            # IOU Check
            h, w = image.shape[:2]
            hull_mask = np.zeros((h, w), dtype=np.uint8)
            rr, cc = draw_polygon(
                np.array(poly.exterior.coords)[:, 1], 
                np.array(poly.exterior.coords)[:, 0],
                shape=(h, w))
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

    def medial_axis_sampling(self,
        points: np.ndarray,
        hull_vertices: np.ndarray,
        offset: float = 0.2
    ) -> List[List[List[float]]]:
        pca = PCA(n_components=1)
        pca.fit(points)

        # Get the medial axis (principal direction with maximum variance)
        principal_axis = pca.components_[0]
        mean = pca.mean_

        # Project hull points onto the principal axis
        projections = (hull_vertices - mean) @ principal_axis
        min_proj, max_proj = projections.min(), projections.max()

        # Compute three key points (end-points at least offset% within the object)
        center_proj = (min_proj + max_proj) / 2
        offset_proj = (max_proj - min_proj) * offset
        left_proj = min_proj + offset_proj
        right_proj = max_proj - offset_proj

        # Convert projections back to 2D points
        center_point = mean + principal_axis * center_proj
        left_point = mean + principal_axis * left_proj
        right_point = mean + principal_axis * right_proj

        sampled_points = [[
            center_point.astype(int).tolist(),
            left_point.astype(int).tolist(),
            right_point.astype(int).tolist()
        ]]

        return sampled_points

    def curvature_spline_sampling(self,
        points: np.ndarray,
        image_shape: tuple,
        num_samples: int = 5
    ) -> List[List[List[float]]]:
        object_mask = np.zeros(image_shape, dtype=bool)
        points = np.round(points).astype(int)
        object_mask[points[:, 1], points[:, 0]] = True

        # Compute extent of the object
        rect = cv2.minAreaRect(points.astype(np.float32))
        diagonal = np.linalg.norm(rect[1])

        # Sample fewer points for smaller objects
        num_samples = 3 if diagonal < 0.25 * min(*image_shape) else \
            5 if diagonal < 0.75 * min(*image_shape) else 7

        # Resize mask to make morph faster
        object_mask_ = resize(object_mask, (image_shape[0]//10, image_shape[1]//10),
                        anti_aliasing=False, order=0)
        object_mask_ = (object_mask_ > 0).astype(int)

        # Perform morphological closing to fill holes
        object_mask_ = closing(object_mask_, disk(5))

        # Extract largest connected skeleton
        skeleton, dist_transform = medial_axis(object_mask_, return_distance=True)
        skeleton_points = np.column_stack(np.where(skeleton)[::-1]) # (y,x) to (x,y)

        # Find the longest path in the skeleton
        dist_mat = distance_matrix(skeleton_points, skeleton_points) # Compute pairwise distances
        start_idx = np.argmax(dist_mat[0]) # Farthest from an arbitrary point
        end_idx = np.argmax(dist_mat[start_idx]) # Farthest from the true start
        center_idx = np.argmax(dist_transform[skeleton])

        cost_map = np.where(skeleton, 1 / (dist_transform + 1e-6), np.inf) # Bias toward center
        path1, _ = route_through_array(
            cost_map, skeleton_points[start_idx][::-1], skeleton_points[center_idx][::-1],
            fully_connected=True)
        path2, _ = route_through_array(
            cost_map, skeleton_points[center_idx][::-1], skeleton_points[end_idx][::-1],
            fully_connected=True)
        path_coords = np.vstack((np.array(path1)[:, ::-1], np.array(path2)[1:, ::-1]))

        # Compute cumulative arc length to sample points evenly
        segment_lengths = np.linalg.norm(np.diff(path_coords, axis=0), axis=1)
        arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

        # sample_arc_lengths = np.linspace(0, arc_lengths[-1], num_samples)
        sample_arc_lengths = np.linspace(
            0.05 * arc_lengths[-1], (1 - 0.05) * arc_lengths[-1], num_samples) # 5% margin
        sampled_points = np.column_stack([
            np.interp(sample_arc_lengths, arc_lengths, path_coords[:, 0]),
            np.interp(sample_arc_lengths, arc_lengths, path_coords[:, 1])
        ]).astype(int) * 10 # Rescale to original size

        return [sampled_points.tolist()]

    def local_coverage_sampling(self, poly: Polygon, points: np.ndarray, mode: str, 
            candidates_count=2000) -> List[List[List[float]]]:
        """Generates prompt points using circle packing or point density."""

        n_circles, radius = (8,150) if poly.area > 1_000_000 \
            else (5,150) if poly.area > 500_000 \
                else (3,100) if poly.area > 100_000 \
                    else (2,90)
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

        return [[list(p.coords[0]) for p in selected]]
