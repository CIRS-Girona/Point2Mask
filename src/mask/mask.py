import gc
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from tqdm import tqdm

from ..depth import Sensor, get_world_coordinates

from .data_loader import Annotations, IDMap
from .sam_engine import SAMEngine
from .image_ops import enhance_image, post_process_mask
from .coco_exporter import CocoExporter


def process_masks(
        output_dir: Path,
        images_dir: Path,
        depths_dir: Path,
        prompt_type: str,
        sampling_mode: str,
        min_area: int,
        mapping: Dict[str, int],
        clahe,
        sensor: Sensor,
        sam: SAMEngine,
        annotations: Annotations,
        id_map: IDMap,
        occlusion_th: float,
        distance_th: float,
        bb_length_th: float,
        point_sample_th: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = {p.label: p for p in sensor.poses}

    coco = CocoExporter()
    for img_name, (labels, points) in tqdm(annotations.image_data.items()):
        pose = poses.get(img_name, None)
        if pose is None:
            continue

        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        depth_path = depths_dir / f"{img_name}.png"
        if not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).as_type(np.float32)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        image = enhance_image(image, clahe)
        positions = get_world_coordinates(depth, sensor, pose)

        h, w = image.shape[:2]
        creation_time = img_path.stat().st_ctime
        coco_img_id = coco.add_image(f"{img_name}.jpg", h, w, creation_time)

        rgb_mask_accum = np.zeros_like(image)
        idx_mask_accum = np.zeros(image.shape[:2], dtype=np.uint8)

        unique_labels = np.unique(labels)
        colors = id_map.get_colors(len(unique_labels))

        has_mask = False
        for label, color in zip(unique_labels, colors):  # Process every object in the image
            group_points = points[labels == label]

            # Filter points based on distance from the camera
            group_points = group_points[depth / 1000.0 <= distance_th]

            # Filter occluded points using KDTree
            tree = annotations.prompt_data.get(label, None)
            if tree is None:
                continue

            pixel_positions = positions[group_points[:, 1].astype(int), group_points[:, 0].astype(int)]

            distances, _ = tree.query(pixel_positions, k=1, distance_upper_bound=occlusion_th)

            group_points = group_points[distances != np.inf]

            # Process the mask using SAM
            raw_mask = sam.infer(
                image, group_points, label,
                bb_length_th, prompt_type, sampling_mode, point_sample_th
            )
            if raw_mask is None: continue

            filled_mask, colored_layer = post_process_mask(raw_mask, color, min_area)

            rgb_mask_accum = cv2.add(rgb_mask_accum, colored_layer)

            category_name = label.split('_')[0]
            coco_cat_id = coco.add_category(category_name)
            segmentation = coco.add_annotation(coco_img_id, coco_cat_id, filled_mask)

            label_idx = mapping.get(category_name, 0)
            idx_mask_accum[filled_mask == 1] = label_idx
            has_mask = True

        if has_mask:
            cv2.imwrite(str(output_dir / f"{img_name}_rgb.png"),
                cv2.cvtColor(rgb_mask_accum, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{img_name}_idx.png"),
                idx_mask_accum)

            vis = cv2.addWeighted(image, 1, rgb_mask_accum, 0.6, 0)
            cv2.imwrite(str(output_dir / f"{img_name}_overlay.jpg"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Save files in case of crash
        id_map.save()
        coco.save(output_dir / "annotations_coco.json")

    # Cleanup per directory
    del annotations, coco
    gc.collect()
