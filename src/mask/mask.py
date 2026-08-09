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
        min_area: int,
        mapping: Dict[str, int],
        clahe,
        sensor: Sensor,
        sam: SAMEngine,
        annotations: Annotations,
        id_map: IDMap,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = {p.label: p for p in sensor.poses}
    processed = {im.name.split('_overlay')[0] for im in output_dir.glob("*.jpg")}

    coco = CocoExporter()
    for i, img_path in enumerate(tqdm(sorted(images_dir.glob("*.jpg")), desc="Mask Generation")):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        image = enhance_image(image, clahe)

        # Always feed frame to model to maintain temporal consistency
        if i > 0:
            sam._infer(image, None, i, 0)

        img_name = img_path.name.split('.')[0]
        if img_name in processed:
            continue

        labels, points = annotations.image_data.get(img_name, (None, None))
        if labels is None or points is None:
            continue

        pose = poses.get(img_name, None)
        if pose is None:
            continue

        depth_path = depths_dir / f"{img_name}.png"
        if not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        positions = get_world_coordinates(depth, sensor, pose)

        h, w = image.shape[:2]
        creation_time = img_path.stat().st_ctime
        coco_img_id = coco.add_image(f"{img_name}.jpg", h, w, creation_time)

        rgb_mask_accum = np.zeros_like(image)
        enc_mask_accum = np.zeros_like(image)
        idx_mask_accum = np.zeros(image.shape[:2], dtype=np.uint8)

        unique_labels = np.unique(labels)
        colors = id_map.get_colors(len(unique_labels))

        has_mask = False
        for label, color in zip(unique_labels, colors):  # Process every object in the image
            valid_mask = labels == label
            valid_mask &= np.all(points >= 0, axis=1)  # Ensure points are valid
            valid_mask &= np.all(points[:, :2] < [w, h], axis=1)  # Ensure points are within image bounds

            tree = annotations.prompt_data.get(label, None)
            prompt, poly = sam.get_prompt(points, depth, positions, valid_mask, tree)
            obj_id, enc_rgb = id_map.get_id(label)

            if prompt is None or poly is None:
                continue

            raw_mask = sam.infer(image, prompt, poly, i, obj_id)

            filled_mask, colored_layer = post_process_mask(raw_mask, color, min_area)
            filled_mask, encoded_layer = post_process_mask(raw_mask, enc_rgb, min_area)

            rgb_mask_accum = cv2.add(rgb_mask_accum, colored_layer)
            enc_mask_accum = cv2.add(enc_mask_accum, encoded_layer)

            category_name = label.split('_')[0]
            coco_cat_id = coco.add_category(category_name)
            segmentation = coco.add_annotation(coco_img_id, coco_cat_id, filled_mask)

            label_idx = mapping.get(category_name, 0)
            idx_mask_accum[filled_mask == 1] = label_idx
            has_mask = True

        if has_mask:
            cv2.imwrite(str(output_dir / f"{img_name}_rgb.png"),
                cv2.cvtColor(enc_mask_accum, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{img_name}_idx.png"),
                idx_mask_accum)

            vis = cv2.addWeighted(image, 1, rgb_mask_accum, 0.6, 0)
            cv2.imwrite(str(output_dir / f"{img_name}_overlay.jpg"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Save files in case of crash
        coco.save(output_dir / "annotations_coco.json")

    # Cleanup per directory
    del annotations, coco
    gc.collect()
