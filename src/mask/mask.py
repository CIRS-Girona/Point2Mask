import gc
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from tqdm import tqdm

from ..depth import Sensor, get_world_coordinates

from .data_loader import Annotations, IDMap
from .sam_engine import SAMEngine
from .coco_exporter import CocoExporter


def process_masks(
        output_dir: Path,
        images_dir: Path,
        depths_dir: Path,
        mapping: Dict[str, int],
        sensor: Sensor,
        sam: SAMEngine,
        annotations: Annotations,
        id_map: IDMap,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = {p.label: p for p in sensor.poses}
    processed = {im.name.split('_overlay')[0] for im in output_dir.glob("*.jpg")}

    coco = CocoExporter()
    for img_name, (labels, points) in tqdm(annotations.image_data.items(), desc="Mask Generation"):
        img_path = images_dir / f"{img_name}.jpg"
        depth_path = depths_dir / f"{img_name}.png"
        pose = poses.get(img_name, None)

        if img_name in processed or not img_path.exists() or not depth_path.exists() or pose is None:
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        if image is None:
            continue

        image = sam.enhance_image(image)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        positions = get_world_coordinates(depth, sensor, pose)

        h, w = image.shape[:2]
        creation_time = img_path.stat().st_ctime
        coco_img_id = coco.add_image(f"{img_name}.jpg", h, w, creation_time)

        rgb_mask_accum = np.zeros_like(image)
        enc_mask_accum = np.zeros_like(image)
        idx_mask_accum = np.zeros(image.shape[:2], dtype=np.uint8)

        prompts, obj_ids = [], []
        for label in np.unique(labels):  # Process every object in the image
            valid_mask = labels == label
            valid_mask &= np.all(points >= 0, axis=1)  # Ensure points are valid
            valid_mask &= np.all(points[:, :2] < [w, h], axis=1)  # Ensure points are within image bounds

            tree = annotations.prompt_data.get(label, None)
            prompt = sam.get_prompt(points, depth, positions, valid_mask, tree)

            if prompt is None:
                continue

            obj_id, _ = id_map.get_id(label)

            prompts.append(prompt)
            obj_ids.append(obj_id)

        if len(obj_ids) == 0:
            continue

        raw_mask = sam.infer(image, prompts, obj_ids)

        colors = IDMap.get_colors(len(obj_ids))
        for obj_id, color in zip(obj_ids, colors):
            loc = raw_mask == obj_id

            rgb_mask_accum[loc] = color
            enc_mask_accum[loc] = id_map.object_id_to_rgb(obj_id)

            filled_mask = 0 * raw_mask
            filled_mask[loc] = 1

            category = id_map.get_label(obj_id)
            coco_cat_id = coco.add_category(category)
            segmentation = coco.add_annotation(coco_img_id, coco_cat_id, filled_mask)

            label_idx = mapping.get(category, 0)
            idx_mask_accum[filled_mask == 1] = label_idx

        if raw_mask is not None and raw_mask.any():
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
