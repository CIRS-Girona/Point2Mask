import cv2, tqdm, gc
import numpy as np

from src.settings import Config
from src.data_loader import Annotations, Colormap
from src.sam_engine import SAMEngine
from src.image_ops import enhance_image, post_process_mask, render_polygon_mask
from src.coco_exporter import CocoExporter


def main():
    cfg = Config("config.yaml")
    sam = SAMEngine()
    colormap = Colormap(cfg.colormap_path)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clip_limit,
        tileGridSize=(cfg.tile_grid, cfg.tile_grid)
    )

    for working_dir in tqdm.tqdm(cfg.directories):
        print(f"Processing: {working_dir}")
        paths = cfg.get_paths(working_dir)

        if not paths['annot'].exists():
            print(f"  - Missing annotations: {paths['annot']}")
            continue

        print(f"Loading seed points")
        try:
            annotations = Annotations(paths['annot'])
        except Exception as e:
            print(f"  - Error reading annotations: {e}")
            continue

        paths['output'].mkdir(parents=True, exist_ok=True)

        coco = CocoExporter()
        for img_name, (labels, points) in tqdm.tqdm(annotations.data.items()):
            img_path = paths['images'] / f"{img_name}.jpg"
            
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
            image = enhance_image(image, clahe)

            h, w = image.shape[:2]
            creation_time = img_path.stat().st_ctime
            coco_img_id = coco.add_image(f"{img_name}.jpg", h, w, creation_time)

            rgb_mask_accum = np.zeros_like(image)
            idx_mask_accum = np.zeros(image.shape[:2], dtype=np.uint8)

            has_mask = False
            for label in np.unique(labels):  # Process every object in the image
                group_points = points[labels == label]

                raw_mask = sam.infer(
                    image, group_points, label,
                    cfg.prompt_type, cfg.sampling_mode
                )
                if raw_mask is None: continue

                color = colormap.get_color(label)
                filled_mask, colored_layer = post_process_mask(raw_mask, color, cfg.min_area)

                rgb_mask_accum = cv2.add(rgb_mask_accum, colored_layer)

                category_name = label.split('_')[0]
                coco_cat_id = coco.add_category(category_name)
                segmentation = coco.add_annotation(coco_img_id, coco_cat_id, filled_mask)

                label_idx = cfg.mapping.get(category_name, 0)
                idx_mask_accum[filled_mask == 1] = label_idx
                has_mask = True

            if has_mask:
                cv2.imwrite(str(paths['output'] / f"{img_name}_rgb.png"),
                    cv2.cvtColor(rgb_mask_accum, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(paths['output'] / f"{img_name}_idx.png"),
                    idx_mask_accum)

                vis = cv2.addWeighted(image, 1, rgb_mask_accum, 0.6, 0)
                cv2.imwrite(str(paths['output'] / f"{img_name}_overlay.jpg"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        coco_output_path = paths['output'] / "annotations_coco.json"
        coco.save(coco_output_path)

        # Cleanup per directory
        del annotations, coco
        gc.collect()

    # Save colormap updates once at the very end
    colormap.save()
    print("Processing complete.")


if __name__ == '__main__':
    main()