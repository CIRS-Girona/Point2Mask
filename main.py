import cv2, tqdm, gc
import numpy as np

from src.settings import Config
from src.data_loader import Annotations, Colormap
from src.sam_engine import SAMEngine
from src.image_ops import enhance_image, post_process_mask
from src.coco_exporter import CocoExporter


def main():
    cfg = Config("config.yaml")
    
    # Initialize shared resources
    clahe = cv2.createCLAHE(clipLimit=cfg.clip_limit, tileGridSize=(cfg.tile_grid, cfg.tile_grid))
    colormap = Colormap(cfg.colormap_path)
    sam = SAMEngine()

    for working_dir in tqdm.tqdm(cfg.directories):
        print(f"Processing: {working_dir}")
        paths = cfg.get_paths(working_dir)

        if not paths['annot'].exists():
            print(f"  - Missing annotations: {paths['annot']}")
            continue

        try:
            annotations = Annotations(paths['annot'])
        except Exception as e:
            print(f"  - Error reading annotations: {e}")
            continue

        paths['output'].mkdir(parents=True, exist_ok=True)

        coco = CocoExporter()
        for img_name, (labels, points) in annotations.data.items():
            img_path = paths['images'] / f"{img_name}.jpg"
            
            if not img_path.exists():
                continue

            # Load & Preprocess
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
            image = enhance_image(image, clahe)

            h, w = image.shape[:2]
            creation_time = img_path.stat().st_ctime
            coco_img_id = coco.add_image(f"{img_name}.jpg", h, w, creation_time)

            final_overlay = np.zeros_like(image)
            binary_mask_accum = np.zeros(image.shape[:2], dtype=np.uint8)
            has_mask = False

            # Process every object in the image
            for label_idx in np.unique(labels):
                group_points = points[labels == label_idx]
                
                raw_mask = sam.infer(image, group_points, label_idx)
                if raw_mask is None: continue

                color = colormap.get_color(label_idx)
                filled_mask, colored_layer = post_process_mask(raw_mask, color, cfg.min_area)

                final_overlay = cv2.add(final_overlay, colored_layer)

                category_name = label_idx.split('_')[0]
                coco_cat_id = coco.add_category(category_name)
                coco.add_annotation(coco_img_id, coco_cat_id, filled_mask)

                bin_val = cfg.mapping.get(category_name, 0)
                binary_mask_accum[filled_mask == 1] = bin_val
                has_mask = True

            if has_mask:
                # Save outputs
                cv2.imwrite(str(paths['output'] / f"{img_name}.png"), cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(paths['output'] / f"{img_name}_binary.png"), binary_mask_accum)
                
                vis = cv2.addWeighted(image, 1, final_overlay, 0.6, 0)
                cv2.imwrite(str(paths['output'] / f"{img_name}_overlay.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Save COCO JSON
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