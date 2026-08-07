import cv2
import yaml
from pathlib import Path

from tqdm import tqdm

from src.utils import SFTPHelper, camera_parser, process_meshes, download_contents, upload_contents
from src.mask import Annotations, IDMap, SAMEngine, process_masks
from src.depth import process_depthmaps


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml", "r"))

    sftp = SFTPHelper()

    dataset_dir = Path(cfg['dataset_dir'])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if cfg['sftp_settings']['download_data']:
        print("Downloading files via SFTP...")
        for day in tqdm(cfg['days'], desc="Days downloaded"):
            download_contents(
                sftp,
                '',
                str(dataset_dir),
                cfg['sftp_settings']['download_dir'],
                cfg['images_dir'],
                cfg['cameras'],
                (day, )
            )

    if cfg['sftp_settings']['process_meshes']:
        print("Simplifying meshes...")
        process_meshes(
            str(dataset_dir),
            cfg['sftp_settings']['mesh_gb_threshold'],
            cfg['sftp_settings']['decimate_percent']
        )

    sam = SAMEngine()
    id_map = IDMap(str(dataset_dir))
    clahe = cv2.createCLAHE(
        clipLimit=cfg['clip_limit'],
        tileGridSize=(cfg['tile_grid_size'], cfg['tile_grid_size'])
    )

    dirs = []
    for day in dataset_dir.iterdir():
        if not day.is_dir(): continue
        if day.name not in cfg['days']: continue

        for plot in day.iterdir():
            if not plot.is_dir(): continue

            for camera in plot.iterdir():
                if not camera.is_dir(): continue
                if camera.name.lower() not in cfg['cameras']: continue

                dirs.append((day, plot, camera))

    for day, plot, camera in tqdm(dirs, desc="Plots processed"):
        print(f"Processing {day.name}/{plot.name}/{camera.name}")

        camera_file = Path(camera) / cfg['cameras_file']
        if not camera_file.exists():
            print(f"Camera file not found for {day.name}/{plot.name}/{camera.name}. Skipping.")
            continue

        sensors = camera_parser(str(camera_file))
        for sensor in sensors:
            sensor.compute_distortion_maps(
                max_iter=cfg['depthmap_generation']['max_iterations'],
                tol=cfg['depthmap_generation']['tolerance'],
                eta=cfg['depthmap_generation']['damping']
            )

        if cfg['depthmap_generation']['enabled']:
            mesh_file = Path(camera) / cfg['mesh_file']
            if not mesh_file.exists():
                print(f"Mesh file not found for {day.name}/{plot.name}/{camera.name}. Skipping depthmap and mask generation.")
                continue

            process_depthmaps(
                sensors,
                Path(camera) / cfg['depths_dir'],
                mesh_file
            )

        if cfg['mask_generation']['enabled']:
            seed_images = Path(camera) / cfg['seedpoints_images_file']
            seed_points = Path(camera) / cfg['seedpoints_3d_file']

            if not seed_images.exists() or not seed_points.exists():
                print(f"Seedpoints files not found for {day.name}/{plot.name}/{camera.name}. Skipping mask generation.")
                continue

            process_masks(
                Path(camera) / cfg['masks_dir'],
                Path(camera) / cfg['images_dir'],
                Path(camera) / cfg['depths_dir'],
                cfg['mask_generation']['prompt_type'],
                cfg['mask_generation']['sampling_mode'],
                cfg['mask_generation']['min_area'],
                cfg['mask_generation']['indexed_mapping'],
                clahe,
                sensors[0],
                sam,
                Annotations(
                    seed_images,
                    seed_points
                ),
                id_map,
                cfg['mask_generation']['occlusion_threshold'],
                cfg['mask_generation']['distance_threshold'],
                cfg['mask_generation']['bb_area_threshold'],
                cfg['mask_generation']['point_sample_threshold']
            )

    if cfg['sftp_settings']['upload_data']:
        print("Uploading files via SFTP...")
        upload_contents(
            sftp,
            str(dataset_dir),
            cfg['sftp_settings']['upload_dir']
        )
