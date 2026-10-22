import yaml
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.utils import SFTPHelper, camera_parser, save_img_stats, process_meshes, download_contents, upload_contents
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
                (day, ),
                cfg['mesh_file'],
                cfg['cameras_file']
            )

    if cfg['sftp_settings']['process_meshes']:
        print("Processing meshes...")
        process_meshes(
            str(dataset_dir),
            cfg['mesh_file'],
            cfg['sftp_settings']['mesh_gb_threshold'],
            cfg['sftp_settings']['decimate_percent']
        )

    sam = SAMEngine(
        cfg['mask_generation']['occlusion_threshold'],
        cfg['mask_generation']['distance_threshold'],
        cfg['mask_generation']['bb_area_threshold'],
        cfg['mask_generation']['point_sample_threshold'],
        cfg['clip_limit'], cfg['tile_grid_size']
    )

    id_map = IDMap(
        str(dataset_dir),
        {int(k): set(v) for k, v in cfg['mask_generation']['superclasses'].items()},
        {str(k): str(v) for k, v in cfg['mask_generation']['label_correction'].items()},
    )

    dirs: List[Tuple[Path, Path, Path]] = []
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
            mesh_file = camera / cfg['mesh_file']
            if not mesh_file.exists():
                print(f"Mesh file not found for {day.name}/{plot.name}/{camera.name}. Skipping depthmap and mask generation.")
                continue

            process_depthmaps(
                sensors,
                camera / cfg['depths_dir'],
                mesh_file
            )

        save_img_stats(
            camera / cfg['image_stats_file'],
            camera / cfg['depths_dir'],
            sensors[0]
        )

        if cfg['mask_generation']['enabled']:
            seed_images = camera / cfg['seedpoints_images_file']

            if seed_images.exists():
                process_masks(
                    camera / cfg['masks_dir'],
                    camera / cfg['images_dir'],
                    camera / cfg['depths_dir'],
                    cfg['mask_generation']['indexed_mapping'],
                    sensors[0],
                    sam,
                    Annotations(seed_images),
                    id_map,
                )

        if cfg['sftp_settings']['upload_data']:
            print("Uploading files via SFTP...")
            upload_contents(
                sftp,
                str(dataset_dir),
                cfg['sftp_settings']['upload_dir'],
                day.name, plot.name, camera.name
            )
