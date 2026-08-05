from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d

from tqdm import tqdm

from .cameras import Sensor, Pose
from .raytrace import raytrace


def get_world_coordinates(depthmap: np.ndarray, sensor: Sensor, pose: Pose) -> np.ndarray:
    # Convert depth to meters
    dist = depthmap.astype(np.float32) / 1000.0

    # Multiply normalized rays by depth to get actual 3D camera coordinates
    x_cam = sensor.x
    y_cam = sensor.y
    z_cam = np.full_like(x_cam, 1.0)

    # Stack into an (H, W, 3) array
    ray_vectors = np.stack((x_cam, y_cam, z_cam), axis=-1)
    ray_vectors /= np.linalg.norm(ray_vectors, axis=-1, keepdims=True)

    # Scale the normalized rays by the Euclidean distance
    ray_vectors *= dist[..., np.newaxis]

    # Rotate local ray vectors to world space
    R = pose.T[:3, :3]
    ray_vectors = np.einsum('ij,hwj->hwi', R, ray_vectors)

    # Broadcast translation to the grid size
    origins = np.broadcast_to(pose.T[:3, 3], ray_vectors.shape)

    return origins + ray_vectors  # World coordinates of each pixel


def process_depthmaps(
    sensors: List[Sensor],
    output_dir: Path,
    mesh_path: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    views: List[Tuple[Sensor, Pose]] = []
    for s in sensors:
        views.extend([(s, p) for p in s.poses])

    views_to_process = []
    completed = set(output_dir.glob("*.png"))
    views_to_process = [
        (s, p) for (s, p) in views
        if f"{p.label}.png" not in completed
    ]

    if not views_to_process:
        print("All views have already been processed.")
        return

    # Load the mesh and build the BVH tree ONCE
    print("Loading mesh...")
    main_mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    RAY_CASTER = o3d.t.geometry.RaycastingScene()
    RAY_CASTER.add_triangles(main_mesh)

    print(f"Processing {len(views_to_process)} views...")
    for view_data in tqdm(views_to_process, desc="Raytracing"):
        sensor, pose = view_data

        try:
            depth, heatmap = raytrace(RAY_CASTER, sensor, pose)
            cv2.imwrite(output_dir / f"{pose.label}.png", depth)
            cv2.imwrite(output_dir / f"{pose.label}_heatmap.jpg", heatmap)
        except Exception as exc:
            print(f"\nView '{pose.label}' generated an exception: {exc}")
