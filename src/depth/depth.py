from pathlib import Path
from typing import List, Tuple

import cv2
import open3d as o3d

from tqdm import tqdm

from .cameras import Sensor, Pose
from .raytrace import raytrace


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
    completed = {file.name for file in output_dir.glob("*.png")}
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
