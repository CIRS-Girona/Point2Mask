import os, gc
from typing import List

import pymeshlab

from tqdm import tqdm

GB_BYTES = 1000**3


def find_meshes(root_dir: str) -> List[str]:
    meshes = []
    for file in os.listdir(root_dir):
        file_path = f"{root_dir}/{file}"

        if os.path.isdir(file_path):
            meshes += find_meshes(file_path)
        elif ".obj" in file:
            meshes.append(file_path)

    return meshes


def process_meshes(root_dir: str, mesh_file: str, gb_threshold: int, target_perc: float):
    meshes = find_meshes(root_dir)
    for mesh in tqdm(meshes, desc="OBJ Meshes Processed"):
        if os.path.getsize(mesh) > gb_threshold * GB_BYTES:
            continue

        try:
            file = mesh.split('/')[-1]
            filepath = '/'.join(mesh.split('/')[:-1])

            if not os.path.exists(f"{filepath}/{mesh_file}"):
                file_size = os.path.getsize(f"{filepath}/{file}") / GB_BYTES
                print(f"Processing {filepath}/{file}: {file_size:.2f} GB")

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(f"{filepath}/{file}")

                ms.apply_filter(
                    'meshing_decimation_quadric_edge_collapse_with_texture',
                    targetperc=target_perc,
                    preservenormal=True,
                )

                ms.save_current_mesh(f"{filepath}/{mesh_file}")

                del ms
                gc.collect()

                os.remove(mesh)
        except Exception as e:
            print(f"An error occurred during conversion of {mesh}:\n{e}")