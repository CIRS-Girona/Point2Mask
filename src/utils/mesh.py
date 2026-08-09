import os, gc
from typing import List

import pymeshlab

GB_BYTES = 1000**3


def simplify_mesh(mesh_path: str, gb_threshold: int, target_perc: float):
    if os.path.getsize(mesh_path) > gb_threshold * GB_BYTES:
        return

    try:
        file = mesh_path.split('/')[-1]
        filepath = '/'.join(mesh_path.split('/')[:-1])

        label = ''.join(file.split('.')[:-1])
        if not os.path.exists(f"{filepath}/{label}.ply"):
            file_size = os.path.getsize(f"{filepath}/{label}.obj") / GB_BYTES
            print(f"Processing {filepath}/{file}: {file_size:.2f} GB")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(f"{filepath}/{label}.obj")

            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse_with_texture',
                targetperc=target_perc,
                preservenormal=True,
            )

            ms.save_current_mesh(f"{filepath}/{label}.ply")

            del ms
            gc.collect()

            os.remove(mesh_path)
    except Exception as e:
        print(f"An error occurred during conversion of {mesh_path}:\n{e}")


def find_all_meshes(root_dir: str) -> List[str]:
    meshes = []
    for file in os.listdir(root_dir):
        file_path = f"{root_dir}/{file}"

        if os.path.isdir(file_path):
            meshes += find_all_meshes(file_path)
        elif ".obj" in file:
            meshes.append(file_path)

    return meshes


def process_meshes(root_dir: str, gb_threshold: int, target_perc: float):
    meshes = find_all_meshes(root_dir)
    for mesh in meshes:
        simplify_mesh(mesh, gb_threshold, target_perc)