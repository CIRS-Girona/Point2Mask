import os, gc, stat

import paramiko
import pymeshlab
from tqdm import tqdm
from dotenv import load_dotenv

from typing import List
from multiprocessing.pool import ThreadPool

GB_BYTES = 1000**3


class SFTPHelper:
    def __init__(self):
        load_dotenv()

        # Get data from .env file
        self.transport = paramiko.Transport((os.getenv("NAS_HOST"), int(os.getenv("NAS_PORT"))))
        self.transport.connect(username=os.getenv("NAS_USER"), password=os.getenv("NAS_PASS"))
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except IOError:
            return False

    def is_dir(self, path):
        try:
            return stat.S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            return False

    def get_size(self, path):
        return self.sftp.stat(path).st_size

    def makedirs(self, remote_directory):
        """Recreates 'os.makedirs' for SFTP."""
        dirs = remote_directory.split('/')
        current_dir = ""
        if remote_directory.startswith('/'):
            current_dir = "/"
        for part in dirs:
            if not part: continue
            current_dir = os.path.join(current_dir, part)
            if not self.exists(current_dir):
                self.sftp.mkdir(current_dir)

    def close(self):
        self.sftp.close()
        self.transport.close()


def download_contents(helper: SFTPHelper, filepath: str):
    nas_dir_path = f"{DOWNLOAD_DIR}/{filepath}"

    if not helper.exists(nas_dir_path):
        return

    for file_attr in helper.sftp.listdir_attr(nas_dir_path):
        file = file_attr.filename
        nas_file_path = f"{nas_dir_path}/{file}"

        # Filtering logic
        if "web" in file or ("_cc" in file and "annotations" not in file) or "mosaic" in file or "color_cal" in file:
            continue

        if stat.S_ISDIR(file_attr.st_mode):
            download_contents(helper, f"{filepath}/{file}")
            continue

        # Extraction logic (Day, Plot, Camera)
        day, plot, camera = None, None, None
        for part in filepath.split('/') + [file]:
            part = part.lower()
            if ("25" in part or "26" in part) and day is None:
                day = part.split('_')[0]
            elif "plot" in part and plot is None:
                plot = '_'.join(part.split('_')[:-1])
            elif ("mm" in part or "gps" in part) and camera is None:
                for cam in CAMERAS:
                    if cam in part:
                        camera = cam
                        break

        if not all([day, plot, camera]):
            continue

        c = camera if camera != "gps" else "GPS"
        local_dir_path = f"{LOCAL_DIR}/{day}/{plot}/{c}/"

        label = '.'.join(file.split('.')[:-1])
        ext = file.split('.')[-1].lower()
        if ext == "jpg":
            local_dir_path += "/images"
            local_file_path = f"{local_dir_path}/{label}.{ext}"
        elif ext == "obj":
            local_file_path = f"{local_dir_path}/mesh.obj"
        elif ext == "xml":
            local_file_path = f"{local_dir_path}/cams.xml"
        elif ext in ["csv", "mtl", "png"]:
            local_file_path = f"{local_dir_path}/{label}.{ext}"
        else:
            continue

        os.makedirs(local_dir_path, exist_ok=True)

        # Check for skip
        if os.path.exists(local_file_path) and file_attr.st_size == os.path.getsize(local_file_path):
            continue
        elif ".obj" in file and os.path.exists(f"{local_dir_path}/mesh.ply"):
            continue

        helper.sftp.get(nas_file_path, local_file_path)


def upload_contents(helper: SFTPHelper):
    for day in tqdm(os.listdir(LOCAL_DIR), desc="Days uploaded"):
        day_path = os.path.join(LOCAL_DIR, day)
        if not os.path.isdir(day_path): continue

        for plot in os.listdir(day_path):
            plot_path = os.path.join(day_path, plot)
            for camera in os.listdir(plot_path):
                remote_base = f"{UPLOAD_DIR}/{day}/{plot}/{camera}"
                helper.makedirs(remote_base)

                print(f"Processing {day}/{plot}/{camera}")

                for d in ("masks", "depthmaps"):
                    local_sub_dir = f"{plot_path}/{camera}/{d}/"
                    if not os.path.exists(local_sub_dir):
                        continue

                    remote_sub_dir = f"{remote_base}/{d}"
                    helper.makedirs(remote_sub_dir)

                    for f in os.listdir(local_sub_dir):
                        l_file = os.path.join(local_sub_dir, f)
                        r_file = f"{remote_sub_dir}/{f}"

                        # Skip if exists and size matches
                        if helper.exists(r_file) and os.path.getsize(l_file) == helper.get_size(r_file):
                            continue

                        helper.sftp.put(l_file, r_file)


def simplify_mesh(mesh_path: str):
    if os.path.getsize(mesh_path) > GB_THRESHOLD:
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
                targetfacenum=face_num,
                preservenormal=True,
                preservetopology=True
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


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)

    nas_helper = SFTPHelper()

    try:
        if UPLOAD_DATA:
            print("Uploading files via SFTP...")
            upload_contents(nas_helper)
        else:
            print("Downloading files via SFTP...")
            for day in tqdm(DAYS):
                download_contents(nas_helper, day)
                download_contents(nas_helper, f"{day}_processed")
    finally:
        nas_helper.close()

    print("Simplifying meshes...")
    pool = ThreadPool(THREAD_COUNT)

    meshes = find_all_meshes(LOCAL_DIR)
    list(tqdm(pool.imap_unordered(simplify_mesh, meshes), total=len(meshes)))
