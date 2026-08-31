import os

from tqdm import tqdm
from typing import List

from .sftp import SFTPHelper


def download_contents(
    sftp: SFTPHelper,
    filepath: str,
    local_dir: str,
    download_dir: str,
    images_dir: str,
    cameras: List[str],
    days: List[str],
    mesh_file: str,
    cam_file: str
):
    nas_dir_path = f"{download_dir}/{filepath}"

    if not sftp.exists(nas_dir_path):
        return

    for file in sftp.listdir(nas_dir_path):
        nas_file_path = f"{nas_dir_path}/{file}"

        # Filtering logic
        if "web" in file or "mosaic" in file or "color_cal" in file:
            continue

        if sftp.is_dir(nas_file_path):
            download_contents(
                sftp,
                f"{filepath}/{file}",
                local_dir,
                download_dir,
                images_dir,
                cameras,
                days,
                mesh_file,
                cam_file
            )
            continue

        # Extraction logic (Day, Plot, Camera)
        day, plot, camera = None, None, None
        for part in filepath.split('/') + [file]:
            part = part.lower()
            if ("25" in part or "26" in part) and day is None:
                first_part = part.split('_')[0]
                if first_part.isdigit() and len(first_part) == 6:
                    day = first_part
            elif "plot" in part and plot is None:
                plot = '_'.join(part.split('_')[:-1])
            elif ("mm" in part or "gps" in part) and camera is None:
                for cam in cameras:
                    if cam in part:
                        camera = cam
                        break

        if not all([day, plot, camera]):
            continue
        elif day not in days:
            continue

        c = camera if camera != "gps" else "GPS"
        local_dir_path = f"{local_dir}/{day}/{plot}/{c}/"

        label = '.'.join(file.split('.')[:-1])
        ext = file.split('.')[-1].lower()
        if ext == "jpg" and "cc" not in nas_file_path:
            local_dir_path += f"/{images_dir}"
            local_file_path = f"{local_dir_path}/{label}.{ext}"
        elif ext == "obj" and ("gps" in file or "cc" in file):
            local_file_path = f"{local_dir_path}/{mesh_file.split('.')[0]}.obj"
        elif ext == "xml" and ("gps" in file or "cc" in file):
            local_file_path = f"{local_dir_path}/{cam_file}"
        elif ext == "csv" or (ext in ["mtl", "png"] and ("gps" in file or "cc" in file)):
            local_file_path = f"{local_dir_path}/{label}.{ext}"
        else:
            continue

        os.makedirs(local_dir_path, exist_ok=True)

        # Check for skip
        if os.path.exists(local_file_path) and sftp.get_size(nas_file_path) == os.path.getsize(local_file_path):
            continue
        elif ".obj" in file and os.path.exists(f"{local_dir_path}/{mesh_file}"):
            continue

        sftp.download(nas_file_path, local_file_path)


def upload_contents(
    sftp: SFTPHelper,
    local_dir: str,
    upload_dir: str
):
    for day in tqdm(os.listdir(local_dir), desc="Days uploaded"):
        if not os.path.isdir(f"{local_dir}/{day}"): continue
        for plot in os.listdir(f"{local_dir}/{day}"):
            if not os.path.isdir(f"{local_dir}/{day}/{plot}"): continue
            for camera in os.listdir(f"{local_dir}/{day}/{plot}"):
                if not os.path.isdir(f"{local_dir}/{day}/{plot}/{camera}"): continue

                local_base = f"{local_dir}/{day}/{plot}/{camera}"
                remote_base = f"{upload_dir}/{day}/{plot}/{camera}"
                sftp.makedirs(remote_base)

                print(f"Processing {day}/{plot}/{camera}")

                stats_path = f"{local_base}/stats.csv"
                if os.path.exists(stats_path):
                    sftp.sftp.put(stats_path, f"{remote_base}/stats.csv")

                for d in ("masks", "depthmaps"):
                    local_sub_dir = f"{local_base}/{d}/"
                    if not os.path.exists(local_sub_dir):
                        continue

                    remote_sub_dir = f"{remote_base}/{d}"
                    sftp.makedirs(remote_sub_dir)

                    for f in os.listdir(local_sub_dir):
                        l_file = f"{local_sub_dir}/{f}"
                        r_file = f"{remote_sub_dir}/{f}"

                        # Skip if exists and size matches
                        if sftp.exists(r_file) and os.path.getsize(l_file) == sftp.get_size(r_file):
                            continue

                        sftp.sftp.put(l_file, r_file)