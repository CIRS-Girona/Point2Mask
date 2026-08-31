import csv
from pathlib import Path

import cv2
import numpy as np

from tqdm import tqdm

from ..depth import Sensor, get_viewing_angles

HEADERS = [
    'image_name',
    'min dist. (m)', 'max dist. (m)', 'median dist. (m)', 'mean dist. (m)', 'std dist. (m)',
    'min angle (deg.)', 'max angle (deg.)', 'median angle (deg.)', 'mean angle (deg.)', 'std angle (deg.)',
]


def save_img_stats(output_file: Path, depths_dir: Path, sensor: Sensor):
    get_stats = lambda x: [np.min(x), np.max(x), np.median(x), np.mean(x), np.std(x)]

    if output_file.exists():
        return

    stats = []
    for pose in tqdm(sensor.poses, desc="Image Stats Processed"):
        depth_path = depths_dir / f"{pose.label}.png"
        if not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if not np.any(depth > 0):
            continue

        depthmap = depth / 1000.0
        angles = get_viewing_angles(depth, sensor, pose)

        stats.append([pose.label] + get_stats(depthmap[depth > 0]) + get_stats(angles[np.isfinite(angles)]))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(HEADERS)
        for stat in stats:
            writer.writerow(stat)
