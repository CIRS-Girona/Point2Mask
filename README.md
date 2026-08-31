# Point2Mask: SERDP Photogrammetry & SAM 2 Mask Generation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)
![SAM 2](https://img.shields.io/badge/Model-SAM%202.1--Hiera--Large-black.svg)
![Open3D](https://img.shields.io/badge/Raytracing-Open3D-green.svg)
![License](https://img.shields.io/badge/License-GPLv3-blue)

**Point2Mask** is an end-to-end computer vision and photogrammetry pipeline designed for processing multi-temporal field imagery. It integrates 3D mesh raytracing, camera distortion modeling, 3D-to-2D spatial prompt filtering, and target object segmentation using Meta's **Segment Anything Model 2 (SAM 2)**.

The pipeline manages the entire lifecycle of field dataset processing—from automated SFTP synchronization with remote NAS storage to generating high-precision depth maps, image enhancement, COCO-formatted mask exports, class-indexed mask maps, and geometric analytics.

---

## 🌟 Key Features

* **Automated NAS Synchronization**: Download raw dataset assets (images, 3D meshes, camera calibrations, and seedpoints) and automatically upload processed results via SFTP.
* **3D Mesh Decimation & Optimization**: Converts large photogrammetric `.obj` meshes into optimized `.ply` files using PyMeshLab quadric edge collapse decimation for fast raytracing.
* **Photogrammetric Camera Calibration**: Parses Agisoft Metashape camera XML files (`cams.xml`) to construct full projection matrices and compute non-linear Brown-Conrady radial and tangential lens distortion mappings.
* **Hardware-Accelerated Depth Raytracing**: Leverages Open3D BVH raycasting scenes to project 3D geometry onto 2D image coordinates, producing 16-bit millimeter depth maps and Inferno depth heatmaps.
* **Enhanced Image Pre-processing**: Applies bilateral denoising, log-power per-channel color balancing, HSV value/saturation adjustment, and Contrast Limited Adaptive Histogram Equalization (CLAHE) for optimal segmentation quality.
* **Spatial 3D Seedpoint Prompting**: Uses SciPy 3D KDTrees to filter occluded, out-of-bounds, or distant prompt points, dynamically constructing tight convex hull box prompts for SAM 2 (`facebook/sam2.1-hiera-large`).
* **Multi-Format Mask Export**: Generates:
  - **COCO Annotations**: Standardized `annotations_coco.json` polygon contours & bounding boxes.
  - **Indexed Masks**: 1-channel 8-bit PNGs (`*_idx.png`) mapped to user-defined class integer values.
  - **RGB Encoded Masks**: 3-channel instance-encoded RGB PNGs (`*_rgb.png`).
  - **Visual Overlays**: Colorized mask overlays (`*_overlay.jpg`) for visual inspection.
* **Depth & Viewing Angle Analytics**: Automatically computes min, max, mean, median, and standard deviation for target distance (in meters) and viewing angles (in degrees), saved to `stats.csv`.

---

## 📁 Dataset Directory Structure

Input datasets are structured in `Day / Plot / Camera` hierarchies under `dataset_dir`:

```
dataset_dir/
├── 260204/                               # Day folder (YYMMDD format)
│   ├── plot_1/                           # Plot identifier
│   │   ├── 18mm/                         # Camera / Lens folder (e.g., 18mm, 24mm, 35mm, 55mm, GPS)
│   │   │   ├── images/                   # Raw 2D imagery (.jpg)
│   │   │   ├── depthmaps/                # Output depth maps (.png) & heatmaps (_heatmap.jpg)
│   │   │   ├── masks/                    # Output SAM 2 masks (*_rgb.png, *_idx.png, annotations_coco.json)
│   │   │   ├── cams.xml                  # Agisoft Metashape camera intrinsics & extrinsics
│   │   │   ├── mesh.ply                  # Decimated 3D surface mesh
│   │   │   ├── seedpoints_on_images.csv  # 2D image-space seed points (u, v, label)
│   │   │   ├── seedpoints_in_3D.csv      # 3D mesh-space seed points (x, y, z, label)
│   │   │   └── stats.csv                 # Distance and viewing angle statistics per camera pose
│   │   └── 24mm/
│   │       └── ...
│   └── plot_2/
│       └── ...
└── 260227/
    └── ...
```

---

## 🚀 Getting Started

### Prerequisites

* Linux / Unix environment
* Python 3.10+
* CUDA-compatible GPU (recommended for SAM 2 inference & Open3D raytracing)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CIRS-Girona/Point2Mask.git
   cd Point2Mask
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory specifying remote NAS SFTP credentials:
   ```env
   NAS_HOST="nas.address"
   NAS_PORT="22"
   NAS_USER="your_username"
   NAS_PASS="your_password"
   ```

---

## 🖼️ Example Output

Below is an illustration of original field imagery alongside generated 16-bit depth heatmaps and SAM 2 colorized mask overlays across different camera focal lengths:

<table style="width: 100%; text-align: center;"> 
  <tr>
    <th>Original Image</th>
    <th>Depth Heat Map</th>
    <th>Mask Overlay</th>
  </tr>
  <tr>
    <td><img src="assets/18mm.jpg" alt="18MM Original Image" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/18mm_heatmap.jpg" alt="18MM Heat Map" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/18mm_overlay.jpg" alt="18MM Mask Overlay" style="max-width: 512px; width: 100%; height: auto;"/></td>
  </tr>
  <tr>
    <td><img src="assets/24mm.jpg" alt="24MM Original Image" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/24mm_heatmap.jpg" alt="24MM Heat Map" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/24mm_overlay.jpg" alt="24MM Mask Overlay" style="max-width: 512px; width: 100%; height: auto;"/></td>
  </tr>
  <tr>
    <td><img src="assets/gopro.jpg" alt="GoPro Original Image" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/gopro_heatmap.jpg" alt="GoPro Heat Map" style="max-width: 512px; width: 100%; height: auto;"/></td>
    <td><img src="assets/gopro_overlay.jpg" alt="GoPro Mask Overlay" style="max-width: 512px; width: 100%; height: auto;"/></td>
  </tr>
</table>

---

## 📜 License

This project is distributed under the terms of the GNU General Public License v3.0 (`LICENSE`).

