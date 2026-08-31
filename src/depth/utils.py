import numpy as np

from .cameras import Sensor, Pose


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

    # Normalize the columns of R to remove the Agisoft chunk scale factor
    R = R / np.linalg.norm(R, axis=0)

    # Rotate local ray vectors to world space
    ray_vectors = np.einsum('ij,hwj->hwi', R, ray_vectors)

    # Broadcast translation to the grid size
    origins = np.broadcast_to(pose.T[:3, 3], ray_vectors.shape)

    return origins + ray_vectors  # World coordinates of each pixel


def get_viewing_angles(depthmap: np.ndarray, sensor: Sensor, pose: Pose) -> np.ndarray:
    """
    Computes viewing angle stats using world coordinates.
    """
    world_points = get_world_coordinates(depthmap, sensor, pose)

    # Compute view rays in world space (Vector from camera origin to surface point)
    cam_origin = pose.T[:3, 3]
    view_rays = world_points - cam_origin
    
    view_ray_norms = np.linalg.norm(view_rays, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        view_rays_normalized = view_rays / view_ray_norms

    # Compute surface normals in world space
    dp_du = np.gradient(world_points, axis=1)
    dp_dv = np.gradient(world_points, axis=0)
    
    normals = np.cross(dp_du, dp_dv)
    normal_norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normals_normalized = normals / normal_norms

    # Compute the viewing angle
    dot_product = np.sum(normals_normalized * view_rays_normalized, axis=-1)
    dot_product = np.clip(np.abs(dot_product), 0.0, 1.0)
    
    return np.degrees(np.arccos(dot_product))