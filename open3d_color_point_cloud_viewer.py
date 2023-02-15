import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2

voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True
        )
    )
    k4a.start()
    
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510


    capture = k4a.get_capture()
    rgb = capture.color
    raw_pcd = capture.transformed_depth_point_cloud

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb[:, :, :3].astype(np.float32) / 255
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:

        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])

        max_range_mask = np.where(raw_pcd[:, 2]<600)
        min_range_mask = np.where(raw_pcd[:, 2]>300)

        raw_pcd = raw_pcd[max_range_mask]
        rgb = rgb[max_range_mask]
        raw_pcd = raw_pcd[min_range_mask]
        rgb = rgb[min_range_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Visualize the result
        o3d.visualization.draw_geometries([pcd])