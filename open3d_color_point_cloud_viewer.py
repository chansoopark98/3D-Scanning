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
    depth = capture.depth
    raw_pcd = capture.transformed_depth_point_cloud

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    test = rgb[:, :, :3].astype(np.float32) / 255
    
    print(test.dtype)
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:

        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        test = np.reshape(test, [-1, 3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(test) 

        # Visualize the result
        o3d.visualization.draw_geometries([pcd])