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
            synchronized_images_only=False
        )
    )
    k4a.start()
    
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    idx = 0
    pcds = []
    depth_list = []
    icp_idx = 0
    while True:
        idx += 1
        if idx == 20:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        raw_pcd = capture.depth_point_cloud
        rgb = capture.color
        
        
        
        raw_pcd = np.reshape(raw_pcd, [-1, 3])
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        # pcd.voxel_down_sample(voxel_size)

        time.sleep(0.5)
    
        pcds.append(pcd)
    
    # Combine all point clouds into a single point cloud
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd

    # Visualize the result
    o3d.visualization.draw_geometries([combined_pcd])