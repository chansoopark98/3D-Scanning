import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2

voxel_size = 10
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
        if idx == 36:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        align_depth = capture.depth
        align_depth = np.where(align_depth>600, 0, align_depth)
        align_depth = align_depth.astype(np.float32)
        align_depth /= 1000
        align_depth *= 255
        align_depth = align_depth.astype(np.uint8)
        align_depth = np.where(align_depth>95, align_depth, 0)
        
        output_depth = align_depth.copy()
        output_depth = output_depth.astype(np.float32)
        output_depth /= 255
        output_depth *= 1000
        output_depth = output_depth.astype(np.int16)
    
        depth_list.append(output_depth)

        cv2.imshow('test', align_depth)
        if cv2.waitKey(1000) == ord('q'): # q를 누르면 종료   
            break
    
    cv2.destroyAllWindows()

    # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        
    """
        camera intrinsic matrix =>
            [[503.90097046   0.         323.00683594]
            [  0.         503.88079834 340.80682373]
            [  0.           0.           1.        ]]
    """

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    

    for depth_idx in range(len(depth_list)):
        print('Convert depth to pcs. Try index = {0}'.format(depth_idx))
        # Load the aligned depth image
        depth_image = depth_list[depth_idx]

        # Calculate the 3D coordinates of each pixel
        raw_pcd = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                depth = depth_image[v, u]
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                raw_pcd[v, u] = np.array([x, y, z], dtype=np.float32)

        raw_pcd = np.reshape(raw_pcd, [-1, 3])
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.voxel_down_sample(voxel_size)
        time.sleep(1)
    
        pcds.append(pcd)
    
    # Combine all point clouds into a single point cloud
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd

    # Visualize the result
    o3d.visualization.draw_geometries([combined_pcd])