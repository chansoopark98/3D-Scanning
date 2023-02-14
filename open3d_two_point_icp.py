import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy

voxel_size = 0.05
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
max_iterations = 5

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
    icp_pcds = []
    depth_list = []

    while True:
        if idx == 2:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        raw_depth = capture.depth
        raw_depth = np.where(raw_depth>600, 0, raw_depth)
        raw_depth = np.where(raw_depth<300, 0, raw_depth)

        depth_list.append(raw_depth)
        time.sleep(1.5)
        idx += 1

     # Define the intrinsic parameters of the depth camera
    intrinsic_matrix = k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    for depth_idx in range(len(depth_list)):
        print('Convert depth to pointcloud. Try index = {0}'.format(depth_idx))
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
        icp_pcds.append(pcd)


    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

        source = icp_pcds[0]
        target = icp_pcds[1]
        
        # Estimate normal vectors
        print('Estimate normal vectors')
        source.estimate_normals()
        target.estimate_normals()
        
        print('Calculate ICP')
        result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance_coarse, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))

        print('Transform target pcd')
        target.transform(result.transformation)

        # Merge the point clouds
        merged_pcd = o3d.geometry.PointCloud()
        
        merged_pcd += source
        merged_pcd += target

        # Visualize the mesh
        o3d.visualization.draw_geometries([merged_pcd])
