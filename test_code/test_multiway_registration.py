import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2
from open3d_multiway_registration import full_registration
from typing import Optional, Tuple

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return 

voxel_size = 0.05
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
        if idx == 3:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        align_depth = capture.depth
        # align_depth = np.where(align_depth>600, 0, align_depth)
        # align_depth = np.where(align_depth<300, 0, align_depth)
        print(capture._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH))
        print(align_depth.shape)
    
        depth_list.append(align_depth)

        # cv2.imshow('test', colorize(align_depth, (None, 5000), cv2.COLORMAP_HSV))
        # if cv2.waitKey(3000) == ord('q'): # q를 누르면 종료   
            # break
        time.sleep(1)
        idx += 1
    
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
        # pcd.voxel_down_sample(voxel_size)
        
        time.sleep(1.5)
        pcds.append(pcd)
    
    # Check before ICP
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd

    # Visualize the result
    o3d.visualization.draw_geometries([combined_pcd])


    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
        print(pose_graph)
        
        # pose_graph optimization
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])