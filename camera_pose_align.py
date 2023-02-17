import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2
from modern_robotics import *


if __name__ == "__main__":
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # Load point clouds
        pcd_list = []
        for i in range(24):
            pcd = o3d.io.read_point_cloud(f"./test_pointclouds/test_pointcloud_{i}.pcd")
            pcd_list.append(pcd)


        # Visualize the combined point cloud
        o3d.visualization.draw_geometries(pcd_list)
        # Initialize transformation list with identity matrix
        transformation_list = [np.identity(4)]

        # Calculate transformation matrices
        for i in range(1, len(pcd_list)):
            source = pcd_list[i]
            target = pcd_list[0]

            # Calculate transformation matrix from source to target
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance=0.02,
                init=np.identity(4), estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
                # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100))
            T = reg_result.transformation

            # Accumulate the transformation matrix
            prev_T = transformation_list[i-1]
            transformation_list.append(np.dot(T, prev_T))

        # Apply transformations to point clouds
        aligned_pcd_list = []
        for i, pcd in enumerate(pcd_list):
            aligned_pcd = pcd.transform(transformation_list[i])
            aligned_pcd_list.append(aligned_pcd)
            # Visualize the combined point cloud
            o3d.visualization.draw_geometries(aligned_pcd_list)

        # Combine point clouds into a single point cloud
        combined_pcd = o3d.geometry.PointCloud()
        for aligned_pcd in aligned_pcd_list:
            combined_pcd += aligned_pcd

        # Visualize the combined point cloud
        o3d.visualization.draw_geometries([combined_pcd])