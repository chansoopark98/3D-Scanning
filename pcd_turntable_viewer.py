import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

voxel_size = 0.02
max_correspondence_distance_coarse = 1000
max_correspondence_distance_fine = 10
max_iterations = 500

def calc_turntable_matrix(current_time):
    # 턴테이블 25초간 회전
    rotation_time = 24
    # # Define the rotation time and velocity of the turntable
    angular_velocity = 2 * np.pi / rotation_time  # in radians per second

    print(angular_velocity)

    # # Calculate the angle of rotation based on the rotation time and angular velocity
    angle = angular_velocity * current_time
    
    print(angle)

    # Create a rotation matrix that describes the rotation of the turntable
    transformation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])
    

    # Define the diameter of the turntable
    diameter = 0.8  # in meters

    # Define the distance between the camera and the center of the turntable
    distance = 0.42  # in meters

    # Calculate the horizontal and vertical displacements of the turntable
    horizontal_displacement = diameter / 2 * np.cos(angle)
    vertical_displacement = diameter / 2 * np.sin(angle)

    # Calculate the x, y, and z coordinates of the center of the turntable in the coordinate system of the point cloud
    x = distance * np.sin(np.arccos(horizontal_displacement / distance))
    y = distance * np.sin(np.arccos(vertical_displacement / distance))
    z = distance * np.cos(np.arccos(horizontal_displacement / distance)) * np.cos(np.arccos(vertical_displacement / distance))

    # Create the translation vector
    # transformation_matrix = np.identity(4)
    translation_vector = np.array([x, y, z])
    # transformation_matrix[:3, 3] = translation_vector

    # transformation_matrix = np.dot(transformation_matrix, rotation_matrix)



    print('matrix', transformation_matrix)
    return transformation_matrix



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

    idx = 0
    pcds = []
    capture_idx = 4
    
    start_time = time.time()
    while True:
        idx += 1
        if idx == capture_idx + 1:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        raw_pcd = capture.transformed_depth_point_cloud
        
        current_time = time.time()

        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.float32) / 255
        
        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])

        max_range_mask = np.where(np.logical_and(raw_pcd[:, 2]<700, raw_pcd[:, 2]>400))
        # max_range_mask = np.where(raw_pcd[:, 2]<700)
        
        raw_pcd = raw_pcd[max_range_mask]
        rgb = rgb[max_range_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Get current point cloud center
        center = pcd.get_center()

        # Set new origin coordinate
        new_origin = [1, 1, 1]

        # Calculate translation vector to move point cloud to new origin coordinate
        translation_vector = np.subtract(new_origin, center)

        # Move point cloud to new origin coordinate
        pcd.translate(translation_vector)

        # if idx >= 2:
            
        trans_matrix = calc_turntable_matrix(float(idx-1))
        pcd.transform(trans_matrix)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        

        time.sleep(1)
        pcds.append(pcd)

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries(pcds)

    from open3d_multiway_registration import full_registration

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
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