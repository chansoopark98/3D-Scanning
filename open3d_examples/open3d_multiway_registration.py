import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

voxel_size = 0.01
max_correspondence_distance_coarse = 10
max_correspondence_distance_fine = 1
max_iterations = 300


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    # ICP할 때 노말벡터 계산해야 함
    source.estimate_normals()
    target.estimate_normals()

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        print('ICP index {0}'.format(source_id))
        for target_id in range(source_id + 1, n_pcds):
            print('try icp registration {0}'.format(target_id))
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

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
    icp_pcds = []

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # select roit
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    mask = np.expand_dims(mask, axis=-1)

    
    while True:
        idx += 1
        if idx == 25:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        raw_pcd = capture.transformed_depth_point_cloud

        cv2.imshow('Pointcloud capture', rgb)
        cv2.waitKey(1)
            
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.float32) / 255

        raw_pcd = raw_pcd * mask
        rgb = rgb * mask

        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])

        max_range_mask = np.where(np.logical_and(raw_pcd[:, 2]<550, raw_pcd[:, 2]>400))
        raw_pcd = raw_pcd[max_range_mask]
        rgb = rgb[max_range_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Compute mean distance of points from origin
        distances = np.sqrt(np.sum(np.square(np.asarray(pcd.points)), axis=1))
        mean_distance = np.mean(distances)

        # Normalize point cloud
        pcd.scale(1 / mean_distance, center=pcd.get_center())
        pcd.translate(-pcd.get_center())

        time.sleep(1)
        icp_pcds.append(pcd)

    # raw_pcds = copy.deepcopy(icp_pcds)

    o3d.visualization.draw_geometries(icp_pcds)

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(icp_pcds,
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
    for point_id in range(len(icp_pcds)):
        icp_pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += icp_pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
