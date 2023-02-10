import matplotlib.pyplot as plt

import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d

import numpy as np
import cv2



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
    while True:
        idx += 1
        if idx == 10:
            break
        capture = k4a.get_capture()

        # rgb = capture.color[:, :, :3]
        raw_pcd = capture.depth_point_cloud
        plt.imshow(raw_pcd)
        plt.show()
        raw_pcd = np.reshape(raw_pcd, [-1, 3])

        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.voxel_down_sample(voxel_size=0.02)

        pcds.append(pcd)
        # print(pcd.shape)
        # print(pcd.dtype)
        
        # cv2.imshow('test', pcd)
        # cv2.waitKey(500)

    # Perform multi-way registration
    point_clouds = pcds[1:]
    reference = pcds[0]
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for i, point_cloud in enumerate(point_clouds):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(i, point_cloud, reference.transform))
    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(0, 1, o3d.registration.get_identity_matrix(), 
                                                                    information=o3d.registration.get_identity_matrix(), 
                                                                    uncertain=False))
    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(1, 2, o3d.registration.get_identity_matrix(), 
                                                                    information=o3d.registration.get_identity_matrix(), 
                                                                    uncertain=False))
    o3d.pipelines.registration.global_optimization(pose_graph)

    # Apply the transformations to the point clouds
    for i in range(len(point_clouds)):
        point_clouds[i].transform(pose_graph.nodes[i].pose)

    # Combine the registered point clouds into a single point cloud
    registered_point_cloud = o3d.geometry.PointCloud()
    for point_cloud in point_clouds:
        registered_point_cloud += point_cloud