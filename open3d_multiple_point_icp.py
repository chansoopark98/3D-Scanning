import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import cv2

voxel_size = 0.05
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
max_iterations = 5

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

    pcds = []
    idx = 0
    while True:
        idx += 1
        if idx == 10:
            break
        print('capture idx {0}'.format(idx))
        capture = k4a.get_capture()
        rgb = capture.color
        raw_pcd = capture.transformed_depth_point_cloud

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.float32) / 255
        
        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        rgb = np.reshape(rgb, [-1, 3])

        max_range_mask = np.where(np.logical_and(raw_pcd[:, 2]<600, raw_pcd[:, 2]>400))
        
        raw_pcd = raw_pcd[max_range_mask]
        rgb = rgb[max_range_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        pcds.append(pcd)
        time.sleep(1)

    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        icp_idx = 0
        source = pcds[0]
        
        # 첫 번째 포인트 클라우드가 그 다음번째 인덱스 포인드 클라우드로 맞춰감
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                print('Estimate pcd normal vectors {0}'.format(target_id))
                source = pcd[source_id]
                target = pcd[target_id]
                source.estimate_normals()
                target.estimate_normals()
            
                print('Calculate ICP {0}'.format(target_id))
                transformation = o3d.pipelines.registration.registration_icp(
                    source, target, max_correspondence_distance_coarse, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
                source.transform(transformation.transformation)
                source += target
                source = source.voxel_down_sample(voxel_size)
        # source = source.voxel_down_sample_and_trace(voxel_size)
        # Visualize the result
        o3d.visualization.draw_geometries([source])