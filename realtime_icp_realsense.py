import open3d as o3d
import copy 
import numpy as np
from pyrealsense import PyRealSenseCamera
import time


# Set up necessary configurations for real-time data acquisition
voxel_size = 0.01
factor = 20
max_correspondence_distance = voxel_size * factor
radius_normal = voxel_size * 5


def preprocess_point_cloud(pcd_down, voxel_size):
    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    

    return pcd_down, pcd_fpfh

# def preprocess_point_cloud(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     radius_normal = voxel_size * 2
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#     radius_feature = voxel_size * 5
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

def register_point_clouds(source, target):
    start_time = time.time()
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    elapsed_time = time.time() - start_time
    print('preprocess pcd time', elapsed_time)

    
    distance_threshold = voxel_size * 0.5
    
    start_time = time.time()

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    elapsed_time = time.time() - start_time
    print('calc fpfh time', elapsed_time)
    return result.transformation

def convert_to_pcd(rgb_image, depth_image, camera_intrinsics):
    # rgb image scaling 
    rgb_image = rgb_image.astype('uint8')

    # convert rgb image to open3d depth map
    rgb_image = o3d.geometry.Image(rgb_image)

    # depth image scaling
    depth_image = depth_image.astype('uint16')
    
    # convert depth image to open3d depth map
    depth_image = o3d.geometry.Image(depth_image)
    
    # convert to rgbd image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,
                                                                    depth_image,
                                                                    convert_rgb_to_intensity=False)

    # rgbd image convert to pointcloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    return pcd


if __name__ == '__main__':
    # Set up the point cloud registration algorithm
    # Initialize the point cloud registration algorithm
    icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    prev_pcd = None
    cam_pose = np.eye(4)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd_combined = o3d.geometry.PointCloud()

    camera = PyRealSenseCamera()
    camera.capture()
    rgb = camera.get_color()[:, :, :3].astype(np.uint8)
    intrinsic_matrix = camera.get_camera_intrinsic()
    
    width = rgb.shape[1]
    height = rgb.shape[0]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Create Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                          height=height,
                                                          fx=fx,
                                                          fy=fy,
                                                          cx=cx,
                                                          cy=cy)
    

    while True:
        camera.capture()
        
        rgb_image = camera.get_color()[:, :, :3].astype(np.uint8)
        depth_image = camera.get_depth() / 10.
        
        pcd = convert_to_pcd(rgb_image=rgb_image,
                             depth_image=depth_image,
                             camera_intrinsics=camera_intrinsics)
        
        

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        raw_pcd = copy.deepcopy(pcd)

        pcd = pcd.voxel_down_sample(voxel_size)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
        
        # Preprocess the point clouds if necessary
        # (assuming you have already implemented this)

        # Estimate the camera motion between the previous and current frames
        if prev_pcd is not None:
            # cam_pose = register_point_clouds(source=prev_pcd, target=pcd)
            # result = o3d.pipelines.registration.registration_icp(
            #     source=prev_pcd, 
            #     target=pcd, 
            #     max_correspondence_distance=max_correspondence_distance, 
            #     init=cam_pose, 
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
            # )

            # result = o3d.pipelines.registration.registration_icp(prev_pcd, pcd, max_correspondence_distance,
            #     cam_pose, o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            result = o3d.pipelines.registration.registration_colored_icp(prev_pcd,
                                                                              pcd, max_correspondence_distance,
                cam_pose,  o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                )
            
            
            cam_pose = result.transformation

        # Register the current point cloud with the previous point cloud using ICP
        if prev_pcd is not None:
            # result = o3d.pipelines.registration.registration_icp(
            #     source=pcd, 
            #     target=prev_pcd, 
            #     max_correspondence_distance=max_correspondence_distance, 
            #     init=cam_pose, 
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
            # )
            
            result = o3d.pipelines.registration.registration_colored_icp(pcd,
                                                                              prev_pcd, max_correspondence_distance,
                cam_pose,  o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                )
            
            print(result.transformation)
            pcd.transform(result.transformation)
            raw_pcd.transform(result.transformation)
        # Add the current point cloud to the visualizer
        vis.add_geometry(raw_pcd)

        # Update the visualizer
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        # Update the previous point cloud for the next iteration
        prev_pcd = pcd
        # time.sleep(0.5)
