import open3d as o3d
import copy 
import numpy as np
from pyrealsense import PyRealSenseCamera
import time
import open3d.core as o3c

# treg = o3d.cuda.pybind.t.pipelines.registration
treg = o3d.t.pipelines.registration

# Set up necessary configurations for real-time data acquisition
voxel_size = 0.01
factor = 100
max_correspondence_distance = voxel_size * factor
radius_normal = voxel_size * 5

# Initial alignment or source to target transform.
cam_pose = np.identity(4)

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

def point_cloud_to_cuda_tensor(pcd):
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    # print(points.shape)

    # pcd_tensor = o3d.t.geometry.PointCloud(points, np.float32)

    points_tensor = o3c.Tensor(points, o3c.float32, device=o3c.Device("CUDA:0"))
    colors_tensor = o3c.Tensor(colors, o3c.float32, device=o3c.Device("CUDA:0"))
    
    pcd_tensor = o3d.t.geometry.PointCloud(points_tensor)
    pcd_tensor.point.colors = colors_tensor

    pcd_tensor.point["colors"] = pcd_tensor.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

    # points = np.asarray(pcd.points, dtype=np.float32)
    # o3c.Tensor(points, device=o3c.Device("CUDA:0")).cuda(0)
    return pcd_tensor

if __name__ == '__main__':
    
    print(o3d.t.io.RealSenseSensor.list_devices())

    prev_pcd = None

    vis = o3d.visualization.Visualizer()
    vis.create_window()

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
    
    plane_estimation = treg.TransformationEstimationPointToPlane()
    color_estimation = treg.TransformationEstimationForColoredICP()
    point_estimation = treg.TransformationEstimationPointToPoint()

    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                       relative_rmse=0.000001,
                                       max_iteration=100)
    
    # callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    # loss_log_map["iteration_index"].item(), 
    # loss_log_map["scale_index"].item(), 
    # loss_log_map["scale_iteration_index"].item(), 
    # loss_log_map["fitness"].item(), 
    # loss_log_map["inlier_rmse"].item()))

    import matplotlib.pyplot as plt
    while True:
        camera.capture()
        
        rgb_image = camera.get_color()[:, :, :3].astype(np.uint8)
        depth_image = camera.get_depth() / 10.
        
        pcd = convert_to_pcd(rgb_image=rgb_image,
                             depth_image=depth_image,
                             camera_intrinsics=camera_intrinsics)
        
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        pcd = pcd.voxel_down_sample(voxel_size)

        raw_pcd = copy.deepcopy(pcd)
        
        pcd_tensor = point_cloud_to_cuda_tensor(pcd)
        
        pcd_tensor.estimate_normals()

        if prev_pcd is not None:
            source_cuda = pcd_tensor.cuda(0)
            target_cuda = prev_pcd.cuda(0)

            start_time = time.time()

            result = treg.icp(target_cuda, source_cuda, max_correspondence_distance,
                            cam_pose, plane_estimation, criteria,
                            voxel_size)
            
            # result = treg.multi_scale_icp(prev_pcd,
            #                               pcd_tensor,
            #                               voxel_size,
            #                               criteria_list,
            #                               max_correspondence_distances,
            #                               cam_pose, plane_estimation)
                

            elapsed_time = time.time() - start_time
            print('color 1', elapsed_time)
            
            cam_pose = result.transformation

        
        if prev_pcd is not None:
            start_time = time.time()

            result = treg.icp(source_cuda, target_cuda, max_correspondence_distance,
                            cam_pose, point_estimation, criteria,
                            voxel_size)

            # result = treg.multi_scale_icp(source_cuda,
            #                               target_cuda,
            #                               voxel_sizes,
            #                               criteria_list,
            #                               max_correspondence_distances,
            #                               cam_pose, color_estimation)
            
            elapsed_time = time.time() - start_time
            print('color 2', elapsed_time)
            
            pose_matrix = result.transformation.numpy()
            print(pose_matrix)
            pcd_tensor.transform(pose_matrix)
            raw_pcd.transform(pose_matrix)
            
        # Add the current point cloud to the visualizer
        vis.add_geometry(raw_pcd)
        vis.poll_events()
        vis.update_renderer()

        prev_pcd = pcd_tensor
        # time.sleep(0.5)