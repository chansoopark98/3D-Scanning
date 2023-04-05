import open3d as o3d
import numpy as np
import time
from pyrealsense import PyRealSenseCamera
import matplotlib.pyplot as plt
# from slam_config import ConfigParser
from slam_common import save_poses, extract_trianglemesh

depth_scale = 1000.0 # 1000.0
depth_min = 0.1 # 0.1
depth_max = 10.0 # 3.0
odometry_distance_thr = 0.07 # 0.07
trunc_voxel_multiplier = 8.0 # 8.0
voxel_size = 0.001 # 0.001
block_count = 10000 # 40000

def slam(depth_file_names, color_file_names, intrinsic):
    n_files = len(color_file_names)
    device = o3d.core.Device('CUDA:0')

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(voxel_size, 16,
                                       block_count, T_frame_to_model,
                                       device)
    depth_ref = o3d.t.geometry.Image((depth_file_names[0]))

    print(depth_ref.rows, 'rows')
    print(depth_ref.columns, 'columns')

    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.columns, depth_ref.rows, 
                                             intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.columns, depth_ref.rows, intrinsic,device)

    poses = []

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.geometry.Image(depth_file_names[i]).to(device)
        color = o3d.t.geometry.Image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        raycast_frame.set_data_from_image('depth', depth)
        raycast_frame.set_data_from_image('color', color)


        if i > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                depth_scale,
                                                depth_max,
                                                odometry_distance_thr)
            
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, depth_scale, depth_max,
                        trunc_voxel_multiplier)
        model.synthesize_model_frame(raycast_frame, depth_scale,
                                     depth_min, depth_max,
                                     trunc_voxel_multiplier, False)
        stop = time.time()
        print('{:04d}/{:04d} slam takes {:.4}s'.format(i, n_files,
                                                       stop - start))

    return model, model.voxel_grid, poses

if __name__ == '__main__':
    print(o3d.t.io.RealSenseSensor.list_devices())
    camera = PyRealSenseCamera()
    # camera.capture()
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

    camera_intrinsics = o3d.core.Tensor(camera_intrinsics.intrinsic_matrix,
                               o3d.core.Dtype.Float64)
    
    rgb_list = []
    depth_list = []

    camera.capture()

    for i in range(30):
        print('capture')
        camera.capture()
        
        rgb_image = camera.get_color()[:, :, :3].astype(np.uint8) / 255
        depth_image = camera.get_depth() #/ 1000.

        if i !=0:
            #depth_image = np.where(depth_image>=3.0, 0, depth_image)
            plt.imshow(rgb_image)
            plt.show()
            plt.imshow(depth_image)
            plt.show()

            
            rgb_image = rgb_image.astype('float32')
            depth_image = depth_image.astype('float32')

            rgb_list.append(rgb_image)
            depth_list.append(depth_image)
            # time.sleep(0.)
    
    model, volume, poses = slam(depth_file_names=depth_list, color_file_names=rgb_list, intrinsic=camera_intrinsics)
    pcd = model.extract_pointcloud().to_legacy()
    print(pcd)
    o3d.visualization.draw_geometries([pcd])