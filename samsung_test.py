from pyrealsense import PyRealSenseCamera
import numpy as np
import open3d as o3d
import cv2
from global_registration import registerLocalCloud
import copy

def create_point_cloud(rgb_image, depth_image, camera_intrinsics):
    # rgb image scaling 
    rgb_image = rgb_image.astype('uint8')

    # depth image scaling
    depth_image = depth_image.astype('uint16')


    rgb_image = o3d.geometry.Image(rgb_image)
    depth_image = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics)

    return point_cloud


if __name__ == '__main__':
    transform_list = [0.2]
    voxel_size = 0.001
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1

    camera = PyRealSenseCamera()
    camera.capture()
    print('depth scale =', camera.get_depth_scale())
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
    
    rgb_list = []
    depth_list = []
    pcds = []

    while True:
        key = cv2.waitKey(1000)
        camera.capture()
        rgb_image = camera.get_color()
        depth_image = camera.get_depth()

        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        
        if key == ord('d'):
            print('capture')
            rgb_list.append(rgb_image)
            depth_list.append(depth_image)

            point_cloud = create_point_cloud(rgb_image, depth_image, camera_intrinsics)
            pcds.append(point_cloud)
        
        cv2.imshow('test', rgb_image)


    for i in range(len(pcds)-1):
        print('transform')
        transformation_matrix = np.identity(4)
        transformation_matrix[0, 3] = transform_list[i] 
        pcds[i+1] = pcds[i+1].transform(transformation_matrix)


    o3d.visualization.draw_geometries(pcds)

    cloud_base = pcds[0]

    cloud1 = copy.deepcopy(cloud_base)

    detectTransLoop = np.identity(4)
    posWorldTrans = np.identity(4)

    for cloud2 in pcds[1:]:
        posLocalTrans = registerLocalCloud(cloud1, cloud2)

        detectTransLoop = np.dot(posLocalTrans, detectTransLoop)

        posWorldTrans =  np.dot(posWorldTrans, posLocalTrans)

        cloud1 = copy.deepcopy(cloud2)
        cloud2.transform(posWorldTrans)
        
        cloud_base = cloud_base + cloud2

    o3d.visualization.draw_geometries([cloud_base])