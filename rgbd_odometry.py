import open3d as o3d
import time
import numpy as np
import cv2
from azure_kinect import PyAzureKinectCamera

if __name__ == "__main__":
    camera = PyAzureKinectCamera(resolution='720')
    camera.capture()
    intrinsic_matrix = camera.get_color_intrinsic_matrix()
    rgb_shape = camera.get_color().shape

    width = rgb_shape[1]
    height = rgb_shape[0]
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

    pcds = []
    rgb_images = []
    depth_images = []

    # Capture
    rgb = camera.get_color()

    # select roi
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    # mask = np.expand_dims(mask, axis=-1)
    
    cv2.destroyAllWindows()

    time.sleep(0.5)

    # Load RGB-D images and extract features
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    depth_scale = 0.001  # depth scale factor (converts mm to meters)

    depth_images = []  # list of depth images
    rgb_images = []  # list of RGB images
    descriptors = []  # list of feature descriptors
    keypoints = []  # list of feature keypoints

    while cv2.waitKey(1000) != ord('q'):
        print('capture!')
        camera.capture()
        rgb = camera.get_color()
        depth = camera.get_transformed_depth()

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)

        object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        roi_rgb = rgb.copy()[y:y+h, x:x+w]

        # 크로마키
        hsv = cv2.cvtColor(roi_rgb.copy(), cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (37, 109, 0), (70, 255, 255)) # 영상, 최솟값, 최댓값
        green_mask = cv2.bitwise_not(green_mask)

        object_mask[y:y+h, x:x+w] = green_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        object_mask = cv2.erode(object_mask, kernel, iterations=1)

        object_mask = (object_mask / 255.).astype(np.uint16)
        
        depth *= object_mask.astype(np.uint16)
        rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)
        cv2.imshow('test', rgb)

        gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Convert the depth image to values in the range [0, 255]
        depth_image = (depth / 1000.0) * 255.0
        depth_image = np.round(depth_image).astype(np.uint8)

        kp, des = orb.detectAndCompute(gray_image, None)
        keypoints.append(kp)
        descriptors.append(des)

        rgb_images.append(rgb)
        depth_images.append(depth_image)

        
    # Estimate camera poses using RGB-D images and feature matching
    num_images = len(rgb_images)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_poses = [np.eye(4)]
    reference_pose = np.eye(4)  # initialize the reference pose
    for i in range(1, num_images):
        # Estimate camera pose from RGB-D pair
        depth_image1 = depth_images[i-1]
        depth_image2 = depth_images[i]
        rgb_image1 = rgb_images[i-1]
        rgb_image2 = rgb_images[i]
        kp1, des1 = keypoints[i-1], descriptors[i-1]
        kp2, des2 = keypoints[i], descriptors[i]

        matches = matcher.match(des1, des2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        pts1 = np.expand_dims(pts1, axis=1)
        pts2 = np.expand_dims(pts2, axis=1)

        E, _ = cv2.findEssentialMat(pts2, pts1, camera_matrix, cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, camera_matrix)

        # incremental camera pose를 사용하여 reference pose 업데이트
        T = np.hstack((R, t))
        T = np.vstack((T, [0, 0, 0, 1]))
        pose = np.dot(reference_pose, T)
        camera_poses.append(pose)

        # 다음차례 연산을 위한 reference_pose 업데이트
        reference_pose = pose
        print(pose)


    pcds = []
    for i in range(len(rgb_images)):
        print('save pointclouds {0}'.format(i))
        rgb_image = rgb_images[i]
        depth_image = depth_images[i]
        
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

        test_rgbd_image = np.asarray(rgbd_image)

        print('rgbd shape', test_rgbd_image.shape)

        # rgbd image convert to pointcloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)
        pcd.transform(camera_poses[i])
        pcds.append(pcd)


    o3d.visualization.draw_geometries(pcds)