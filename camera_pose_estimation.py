import cv2
import numpy as np

def camera_pose_estimation(images, K):
    # Feature detection and matching
    sift = cv2.xfeatures2d.SIFT_create()
    kp_list = []
    des_list = []
    bf = cv2.BFMatcher()
    matches_list = []
    for i in range(len(images) - 1):
        kp1, des1 = sift.detectAndCompute(images[i], None)
        kp2, des2 = sift.detectAndCompute(images[i + 1], None)
        kp_list.append(kp1)
        des_list.append(des1)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)
    kp_list.append(kp2)
    des_list.append(des2)

    # Camera pose estimation
    # R_list = [np.eye(4)]
    # t_list = [np.zeros((3, 1))]
    pose_list = [np.eye(4)]
    for i in range(len(matches_list)):
        src_pts = np.float32([kp_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
        # E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
        # _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
        
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.999, 1.0)

        src_pts = src_pts[mask.ravel()==1]
        dst_pts = dst_pts[mask.ravel()==1]


        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = np.reshape(t, [3,])

        print(pose)
        pose_list.append(pose)

    return pose_list

fx = 914.9718627929688
fy = 914.7095947265625
cx = 956.8119506835938
cy = 551.3303833007812
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

images = []
for i in range(1, 27):
    img_path = f'./rgb_images/image_{i}.png'
    img = cv2.imread(img_path)
    images.append(img)

# camera_pose_estimation(images=images, K=camera_matrix)
from modeler import SfM
sfm = SfM('./', False, './test_video.mp4', 3)
sfm.find_structure_from_motion()