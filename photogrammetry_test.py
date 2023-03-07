import cv2
import numpy as np
import open3d as o3d

# 1080p fx 914.9718627929688 fy 914.7095947265625 cx 956.8119506835938 cy 551.3303833007812
fx = 914.9718627929688
fy = 914.7095947265625
cx = 956.8119506835938
cy = 551.3303833007812

# Step 1: Load images

# Load images
images = []
for i in range(1, 27):
    img_path = f'./rgb_images/image_{i}.png'
    img = cv2.imread(img_path)
    images.append(img)

print('num images : {0}'.format(len(images)))
# Step 2: Image preprocessing

# Convert images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

sift = cv2.SIFT_create()

kp_list = []
des_list = []
for gray_image in gray_images:
    kp, des = sift.detectAndCompute(gray_image, None)
    kp_list.append(kp)
    des_list.append(des)

# Step 3: Structure from motion
matcher = cv2.BFMatcher()

matches_list = []
for i in range(len(des_list)-1):
    matches = matcher.knnMatch(des_list[i], des_list[i+1], k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            
            good_matches.append(m)
    matches_list.append(good_matches)


test_matches = matcher.match(des_list[0], des_list[1])
res = cv2.drawMatches(images[0], kp_list[0], images[1], kp_list[1], test_matches, None, \
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('BFMatcher + ORB', res)
cv2.waitKey()

# Estimate camera poses using the matches
print('Estimate camera poses using the matches')
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Estimate camera pose for the first image
pose_list = [np.eye(4)]
for i, matches in enumerate(matches_list):
    src_pts = np.float32([kp_list[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    E, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)
    pose = np.hstack((R, t))
    pose = np.vstack((pose, np.array([0, 0, 0, 1])))
    pose_list.append(np.dot(pose_list[-1], pose))

print(pose_list)


point_cloud = []
for i in range(len(matches_list)):
    src_pts = np.float32([kp_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
    P1 = np.dot(camera_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(camera_matrix, pose_list[i+1][:3, :])
    points_4d = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
    
    point_cloud.append(points_3d.reshape(-1, 3))



output_points = np.vstack(point_cloud)
# output_points = np.reshape(output_points, [-1, 3])

# print(output_points.mean())
# # Convert 3D points to Open3D point cloud
# print('Convert 3D points to Open3D point cloud')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points)

# Visualize point cloud
print('Visualize point cloud')
o3d.visualization.draw_geometries([pcd])