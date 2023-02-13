import matplotlib.pyplot as plt

import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d

import numpy as np
import cv2
import time

from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

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
    shape = None

    capture = k4a.get_capture()
    # rgb = capture.color[:, :, :3]
    raw_pcd = capture.depth_point_cloud
    vis_cv = raw_pcd.copy() / 255
    vis_cv = vis_cv.astype(np.uint8)
    print(raw_pcd.dtype)
    print(raw_pcd.shape)

    crop_x, crop_y, crop_w, crop_h = cv2.selectROI('test', vis_cv)
    cv2.destroyAllWindows()

    while True:
        idx += 1
        if idx == 3:
            break
        capture = k4a.get_capture()
        # rgb = capture.color[:, :, :3]
        capture_pcd = capture.depth_point_cloud
        
        raw_pcd = np.zeros(vis_cv.shape, np.int16)
        raw_pcd[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w] = capture_pcd[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
        raw_pcd = raw_pcd.astype(np.int16)
        plt.imshow(raw_pcd)
        plt.show()

        # shape = raw_pcd.shape
        raw_pcd = np.reshape(raw_pcd, [-1, 3])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(raw_pcd)
        # pcd.voxel_down_sample(voxel_size=0.02)
        
        pcd = raw_pcd
        pcds.append(pcd)

    # ICP
    A = pcds[0]
    dim = 3
    total_time = 0
    N = pcds[0].shape[0]
    output = None
    for i in range(len(pcds)):

        B = np.copy(A)

        print('ICD try {0}'.format(i))
        # Run ICP
        start = time.time()
        T, distances, iterations = icp(B, A, tolerance=1., max_iterations=1)
        total_time += time.time() - start
        print('ICD done {0}, duration {1}'.format(i, total_time))

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T
        output = C

    print(output.shape)
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(output[:, 0:3])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([vis_pcd])

    # Convert the point cloud to a triangle mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud(vis_pcd)

    # Save the triangle mesh to a file
    o3d.io.write_triangle_mesh("mesh.ply", mesh)