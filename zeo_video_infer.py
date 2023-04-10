import cv2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
import torch
import matplotlib.pyplot as plt
from azure_kinect import PyAzureKinectCamera
import open3d as o3d
import numpy as np

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

camera = PyAzureKinectCamera(resolution='720')
camera.capture()
rgb_shape = camera.get_color().shape
intrinsic_matrix = camera.get_color_intrinsic_matrix()
width = rgb_shape[1]
height = rgb_shape[0]
fx = intrinsic_matrix[0, 0]
fy = intrinsic_matrix[1, 1]
cx = intrinsic_matrix[0, 2]
cy = intrinsic_matrix[1, 2]
    
while True:
    camera.capture()
    frame = camera.get_color()
    print('frame shape', frame.shape)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raw_rgb = rgb.copy()

    depth_numpy = zoe.infer_pil(rgb)  # as numpy
    print(np.mean(depth_numpy))
    depth_numpy *= 100.
    print('depth shape', depth_numpy.shape)

    # convert rgb image to open3d depth map
    rgb = rgb.astype('uint8')
    rgb = o3d.geometry.Image(rgb)

    # depth image scaling
    depth_image = depth_numpy.astype('uint16')
    
    # convert depth image to open3d depth map
    depth_image = o3d.geometry.Image(depth_image)
    
    # convert to rgbd image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                    depth_image,
                                                                    convert_rgb_to_intensity=False)

    # Create Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                        height=height,
                                                        fx=fx,
                                                        fy=fy,
                                                        cx=cx,
                                                        cy=cy)
    
    # rgbd image convert to pointcloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)
    # pcd.voxel_down_sample(0.1)

    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # pcd = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries([pcd])

    # Colorize output
    
    # colored = colorize(depth_numpy)
    plt.imshow(depth_numpy)
    plt.show()

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1000.))

    rows = 1
    cols = 3
    fig = plt.figure()
    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(raw_rgb, cmap='plasma', vmin=0.0, vmax=1.0)
    ax0.set_title('Original RGB')
    ax0.axis("off")

    ax0 = fig.add_subplot(rows, cols, 2)
    ax0.imshow(depth_numpy, cmap='plasma', vmin=0.0, vmax=1000)
    ax0.set_title('Pred Depth')
    ax0.axis("off")


    fig.subplots_adjust(right=1.0)
    cbar_ax = fig.add_axes([0.82, 0.35, 0.02, 0.3])
    fig.colorbar(sm, cax=cbar_ax)

    plt.show()
