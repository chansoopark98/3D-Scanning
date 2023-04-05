import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

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

# Configure the streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the stream
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

vis = o3d.visualization.Visualizer()
vis.create_window()

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)

        # Get the aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.expand_dims(np.asanyarray(depth_frame.get_data()), axis=-1).astype('uint16')
        depth_image = np.asanyarray(depth_frame.get_data()).astype('uint16')
        color_image = np.asanyarray(color_frame.get_data()).astype('uint8')

        # Get camera intrinsics and extrinsics
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        extrinsics = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        
        K = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
         ])
        
        width = color_image.shape[1]
        height = color_image.shape[0]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Create Open3D camera intrinsic object
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                            height=height,
                                                            fx=fx,
                                                            fy=fy,
                                                            cx=cx,
                                                            cy=cy)
            
        
        pcd = convert_to_pcd(rgb_image=color_image, depth_image=depth_image, camera_intrinsics=camera_intrinsics)
        
        # Add the current point cloud to the visualizer
        
        vis.add_geometry(pcd)

        # Update the visualizer
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd)  
finally:
    # Stop streaming
    pipeline.stop()
