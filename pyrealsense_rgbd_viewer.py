import pyrealsense2 as rs
import numpy as np
import cv2

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

# Create a point cloud object
pc = rs.pointcloud()

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
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get camera intrinsics and extrinsics
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        extrinsics = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Generate point cloud
        pc.map_to(color_frame)
        pointcloud = pc.calculate(depth_frame)
        points = np.asanyarray(pointcloud.get_vertices()).view(np.float32).reshape(-1, 3)

        # Visualize the data
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
