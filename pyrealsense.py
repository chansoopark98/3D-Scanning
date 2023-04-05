import pyrealsense2 as rs
import numpy as np

class PyRealSenseCamera(object):
    def __init__(self):
        # Configure the streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the stream
        self.profile = self.pipeline.start(self.config)

        # Get the depth sensor's depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Create an align object
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.color_frame = None

    def capture(self):
        self.frames = self.pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        self.aligned_frames = self.align.process(self.frames)

    def get_color(self):
        self.color_frame = self.aligned_frames.get_color_frame()
        self.color_image = np.asanyarray(self.color_frame.get_data()).astype('uint8')
        return self.color_image

    def get_depth(self):
        self.depth_frame = self.aligned_frames.get_depth_frame()
        self.depth_image = np.asanyarray(self.depth_frame.get_data()).astype('uint16')
        return self.depth_image
    
    def get_camera_intrinsic(self):
        if self.color_frame == None:
            self.capture()
            _ = self.get_color()

        color_intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics
        
        K = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        return K