import pyrealsense2 as rs
import numpy as np
import cv2

class PyRealSenseCamera(object):
    def __init__(self):
        # Configure the streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        print(self.config)
        """
        [Open3D INFO]   depth_fps: [100 | 15 | 30 | 300 | 6 | 60 | 90]
        [Open3D INFO]   depth_resolution: [1280,720 | 256,144 | 424,240 | 480,270 | 640,360 | 640,480 | 848,100 | 848,480]
        [Open3D INFO]   depth_format: [RS2_FORMAT_Z16]
        [Open3D INFO]   color_fps: [15 | 30 | 6 | 60]
        [Open3D INFO]   visual_preset: []
        [Open3D INFO]   color_resolution: [1280,720 | 1920,1080 | 320,180 | 320,240 | 424,240 | 640,360 | 640,480 | 848,480 | 960,540]
        [Open3D INFO]   color_format: [RS2_FORMAT_BGR8 | RS2_FORMAT_BGRA8 | RS2_FORMAT_RGB8 | RS2_FORMAT_RGBA8 | RS2_FORMAT_Y16 | RS2_FORMAT_YUYV]
        """
        self.config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 100)
        self.config.enable_stream(rs.stream.color, 1920,1080, rs.format.bgr8, 60)

        # Start the stream
        self.profile = self.pipeline.start(self.config)

        # Get the depth sensor's depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Create an align object
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.color_frame = None
        self.depth_frame = None

        self.sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        self.sensor.set_option(rs.option.exposure, 156.000)

        self.capture()

    def capture(self):
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        self.color_frame = self.aligned_frames.get_color_frame()
        self.depth_frame = self.aligned_frames.get_depth_frame()

    def get_color(self) -> np.ndarray:
        self.color_image = np.asanyarray(self.color_frame.get_data()).astype('uint8')
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        return self.color_image

    def get_depth(self) -> np.ndarray:
        self.depth_image = np.asanyarray(self.depth_frame.get_data()).astype('uint16')
        return self.depth_image
    
    def get_camera_intrinsic(self) -> np.ndarray:
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