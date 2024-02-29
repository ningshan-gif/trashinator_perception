import pyrealsense2 as rs
import numpy as np
import cv2

# CONFIGURATION
show_camera = True

# Get camera pipelines and configuration
pipeline = rs.pipeline()
config = rs.config()

# Configure camera streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Get next color and depth frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get image pixels
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Build depth colormap
        contrast_scale = 0.06 # Lower = less contrast, less transitions, and lower max distance
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=contrast_scale), cv2.COLORMAP_JET)

        if show_camera:
            # Resize depth and color map to matching resolutions
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            # Combine images to one window
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('Trashinator Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Trashinator Camera', images)
            cv2.waitKey(1)

finally:
    # Stop camera streams
    pipeline.stop()