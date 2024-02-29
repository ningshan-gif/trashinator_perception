import pyrealsense2 as rs
import numpy as np
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils


# CONFIGURATION
show_camera = True

# Load the pre-trained trash detection model
model = InceptionV3(weights='imagenet', include_top=True)

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

        # Preprocess color frame
        processed_color_image = cv2.resize(color_image, (299, 299))  # InceptionV3 expects input shape (299, 299)
        processed_color_image = img_to_array(processed_color_image)
        processed_color_image = np.expand_dims(processed_color_image, axis=0)
        processed_color_image = preprocess_input(processed_color_image)

        # Perform object detection
        prediction = model.predict(processed_color_image)
        label = np.argmax(prediction)
        confidence = prediction[0][label]
        actual_prediction = imagenet_utils.decode_predictions(prediction)
        print("predicted object is:")
        print(actual_prediction[0][0][1])
        print("with accuracy")
        print(actual_prediction[0][0][2]*100)

        # Display the results
        cv2.putText(color_image, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', color_image)
        cv2.waitKey(1)
        """
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

        """
finally:
    # Stop camera streams
    pipeline.stop()