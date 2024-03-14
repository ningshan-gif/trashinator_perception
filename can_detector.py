import pyrealsense2 as rs
import numpy as np
import cv2

# Configuration
show_camera = True # Display the live feed
MIN_CONTOUR_AREA = 300 # The smallest object  that could potentially be a can, in pixel area, make bigger to filter more noise
DEPTH_COLORMAP_CONTRAST_SCALE = 0.06 # Lower = less contrast, less transitions, and lower max distance

# Globals
HEIGHT = 480
WIDTH = 640
top = np.zeros((int((5/8.0)*HEIGHT), WIDTH), dtype="uint8")
middle = np.ones((int((3/8.0)*HEIGHT), WIDTH), dtype="uint8")
bottom = np.zeros(((HEIGHT-top.shape[0]-middle.shape[0]), WIDTH), dtype="uint8")
line_follower_mask = np.vstack((top, middle, bottom))

# Color Segmentation Mask
# TODO: Combining multiple filters will enhance detection
low_1 = (115, 190, 0) # Soda Red
high_1 = (125, 255, 255) # Soda Red
#low_2 = (103, 121, 0) 
#high_2 = (111, 161, 255)
#low_3 = (0, 0, 0)
#high_3 = (0, 0, 0)


def image_print(img):
	"""
	Helper function to print out images, for debugging. img is represented as a list

	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def detect_cans():
    """
    Runs can detection on the Intel Realsense Depth Camera D435 in realtime
    Object detection implemented through color segmentation and filtering

    Prints the pixel coordinate bounding box of the can to the standard output
    """

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
            # Get next camera frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to OpenCV Images
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            #Reduce image noise
            blurred_image = cv2.GaussianBlur(color_image, (3,3), 0)
            blurred_image = cv2.erode(blurred_image, (3,3))
            blurred_image = cv2.dilate(blurred_image, (3,3))

            # Convert to HSV
            image_hsv = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HSV)

            # Build Color Mask
            color_mask1 = cv2.inRange(image_hsv, low_1, high_1)
            #color_mask2 = cv2.inRange(image_hsv, low_2, high_2)
            #color_mask3 = cv2.inRange(image_hsv, low_3, high_3)
            COLOR_MASK = color_mask1
            #COLOR_MASK = cv2.bitwise_or(color_mask1, color_mask2)
            #COLOR_MASK = cv2.bitwise_or(color_mask, color_mask3)

            # Apply coke can color mask
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=COLOR_MASK)

            # Identify Potential Contours
            _, thresholded_image = cv2.threshold(COLOR_MASK, 40, 255, cv2.THRESH_BINARY)
            contours, _  = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                best_contour = max(contours, key=cv2.contourArea) # Choose contour of largest area
                if cv2.contourArea(best_contour) >= MIN_CONTOUR_AREA: # Super small contour --> likely just noise
                    # Build Bounding Box
                    x,y,w,h = cv2.boundingRect(best_contour)
                    bounding_box = ((x,y), (x+w, y+h))
                    print('bounding box' + bounding_box)
                    depth = depth_frame.get_distance(int((x+x+w)/2), int((y+y+h)/2))
                    print('depth is' + depth)
            

            
                    # Visualize Bounding Box
                    if show_camera:
                        cv2.rectangle(color_image,bounding_box[0],bounding_box[1],(0,255,0),2)
            
            # Show live feed 
            if show_camera:
                # Build depth colormap
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=DEPTH_COLORMAP_CONTRAST_SCALE), cv2.COLORMAP_JET)
                
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

if __name__ == "__main__":
    print("Starting can detection...")
    detect_cans()