import pyrealsense2 as rs
import numpy as np
import cv2
import math

# Configuration
show_camera = True # Display the live feed
MIN_CONTOUR_AREA = 10 # The smallest object  that could potentially be a can, in pixel area, make bigger to filter more noise
DEPTH_COLORMAP_CONTRAST_SCALE = 0.06 # Lower = less contrast, less transitions, and lower max distance

# Globals
HEIGHT = 480
WIDTH = 640
top = np.zeros((int((4.25/8.0)*HEIGHT), WIDTH), dtype="uint8")
bottom = np.ones((int(HEIGHT - int((4.25/8.0)*HEIGHT)), WIDTH), dtype="uint8")
line_follower_mask = np.vstack((top, bottom))

# Color Segmentation Mask
# TODO: Combining multiple filters will enhance detection
low_1 = (119, 109, 130) # Soda Red
high_1 = (124, 255, 255) # Soda Red
#low_2 = (103, 121, 0) 
#high_2 = (111, 161, 255)
#low_3 = (0, 0, 0)
#high_3 = (0, 0, 0)


#The following collection of pixel locations and corresponding relative
#ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
PTS_IMAGE_PLANE = [[440, 440],
                   [281, 338],
                   [253, 315],
                   [575, 357]] # 03/15/2024 Measurements
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
PTS_GROUND_PLANE = [[32, -5],
                    [68, 7],
                    [91, 13],
                    [56, -22]] # 03/15/2024 Measurements
######################################################

METERS_PER_INCH = 0.0254

#Initialize data into a homography matrix
np_pts_ground = np.array(PTS_GROUND_PLANE)
np_pts_ground = np_pts_ground * METERS_PER_INCH
np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

np_pts_image = np.array(PTS_IMAGE_PLANE)
np_pts_image = np_pts_image * 1.0
np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])
homography, err = cv2.findHomography(np_pts_image, np_pts_ground)

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
            blurred_image = cv2.bitwise_and(blurred_image, blurred_image, mask=line_follower_mask)

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
                    print('bounding box:', bounding_box)
                    depth = depth_image[int((y+y+h)/2), int((x+x+w)/2)]
                    if(depth!=0):
                        print('depth is:', depth)
                    
                    # Homography Transform
                    u = (x+x+w)/2
                    v = (y+y+h)/2
                    homogeneous_point = np.array([[u], [v], [1]])
                    xy = np.dot(homography, homogeneous_point)
                    scaling_factor = 1.0 / xy[2, 0]
                    homogeneous_xy = xy * scaling_factor

        
                    x = homogeneous_xy[0, 0] # Real World X
                    y = homogeneous_xy[1, 0] # Real World Y
                    can_distance = math.sqrt(x**2 + y**2)  # Distance to Can

                    
                    """
                    if threshold reached:  --> can_distance and 1/3 *y < y < 2/3*y
                        call Ben function()
                    """
                    
                    # Visualize Bounding Box
                    if show_camera:
                        cv2.rectangle(color_image,bounding_box[0],bounding_box[1],(0,255,0),2)
                        if(depth!=0):
                            pass
                            cv2.putText(color_image, f"Depth: {round(depth, 2)} Distance: {round(can_distance,2)}mm Bounds:({round(x,2)},{round(y,2)})", (bounding_box[0][0], bounding_box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
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