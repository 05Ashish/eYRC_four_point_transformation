import cv2
import numpy as np
import argparse
from imutils import contours
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler

# Load an example image with ArUco markers
image = cv2.imread('four_point/input.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not read the image.")
else:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    if ids is not None:
        print(f"Detected {len(corners)} markers with IDs: {ids.flatten()}")
    else:
        print("No markers detected.")

def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Adjust detector parameters for better detection
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    
    print(f"Detected corners: {corners}")
    print(f"Detected IDs: {ids}")
    print(f"Rejected candidates: {rejected}")
    
    return corners, ids


'''def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()  # Use this to create the parameters
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)  # Use detectMarkers directly
    return corners, ids
'''
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_obstacles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # Find contours
    contours = measure.find_contours(thresh, level=0.5)
    obstacles = []
    total_area = 0

    for contour in contours:
        area = cv2.contourArea(np.array(contour).astype(np.int32))
        if area > 100:  # Area threshold
            obstacles.append(contour)
            total_area += area
            
    return obstacles, total_area

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to input image")
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Detect ArUco markers
    corners, ids = detect_aruco_markers(image)

    if ids is None:
        print("Error: No ArUco markers detected.")
        return

    print(f"Detected {len(corners)} ArUco markers.")
    print(f"Marker IDs: {ids.flatten()}")

    if len(corners) != 4:
        print(f"Error: Exactly 4 ArUco markers are required, but {len(corners)} were detected.")
        return

    # Extract corners and apply perspective transform
    pts = np.array([corner[0][0] for corner in corners])
    warped = four_point_transform(image, pts)

    # Find obstacles
    obstacles, total_area = find_obstacles(warped)

    # Draw obstacles on the warped image
    for obstacle in obstacles:
        cv2.drawContours(warped, [np.array(obstacle).astype(np.int32)], -1, (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite("output_image.jpg", warped)

    # Generate the output text file
    with open("output.txt", "w") as f:
        f.write(f"List of ArUco's detected: {', '.join(map(str, ids.flatten()))}\n")
        f.write(f"No of Obstacles: {len(obstacles)}\n")
        f.write(f"Total Area covered by obstacles: {total_area}\n")

    print("Processing complete. Check output_image.jpg and output.txt for results.")

if __name__ == "__main__":
    main()
