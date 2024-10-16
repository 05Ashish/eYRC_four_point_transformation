import cv2
import numpy as np
import argparse
import imutils

# Function to detect ArUco markers
def detect_markers(image):
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    return corners, ids

# Function to apply perspective transform
def apply_perspective_transform(image, corners):
    # Assuming corners contains 4 points
    pts = np.array(corners).reshape(-1, 2)
    width, height = 400, 400  # Desired size of output image
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# Function to find obstacles in the image
def find_obstacles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacle_info = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small contours
            obstacle_info.append(area)

    return len(obstacle_info), sum(obstacle_info)

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)

    # Detect ArUco markers
    corners, ids = detect_markers(image)
    
    # If markers are detected
    if ids is not None and len(corners) == 4:
        warped = apply_perspective_transform(image, corners)
        
        # Find obstacles in the warped image
        num_obstacles, total_area = find_obstacles(warped)

        # Output results
        with open("output.txt", "w") as f:
            f.write(f"Detected ArUco IDs: {ids.flatten().tolist()}\n")
            f.write(f"Number of Obstacles: {num_obstacles}\n")
            f.write(f"Total Area Covered by Obstacles: {total_area}\n")

        # Save output image
        cv2.imwrite("output_image.jpg", warped)
    else:
        print("ArUco markers not found or insufficient markers detected.")

if __name__ == "__main__":
    main()
