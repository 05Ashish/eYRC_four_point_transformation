import cv2
import numpy as np
import argparse

def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(image)
    
    # Debugging: print detected corners and ids
    print(f"Detected Corners: {corners}")
    print(f"Detected IDs: {ids}")
    
    return corners, ids

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

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

def detect_obstacles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Use RETR_EXTERNAL to only get the outer contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust this threshold as needed
            obstacles.append(contour)
            total_area += area

    # Debugging: print number of obstacles and total area
    print(f"Obstacles detected: {len(obstacles)}, Total area: {total_area}")
    
    return obstacles, total_area

def main(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return
    else:
        print("Image loaded successfully.")

    # Step 2: Detect ArUco markers
    corners, ids = detect_aruco_markers(image)
    
    if corners and len(corners) == 4:
        pts = np.array([corner[0][0] for corner in corners])
        warped = four_point_transform(image, pts)
        print("Perspective transform applied successfully.")
    else:
        print(f"Could not detect all four ArUco markers. Detected {len(corners) if corners else 0} markers.")
        warped = image  # Bypass transformation if ArUco markers not fully detected

    # Step 3: Detect obstacles in the (possibly transformed) image
    obstacles, total_area = detect_obstacles(warped)
    
    # Step 4: Draw green outlines for detected obstacles
    for obstacle in obstacles:
        cv2.drawContours(warped, [obstacle], 0, (0, 255, 0), 2)

    # Step 5: Save the processed image
    output_image_path = "output_image.jpg"
    if cv2.imwrite(output_image_path, warped):
        print(f"Output image saved as {output_image_path}")
    else:
        print("Error: Failed to save the output image.")

    # Step 6: Save obstacle details in a text file
    output_text_path = "obstacles.txt"
    try:
        with open(output_text_path, "w") as f:
            # Convert ArUco IDs to Python int types
            if ids is not None:
                aruco_ids = [int(id[0]) for id in ids]  # Convert from np.int32 to Python int
                f.write(f"ArUco IDs: {aruco_ids}\n")
            else:
                f.write("No ArUco IDs detected.\n")
            f.write(f"Obstacles: {len(obstacles)}\n")
            f.write(f"Total Area: {total_area:.1f}\n")
        print(f"Obstacle details saved in {output_text_path}")
    except Exception as e:
        print(f"Error writing to {output_text_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ArUco markers and obstacles in an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    main(args.image)
