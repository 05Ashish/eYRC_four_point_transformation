import cv2
import numpy as np
import argparse
def detect_aruco_markers(image):
    # For OpenCV 4.10.0
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(image)
    
    return corners, ids

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

def detect_obstacles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold as needed
            obstacles.append(contour)
            total_area += area
    return obstacles, total_area

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return
    
    corners, ids = detect_aruco_markers(image)
    
    if corners is not None and len(corners) == 4:
        pts = np.array([corner[0][0] for corner in corners])
        warped = four_point_transform(image, pts)
        
        obstacles, total_area = detect_obstacles(warped)
        
        for obstacle in obstacles:
            cv2.drawContours(warped, [obstacle], 0, (0, 255, 0), 2)
        
        cv2.imwrite("output_image.jpg", warped)
        
        with open("obstacles.txt", "w") as f:
            f.write(f"ArUco ID: {[id[0] for id in ids]}\n")
            f.write(f"Obstacles: {len(obstacles)}\n")
            f.write(f"Area: {total_area:.1f}\n")
        
        print("Processing complete. Check output_image.jpg and obstacles.txt for results.")
    else:
        print(f"Could not detect all four ArUco markers. Detected {len(corners) if corners is not None else 0} markers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ArUco markers and obstacles in an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()
    
    main(args.image)