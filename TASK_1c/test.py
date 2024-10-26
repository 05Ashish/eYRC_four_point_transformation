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
    """
    Detect obstacles in the image and calculate their areas using pixel counts
    and relative scale based on the image dimensions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Use RETR_EXTERNAL to only get the outer contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    total_area = 0
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    
    for contour in contours:
        # Calculate the actual contour area
        pixel_area = cv2.contourArea(contour)
        
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Changed from np.int0 to np.int32
        
        # Calculate rectangle area
        rect_width = np.linalg.norm(box[0] - box[1])
        rect_height = np.linalg.norm(box[1] - box[2])
        rect_area = rect_width * rect_height
        
        # Use the larger of the two areas to avoid underestimation
        area = max(pixel_area, rect_area)
        
        # Convert to percentage of total image area
        relative_area = (area / image_area) * 100
        
        if relative_area > 0.1:  # Filter out tiny obstacles (adjust threshold as needed)
            obstacles.append({
                'contour': contour,
                'pixel_area': pixel_area,
                'rect_area': rect_area,
                'relative_area': relative_area,
                'bounding_box': box
            })
            total_area += area

    # Debugging: print detailed information about detected obstacles
    print(f"Obstacles detected: {len(obstacles)}")
    print(f"Total pixel area: {total_area}")
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Total image area: {image_area}")
    print(f"Relative total obstacle area: {(total_area/image_area)*100:.2f}%")
    
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
    
    # Step 4: Draw both the contour and minimum area rectangle for detected obstacles
    output_image = warped.copy()
    for obstacle in obstacles:
        # Draw the contour in green
        cv2.drawContours(output_image, [obstacle['contour']], 0, (0, 255, 0), 2)
        # Draw the minimum area rectangle in blue
        cv2.drawContours(output_image, [obstacle['bounding_box']], 0, (255, 0, 0), 2)
        
        # Add area information near the obstacle
        moment = cv2.moments(obstacle['contour'])
        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
            text = f"{obstacle['relative_area']:.1f}%"
            cv2.putText(output_image, text, (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Step 5: Save the processed image
    output_image_path = "output_image.jpg"
    if cv2.imwrite(output_image_path, output_image):
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
            
            f.write(f"\nDetailed Obstacle Information:\n")
            f.write(f"Number of obstacles: {len(obstacles)}\n")
            f.write(f"Total obstacle area: {total_area:.1f} pixels\n")
            f.write(f"Image dimensions: {warped.shape[1]}x{warped.shape[0]}\n")
            f.write(f"Total image area: {warped.shape[1]*warped.shape[0]} pixels\n")
            f.write(f"Relative total obstacle area: {(total_area/(warped.shape[1]*warped.shape[0]))*100:.2f}%\n\n")
            
            for i, obstacle in enumerate(obstacles, 1):
                f.write(f"\nObstacle {i}:\n")
                f.write(f"  Contour Area: {obstacle['pixel_area']:.1f} pixels\n")
                f.write(f"  Rectangle Area: {obstacle['rect_area']:.1f} pixels\n")
                f.write(f"  Relative Area: {obstacle['relative_area']:.2f}%\n")
                
        print(f"Obstacle details saved in {output_text_path}")
    except Exception as e:
        print(f"Error writing to {output_text_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ArUco markers and obstacles in an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    main(args.image)