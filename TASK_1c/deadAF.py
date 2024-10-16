import cv2
import numpy as np
import argparse
import os

def detect_aruco_markers(image):
    # Try different ArUco dictionaries
    aruco_dicts = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_7X7_50
    ]
    
    for dict_type in aruco_dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(image)
        if ids is not None and len(ids) > 0:
            print(f"Detected markers using dictionary: {dict_type}")
            return corners, ids
    
    print("No markers detected with any of the tried dictionaries.")
    return [], None

# ... [keep other functions unchanged] ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to input image")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: The file '{args.image}' does not exist.")
        return

    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Unable to read the image file '{args.image}'. Please check if it's a valid image file.")
        return

    # Save a copy of the input image for debugging
    cv2.imwrite('debug_input.jpg', image)
    print(f"Saved input image as 'debug_input.jpg' for inspection.")

    corners, ids = detect_aruco_markers(image)
    
    if ids is None or len(corners) != 4:
        print(f"Error: Exactly 4 ArUco markers are required. Detected: {len(corners) if ids is not None else 0}")
        
        # Draw detected markers (if any) on the image for debugging
        debug_image = image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
        cv2.imwrite('debug_markers.jpg', debug_image)
        print("Saved debug image with detected markers as 'debug_markers.jpg'")
        return
    
    # ... [rest of the main function remains unchanged] ...

if __name__ == "__main__":
    main()