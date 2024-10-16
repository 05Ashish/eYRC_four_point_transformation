import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from imutils import resize
from kneed import KneeLocator
from sklearn.cluster import KMeans

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # Example processing (you can replace this with your actual processing steps)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize(gray_image, width=500)

    # Here, you might want to implement K-Means clustering or any other processing
    # Example: Reshape and apply K-Means
    pixel_values = resized_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Implement K-Means
    k = 3  # Choose the number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_values)

    # Generate output (you may change this part according to your task)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Save results to a text file (modify the file path as necessary)
    output_filename = 'output_results.txt'
    with open(output_filename, 'w') as f:
        f.write("Cluster Centers:\n")
        for centroid in centroids:
            f.write(f"{centroid[0]}\n")
        f.write("\nLabels:\n")
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"Results saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Image Processing Task 1C')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    
    process_image(args.image)

if __name__ == '__main__':
    main()
