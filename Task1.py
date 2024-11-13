import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_grid_lines():
    # Load image in grayscale
    image = cv2.imread("Lines.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Failed to load image.")
        return

    # Apply adaptive thresholding to make grid lines more prominent
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    # Use morphological transformations to remove smaller elements (numbers)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)

    # Detect edges
    edges = cv2.Canny(eroded, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Draw detected lines on a copy of the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the result
    plt.imshow(result_image, cmap='gray')
    plt.title('Detected Grid Lines')
    plt.axis('off')
    plt.show()

# Run the function on the uploaded image path
detect_grid_lines()
