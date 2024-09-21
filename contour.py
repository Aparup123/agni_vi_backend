import cv2
import numpy as np

# Read the image
image = cv2.imread('PSR-crater-detection-main\output\CLAHE_Image.jpg')

# Apply Gaussian Blur to smooth the image and reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Convert the image to grayscale
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to binary image using Otsu's thresholding
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detect the contours using cv2.CHAIN_APPROX_SIMPLE to simplify the contours
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Visualize the data structure
print("Total contours detected: {}".format(len(contours)))

# Define a minimum area threshold to filter out smaller contours
min_area = 1000  # You can adjust this value depending on what 'big' means in your case

# Copy the original image for drawing contours
image_copy = image.copy()

# Set the color to yellow for all contours
color = (0, 255, 255)  # Yellow color in BGR

# Loop through contours and only draw those with an area greater than min_area
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:  # Filter based on contour area
        filtered_contours.append(contour)
        image_copy = cv2.drawContours(image_copy, [contour], -1, color, thickness=2, lineType=cv2.LINE_AA)

# Visualize the results
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Drawn Contours (Large Ones)', image_copy)
cv2.imshow('Binary Image', binary)

# Output the number of filtered contours
print("Contours larger than {} pixels: {}".format(min_area, len(filtered_contours)))

cv2.waitKey(0)
cv2.destroyAllWindows()
