import cv2
import numpy as np

# Load the image
image = cv2.imread('final_image')

# Check if the image was loaded correctly    # Convert to grayscale (if needed)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

# Save the denoised image
cv2.imwrite('denoised_image.jpg', denoised_image)
print("Denoised image saved as 'denoised_image.jpg'.")

# Display the result in a resizable window
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)
cv2.namedWindow('Denoised Image', cv2.WINDOW_NORMAL)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
