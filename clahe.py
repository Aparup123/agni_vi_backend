import cv2
import numpy as np

image=cv2.imread('crater2.jpg')
image_bw=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe=cv2.createCLAHE(clipLimit=5)
final_image=clahe.apply(image_bw)

normal_hist=cv2.equalizeHist(image_bw)

cv2.imshow('ordinary',image)
cv2.imshow('CLAHE',final_image)
cv2.imshow('HE',normal_hist)

cv2.imwrite('clahe_image2.jpg', final_image)
print("Clahe image saved.")

cv2.waitKey(0)
cv2.destroyAllWindows()