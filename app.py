from flask import Flask,request, send_file, jsonify
from flask_cors import CORS, cross_origin
import os
import time 

# packages for ML
import cv2
from ultralytics import YOLO
import numpy as np
import shutil
import os

app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/')
def root():
    name=request.json['name']
    data={"name":name, "size":len(name)}
    print(data)
    return data

@app.route('/getImageUrls')
def getImageUrls():
    urls=["im1.jpeg", "im2.jpeg"]
    return urls

@app.route('/getImage/<image_file>')
def getImage(image_file):
    print(os.getcwd())
    image_file_path=os.path.join(os.getcwd(),'output', image_file)
    return send_file(image_file_path)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    print(request.files['file'])
    input_image=request.files['file']
    file_extension=os.path.splitext(input_image.filename)[1]
    
    save_path=(os.path.join(os.getcwd(),'input_images',f'input_file{file_extension}'))
    input_image.save(save_path)
    


        # Ensure output directory exists
    if not os.path.exists('output'):
        os.makedirs('output')

    # Step 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #/////////////////////////////////#
    #/////////////////////////////////#
    image = cv2.imread(save_path)
    #/////////////////////////////////#
    #/////////////////////////////////#

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    clahe_image = clahe.apply(image_bw)

    # Save the CLAHE image
    cv2.imwrite('output/clahe_image.jpg', clahe_image)
    print("CLAHE image saved.")

    # Step 2: Apply Non-Local Means Denoising
    # Load the CLAHE image
    denoise_input_image = clahe_image
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(denoise_input_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Save the denoised image
    cv2.imwrite('output/denoised_image.jpg', denoised_image)
    print("Denoised image saved.")

    # Step 3: Detect Contours on the Denoised Image
    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    # Convert the image to binary using Otsu's thresholding
    ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect the contours using cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw only large contours
    min_area = 1000  # Filter small contours based on the area
    image_copy = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)  # Convert denoised image back to BGR for contour drawing
    color = (0 , 255 , 255)  # Yellow color in BGR

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(image_copy, [contour], -1, color, thickness=2, lineType=cv2.LINE_AA)

    # Save contour-detected image
    cv2.imwrite('output/contour_image.jpg', image_copy)
    print("Contour image saved.")

    # Step 4: Apply Sharpening on the Contour Image
    # Gaussian kernel for sharpening
    gaussian_blur = cv2.GaussianBlur(denoised_image, (7, 7), 2)

    # Sharpen the image using addWeighted()
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, gaussian_blur, -0.5, 0)

    # Save the sharpened image
    cv2.imwrite('output/sharpened_image.jpg', sharpened_image)
    print("Sharpened image saved.")


    # Display all the results
    #cv2.imshow('CLAHE Image', clahe_image)
    #cv2.imshow('Denoised Image', denoised_image)
    #cv2.imshow('Contour Image', image_copy)
    #cv2.imshow('Sharpened Image', sharpened_image)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Delete the 'runs' directory
    runs_dir = 'runs'
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)

    # Load a pretrained YOLOv8n model
    model = YOLO('best.pt')

    # Run inference on 'crater1.jpg' with specified arguments
    results = model.predict("output/clahe_image.jpg", save=True, imgsz=320, conf=0.25, show_labels=False)

    # Define paths
    source_file = 'runs/detect/predict/clahe_image.jpg'
    destination_dir = 'output/'
    new_filename = 'final.jpg'  # Rename the file to 'new_name.jpg'
    destination_file = os.path.join(destination_dir, new_filename)

    # Ensure the output directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the file
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)

    time.sleep(1)

    return {
        "clahe":"clahe_image.jpg",
        "contour":"contour_image.jpg",
        "denoised":"denoised_image.jpg",
        "sharpened":"sharpened_image.jpg",
        "final":"final.jpg"
        }, 200

if __name__=='__main__':
    app.run(debug=True, port=5000)