import os

# Download data from Google Drive
# url = 'https://drive.google.com/drive/u/1/folders/1Fh1XGIrJHLVLDdeMXCA669O5qVCIaldr'
# output_dir = 'data'

rootdir = os.path.join('data',"fotos hojas bromelias")
print(os.listdir(rootdir))

import matplotlib.pyplot as plt
import numpy as np
import cv2

leafname = 'Brom02'
img_input = cv2.imread(os.path.join(rootdir,f'{leafname}.JPG'))
img_output = cv2.imread(os.path.join(rootdir,f'{leafname}F_desdoblada.jpg'))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
axes[0].set_title('Input Image')
axes[1].imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
axes[1].set_title('Output Image')
plt.show()

def refine_crop(image, padding=10):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for color thresholding
    lower_bound = np.array([20, 30, 30])  # Adjust based on target color
    upper_bound = np.array([80, 255, 255])  # Adjust based on target color

    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Apply padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        cropped_image = image[y:y+h, x:x+w]
    else:
        cropped_image = image  # Return original if no contours are found

    return cropped_image

# Apply the refined crop function
refined_cropped_input = refine_crop(img_input, padding=200)
refined_cropped_output = refine_crop(img_output, padding=300)

# # Plot the original and refined cropped image
# fig, axes = plt.subplots(2, 2, figsize=(10, 5))
# axes[0].imshow(cv2.cvtColor(refined_cropped_input, cv2.COLOR_BGR2RGB))
# axes[0].set_title('Input Image')
# axes[1].imshow(cv2.cvtColor(refined_cropped_output, cv2.COLOR_BGR2RGB))
# axes[1].set_title('Output Image')
# plt.show()
