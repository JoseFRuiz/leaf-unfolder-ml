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

def process_images(folded_path, straight_path, output_size=(256, 256), padding=10):
    """
    Processes a pair of images: folded and straightened leaf, crops them using refine_crop, and resizes them.

    Parameters:
        folded_path (str): Path to the folded leaf image.
        straight_path (str): Path to the straightened leaf image.
        output_size (tuple): Target size for resizing (default: 256x256).
        padding (int): Padding applied around the detected leaf.
    """
    # Load images
    folded = cv2.imread(folded_path)
    straight = cv2.imread(straight_path)

    if folded is None or straight is None:
        print("Error: One or both images could not be loaded.")
        return

    # Apply cropping using refine_crop
    cropped_folded = refine_crop(folded, padding=padding)
    cropped_straight = refine_crop(straight, padding=padding)

    # Resize images to 256x256
    resized_folded = cv2.resize(cropped_folded, output_size)
    resized_straight = cv2.resize(cropped_straight, output_size)

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(resized_folded, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Folded Leaf")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(resized_straight, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Straightened Leaf")
    axes[1].axis("off")

    plt.show()

# Example usage
leafname = 'Brom02'
folded_image_path = os.path.join(rootdir,f'{leafname}.JPG')
straight_image_path = os.path.join(rootdir,f'{leafname}F_desdoblada.jpg')
process_images(folded_image_path, straight_image_path, padding=300)

def pad_to_square(image, background_color=(255, 255, 255)):
    """
    Pads the image to make it a square while keeping it centered.

    Parameters:
        image (numpy array): Input image.
        background_color (tuple): Color of the padding (default: white).

    Returns:
        Padded square image.
    """
    h, w, _ = image.shape
    size = max(h, w)  # Determine the square size

    # Create a blank white image of the desired square size
    padded_img = np.full((size, size, 3), background_color, dtype=np.uint8)

    # Calculate the center position
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    # Place the original image in the center
    padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = image

    return padded_img

def process_images(folded_path, straight_path, output_size_folded=(256, 256), output_size_straight=(256, 256), padding=10, display=True):
    """
    Processes a pair of images: folded and straightened leaf, crops them using refine_crop,
    pads them to be squared with a white background, and resizes them independently.

    Parameters:
        folded_path (str): Path to the folded leaf image.
        straight_path (str): Path to the straightened leaf image.
        output_size_folded (tuple): Target size for the folded leaf image.
        output_size_straight (tuple): Target size for the straightened leaf image.
        padding (int): Padding applied around the detected leaf.
    """
    # Load images
    folded = cv2.imread(folded_path)
    straight = cv2.imread(straight_path)

    if folded is None or straight is None:
        print("Error: One or both images could not be loaded.")
        return

    # Apply cropping using refine_crop
    cropped_folded = refine_crop(folded, padding=padding)
    cropped_straight = refine_crop(straight, padding=padding)

    # Pad images to square shape
    squared_folded = pad_to_square(cropped_folded)
    squared_straight = pad_to_square(cropped_straight)

    # Resize images to their respective sizes
    resized_folded = cv2.resize(squared_folded, output_size_folded)
    resized_straight = cv2.resize(squared_straight, output_size_straight)
    resized_folded = cv2.rotate(resized_folded, cv2.ROTATE_90_CLOCKWISE)
    resized_straight = cv2.rotate(resized_straight, cv2.ROTATE_90_CLOCKWISE)

    if display:
      # Display images
      fig, axes = plt.subplots(1, 2, figsize=(8, 4))
      axes[0].imshow(cv2.cvtColor(resized_folded, cv2.COLOR_BGR2RGB))
      axes[0].set_title(f"Folded Leaf ({output_size_folded[0]}x{output_size_folded[1]})")
      axes[0].axis("off")

      axes[1].imshow(cv2.cvtColor(resized_straight, cv2.COLOR_BGR2RGB))
      axes[1].set_title(f"Straightened Leaf ({output_size_straight[0]}x{output_size_straight[1]})")
      axes[1].axis("off")

      plt.show()

    return resized_folded, resized_straight

# Example usage
leafname = 'Brom02'
folded_image_path = os.path.join(rootdir, f"{leafname}.JPG")
straight_image_path = os.path.join(rootdir, f"{leafname}F_desdoblada.jpg")

resized_folded, resized_straight = process_images(folded_image_path, straight_image_path,
                                                  output_size_folded=(128, 128),
                                                  output_size_straight=(256, 256),
                                                  padding=300)


# Pytorch dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class LeafDataset(Dataset):
    def __init__(self, rootdir, leafnames, output_size_folded=(128, 128), output_size_straight=(256, 256), padding=10, transform=None):
        """
        Custom PyTorch dataset to load folded and straightened leaves.

        Parameters:
            rootdir (str): Path to the image folder.
            leafnames (list): List of leaf names (e.g., ['Brom01', 'Brom02']).
            output_size_folded (tuple): Target size for the folded leaf image.
            output_size_straight (tuple): Target size for the straightened leaf image.
            padding (int): Padding applied around the detected leaf.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.rootdir = rootdir
        self.leafnames = leafnames
        self.output_size_folded = output_size_folded
        self.output_size_straight = output_size_straight
        self.padding = padding
        self.transform = transform

    def __len__(self):
        return len(self.leafnames)

    def __getitem__(self, idx):
        leafname = self.leafnames[idx]

        # Define file paths
        folded_path = os.path.join(self.rootdir, f"{leafname}.JPG")
        straight_path = os.path.join(self.rootdir, f"{leafname}F_desdoblada.jpg")

        # Load images
        folded = cv2.imread(folded_path)
        straight = cv2.imread(straight_path)

        if folded is None or straight is None:
            print(f"Error loading images: {folded_path} or {straight_path}")
            return None

        # Apply cropping using refine_crop
        cropped_folded = refine_crop(folded, padding=self.padding)
        cropped_straight = refine_crop(straight, padding=self.padding)

        # Pad images to square shape
        squared_folded = pad_to_square(cropped_folded)
        squared_straight = pad_to_square(cropped_straight)

        # Resize images
        resized_folded = cv2.resize(squared_folded, self.output_size_folded)
        resized_straight = cv2.resize(squared_straight, self.output_size_straight)

        # Randomly rotate folded leaf by any integer degree from 0 to 359
        random_angle = np.random.randint(0, 360)
        h, w = resized_folded.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
        rotated_folded = cv2.warpAffine(resized_folded, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Rotate straightened leaf exactly 90 degrees clockwise
        rotated_straight = cv2.rotate(resized_straight, cv2.ROTATE_90_CLOCKWISE)

        # Convert to RGB (OpenCV loads as BGR)
        rotated_folded = cv2.cvtColor(rotated_folded, cv2.COLOR_BGR2RGB)
        rotated_straight = cv2.cvtColor(rotated_straight, cv2.COLOR_BGR2RGB)

        # Convert to PyTorch tensors
        if self.transform:
            folded_tensor = self.transform(rotated_folded)
            straight_tensor = self.transform(rotated_straight)
        else:
            folded_tensor = torch.tensor(rotated_folded, dtype=torch.float32).permute(2, 0, 1) / 255.0
            straight_tensor = torch.tensor(rotated_straight, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return folded_tensor, straight_tensor

def show_batch(folded_images, straight_images):
    """
    Displays a batch of folded and straightened images.

    Parameters:
        folded_images (tensor): Batch of folded leaf images.
        straight_images (tensor): Batch of straightened leaf images.
    """
    batch_size = folded_images.shape[0]

    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))

    for i in range(batch_size):
        # Convert tensors to NumPy arrays for visualization
        folded_np = folded_images[i].permute(1, 2, 0).numpy()
        straight_np = straight_images[i].permute(1, 2, 0).numpy()

        # Denormalize the images
        folded_np = (folded_np * 0.5 + 0.5).clip(0, 1)  # Denormalize from [-1,1] to [0,1]
        straight_np = (straight_np * 0.5 + 0.5).clip(0, 1)  # Denormalize from [-1,1] to [0,1]

        # Display images
        axes[i, 0].imshow(folded_np)
        axes[i, 0].set_title(f"Folded Leaf {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(straight_np)
        axes[i, 1].set_title(f"Straightened Leaf {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Define the root directory and leafnames
leafnames = ['Brom01', 'Brom02', 'Brom03', 'Brom04', 'Brom05', 'Brom06']

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simple normalization to [-1,1]
])

# Create dataset and dataloader
dataset = LeafDataset(rootdir, leafnames, padding=300, transform=transform)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Iterate through DataLoader and display a batch
for folded_batch, straight_batch in dataloader:
    show_batch(folded_batch, straight_batch)
    break  # Only display the first batch