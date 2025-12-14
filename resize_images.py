import os
import cv2
import numpy as np
import random
from PIL import Image


crop_size = 256
step_size = 126
target_ratio = 0.3
tolerance = 0.1

# Define the folders
folders = {
    'train_images': 'Data/raw/train/image',
    'train_labels': 'Data/raw/train/label',
    'test_images': 'Data/raw/test/image',
    'test_label': 'Data/raw/test/label'
    
    
}

# Define the output folders
resized_folders = {
    'train_images': 'Data/resized/train256/image',
    'train_labels': 'Data/resized/train256/label',
    'test_images': 'Data/resized/test256/image',
    'test_label': 'Data/resized/test256/label'
    
}

# Make output folders if they don't exist
for path in resized_folders.values():
    os.makedirs(path, exist_ok=True)
    print(path)

# Get the ratio of black and white pictures within the mask
def white_ratio(label_cropped):
    # Ensure the input is grayscale
    if len(label_cropped.shape) == 3:
        label_cropped = cv2.cvtColor(label_cropped, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-255 range if needed (e.g., if values are 0 and 1)
    if np.max(label_cropped) <= 1:
        label_cropped = (label_cropped * 255).astype(np.uint8)
    
    # Threshold to create binary image (0 for black, 255 for white)
    _, img_binary = cv2.threshold(label_cropped, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate ratio of white pixels (255)
    white_pixels = np.sum(img_binary == 255)
    total_pixels = img_binary.size
    return white_pixels / total_pixels if total_pixels > 0 else 0
    
def modify_folder(input_label_dir, input_image_dir, output_label_dir, output_image_dir, is_label=True, train_num=0):
    # if is_label:
        #For train and val images since they contain labels
        for fname in os.listdir(input_label_dir):
            if fname.lower().endswith('.tif')and "vis" not in fname.lower():
                #Save label image
                file_path = os.path.join(input_label_dir, fname)
                # img_label = np.array(Image.open(file_path))
                img_label = Image.open(file_path)
                #Save image
                file_path = os.path.join(input_image_dir, fname)
                # img_image = np.array(Image.open(file_path))
                img_image = Image.open(file_path)
                #Get size of picture
                (width, height) = (img_label.width // 2, img_label.height // 2)
                print((width, height))
                img_label_resized = img_label.resize((width, height))
                img_image_resized = img_image.resize((width, height))
                standartized_label= np.array(img_label_resized)
                standartized_image= np.array(img_image_resized) 
                
                #Cycle iterates through the image and takes out portions of it where white and black pix ratio is between 0.4 and 0.6
                for i in range(0, height-crop_size, step_size):
                    for j in range(0, width-crop_size, step_size):

                            cropped_label = standartized_label[i:i+crop_size, j:j+crop_size]
                            ratio = white_ratio(cropped_label)
                            print(ratio)
                            if abs(ratio - target_ratio) <= tolerance:
                                cropped_image = standartized_image[i:i+crop_size, j:j+crop_size]
                                if np.max(cropped_label) > 1:
                                    cropped_label_vis = (cropped_label/cropped_label.max()).astype(np.uint8)
                                #if np.max(cropped_label) <= 1:
                                    #   cropped_label_vis = (cropped_label * 255).astype(np.uint8)
                                else:
                                    cropped_label_vis = cropped_label.astype(np.uint8)
                                
                                cropped_label_vis=Image.fromarray(cropped_label_vis)
                                cropped_image = Image.fromarray(cropped_image)
                            # Save cropped label and image with PNG format and no compression
                                cropped_label_vis.save(os.path.join(output_label_dir, f"{fname}_{i}_{j}.png"), format="PNG", compress_level=0)
                                cropped_image.save(os.path.join(output_image_dir, f"{fname}_{i}_{j}.png"), format="PNG", compress_level=0)
                          
                



            

# Resize all folders
#modify_folder(folders['val_labels'], folders['val_images'], resized_folders['val_labels'], resized_folders['val_images'], is_label=True, train_num=400)
modify_folder(folders['train_labels'], folders['train_images'], resized_folders['train_labels'], resized_folders['train_images'], is_label=True, train_num=0)
modify_folder(folders['test_label'],folders['test_images'],resized_folders['test_label'],resized_folders['test_images'], is_label=False, train_num=1800)


print("All images and labels have been resized automatically!")
