import os
import cv2
import numpy as np
import random
from PIL import Image

# Set the target size for all images
target_size = (256, 256)  # (width, height)

crop_size = 128
step_size = 128
target_ratio = 0.5
tolerance = 0.1

# Define the folders
folders = {
    'train_images': '/workspace/data/train/image',
    'train_labels': '/workspace/data/train/label',
    'val_images': '/workspace/data/val/image',
    'val_labels': '/workspace/data/val/label',
    'test_images': '/workspace/data/test/image'
    
    
}

# Define the output folders
resized_folders = {
    'train_images': '/workspace/data/train/resized',
    'train_labels': '/workspace/data/train/resized_label',
    'val_images': '/workspace/data/val/resized',
    'val_labels': '/workspace/data/val/resized_label',
    'test_images': 'data/test/resized'
    
}

# Make output folders if they don't exist
for path in resized_folders.values():
    os.makedirs(path, exist_ok=True)

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
    if is_label:
        #For train and val images since they contain labels
        for fname in os.listdir(input_label_dir):
            if fname.lower().endswith('.tif')and "vis" not in fname.lower():
                #Save label image
                file_path = os.path.join(input_label_dir, fname)
                img_label = np.array(Image.open(file_path))
                #Save image
                file_path = os.path.join(input_image_dir, fname)
                img_image = np.array(Image.open(file_path))
                #Turn label image to grey scale
                img_label_gray = img_label#cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
                #Get size of picture
                height, width = img_label_gray.shape
                #Cycle iterates through the image and takes out portions of it where white and black pix ratio is between 0.4 and 0.6
                for i in range(0, height-crop_size, step_size):
                    for j in range(0, width-crop_size, step_size):
                        #Crops crop_size pictures from label image
                        cropped_label = img_label_gray[i:i+crop_size, j:j+crop_size]
                        ratio = white_ratio(cropped_label)
                        print(ratio)
                        if abs(ratio - target_ratio) <= tolerance:
                            cropped_label = img_label[i:i+crop_size, j:j+crop_size]
                            cropped_image = img_image[i:i+crop_size, j:j+crop_size]
                            if np.max(cropped_label) > 1:
                                cropped_label_vis = (cropped_label/cropped_label.max()).astype(np.uint8)
                            #if np.max(cropped_label) <= 1:
                             #   cropped_label_vis = (cropped_label * 255).astype(np.uint8)
                            else:
                                cropped_label_vis = cropped_label.astype(np.uint8)
                                
                            cv2.imwrite(os.path.join(output_label_dir, f"{fname}_{i}_{j}.png"), cropped_label_vis)
                            cv2.imwrite(os.path.join(output_image_dir, f"{fname}_{i}_{j}.png"), cropped_image)
                            train_num+=1
                            if train_num>=2000:
                                return
                for pic_number_from_label in range(2):
                    i = random.randint(0,height-crop_size)
                    j = random.randint(0,width-crop_size)
                    cropped_label = img_label[i:i+crop_size, j:j+crop_size]
                    cropped_image = img_image[i:i+crop_size,j:j+crop_size]
                    cv2.imwrite(os.path.join(output_label_dir, f"{fname}_{i}_{j}.png"), (cropped_label*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(output_image_dir, f"{fname}_{i}_{j}.png"), cropped_image)
                    train_num+=1
                    if train_num>=2000:
                        return
                
                
    else:
        #For test images since they do not contain labels
        for fname in os.listdir(input_image_dir):
            if fname.lower().endswith('.tif'):
                img_image = cv2.imread(os.path.join(input_image_dir, fname))
                height, width, channels = img_image.shape
                for pic_number_from_img in range(5):
                    i = random.randint(0,height-crop_size)
                    j = random.randint(0,width-crop_size)
                    cropped_image = img_image[i:i+crop_size,j:j+crop_size]
                    cv2.imwrite(os.path.join(output_image_dir, f"{fname}_{i}_{j}.png"), cropped_image)
                    train_num+=1
                    if train_num>=500:
                        return
                



            

# Resize all folders
#resize_folder(folders['train_images'], resized_folders['train_images'], is_label=False)
#resize_folder(folders['train_labels'], resized_folders['train_labels'], is_label=True)
#resize_folder(folders['val_images'], resized_folders['val_images'], is_label=False)
#resize_folder(folders['val_labels'], resized_folders['val_labels'], is_label=True)
#resize_folder(folders['test_images'], resized_folders['test_images'], is_label=False)
modify_folder(folders['val_labels'], folders['val_images'], resized_folders['val_labels'], resized_folders['val_images'], is_label=True, train_num=400)
modify_folder(folders['train_labels'], folders['train_images'], resized_folders['train_labels'], resized_folders['train_images'], is_label=True, train_num=0)
modify_folder(None,folders['test_images'],None,resized_folders['test_images'], is_label=False, train_num=450)


print("All images and labels have been resized automatically!")
