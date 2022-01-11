# This script resizes all the images in a folder and save them in another folder

import os
import cv2
import matplotlib.pyplot as plt

# Input original folder of dataset and save path after resizing
original_folder = ""
save_folder = ""

for folder in os.listdir(original_folder):
    print("Folder: ", folder)
    folder_path = os.path.join(original_folder, folder)
    for category in os.listdir(folder_path):
        print("Category: ", category)
        category_path = os.path.join(folder_path, category)
        for image_name in os.listdir(category_path):
            print("Image name: ", image_name)
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            dimension = (224, 224)
            resized = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
            save_path = save_folder + folder + "/" + category + "/" + image_name
            cv2.imwrite(save_path, resized)
            print("Saved image to ", save_path)