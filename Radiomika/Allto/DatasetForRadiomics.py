import re
import pandas as pd
import SimpleITK as sitk
import numpy as np
from imutils import paths
from PIL import Image
import cv2

class DatasetForRadiomics:
    def __init__(self, folder_original_path, folder_mask_path, csv_file, params):
        self.folder_original_path = folder_original_path
        self.folder_mask_path = folder_mask_path
        self.csv_file = csv_file
        self.params = params
    
    def read_paths(self):
        originalPaths = sorted(list(paths.list_images(self.folder_original_path)))
        maskPaths = sorted(list(paths.list_images(self.folder_mask_path)))
        
        return originalPaths, maskPaths
    
    def get_length(self):
        originalPaths = sorted(list(paths.list_images(self.folder_original_path)))
        return len(originalPaths)

    def path_to_image(self, original_path, mask_path):
        # Read and convert RGB image as Greyscale

        rgb_image = Image.open(original_path)

        img_array = np.array(rgb_image)

        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        grey_image = rgb_image.convert('L')
        grey_array = np.array(grey_image)
        original = sitk.GetImageFromArray(grey_array)
        original_red = sitk.GetImageFromArray(red_channel)
        original_green = sitk.GetImageFromArray(green_channel)
        original_blue = sitk.GetImageFromArray(blue_channel)

        # Read Greyscale image as Gretscale image
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        return original_blue, mask
    
    def path_to_array(self, original_path, mask_path):

        original_image = Image.open(original_path)
        mask_image = Image.open(mask_path).convert('L')
        original_array = np.array(original_image)
        mask_array = img_array = np.array(mask_image)

        return original_array, mask_array

        
    def get_dataframe(self):
        df = pd.read_csv(self.csv_file)
        return df
    
    # Function to convert
    def list_to_string(self, list):

        # initialize an empty string
        str1 = " "

        # return string
        return (str1.join(list))

    
    def categories_list(self):
        
        lesion_dict = {
            "MEL": 0, 
            "NV": 1,
            "BCC": 2,
            "AKIEC": 3,
            "BKL": 4,
            "DF": 5,
            "VASC": 6
        }

        # Find part of a string that starts with "ISIC_" and ends with number or numbers
        df = self.get_dataframe()
        originalPaths, maskPaths = self.read_paths()
        categories = []
        for path in originalPaths:
            
            name = re.findall(r"ISIC_\d+", path)
        
            # Select a row that has given name in column names
            row = df.loc[df['image'] == self.list_to_string(name)]

            column_name = None
        
            # Iterate through the columns in the subset to find the matching value
            for col in row.columns:
                if 1.0 in row[col].values:
                    column_name = col
                    break
        
            # Append it to a list of categories
            categories.append(lesion_dict[column_name])

        return categories
    
    def get_names(self):

        names = []
        originalPaths, maskPaths = self.read_paths()
        for path in originalPaths:
            name = re.findall(r"ISIC_\d+_\d", path)

            names.append(name)
        return names
    