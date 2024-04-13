import os
import sys
import cv2

sys.path.append('/home/ben/Desktop/imagetools/EED')
from EED_gradmag import EED

def diffuse_grids(directory):
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image file (you may need to adjust the extension list)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    
            # Construct the full path to the image file
            filepath = os.path.join(directory, filename)
        
            # Read the image
            print(filepath)
            image = cv2.imread(filepath)
        
            # Check if the image is valid
            if image is not None:
                # Apply EED
                eed_image = EED(image, 0, 32.0, 4.0, 1, 1, 3, 0.7, 1)  # Assuming EED function takes an 
                cv2.imwrite(filepath, eed_image)
                print("done")
            
if __name__ == "__main__":
    # Specify the directory containing the image files
    directory = '/home/ben/Desktop/imagetools/cardreader/cards/grids'
    
    # Apply EED to images in the specified directory
    diffuse_grids(directory)
