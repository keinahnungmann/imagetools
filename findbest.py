import cv2
import sys
import os
sys.path.append('/home/ben/Desktop/imagetools/EED')
from EED_gradmag import EED
import numpy as np
import scipy.sparse as sp
  
def find_best_parameters(image_path):
    image = cv2.imread(image_path)
    best_similarity = 10e12
    best_parameters = []
    h1 = 1
    h2 = 1

    output_folder = "/home/ben/Desktop/imagetools/testing/churner/sigma"
    os.makedirs(output_folder, exist_ok=True)
    
    plicit_array = [0.0]
    alpha_array = [32.0] #[2.0**i for i in range(0, 7)]
    tau_array = [4.0] #[2.0**i for i in range(2, 4)]
    iter_array = [4] #[2**i for i in range(0, 3+3)]
    rho_array = [0]
    sigma_array = [3,5,7,9,11,13,15,17,19,21] #[i*2 + 1 for i in range(0, 5)]
    quantile_array = [1]
   
    
    for plicit in plicit_array:
       for alpha in alpha_array:
           for tau in tau_array:
               for iterations in iter_array:
                   for rho in rho_array:
                       for sigma in sigma_array:
                           for quantile in quantile_array:
                               # Anisotropic edge enhancing
                               eed = EED(image, plicit, 32.0, 4.0, 1.0, 1.0, 3, quantile, 4)
  
                               # Name and store processed image and difference
                               name = f"EED_gradmag|{plicit}|{alpha}|{tau}|{sigma}|{quantile}|{iterations}0.jpg"
                               output_res = os.path.join(output_folder, name)
                               cv2.imwrite(output_res, eed)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_image_path> ")
        sys.exit(1)
        
    image_path = sys.argv[1]

    find_best_parameters(image_path)

