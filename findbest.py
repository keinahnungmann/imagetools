import cv2
import sys
import itertools
sys.path.append("/home/ben/Schreibtisch/imagetools/EED")
sys.path.append("/home/ben/Schreibtisch/imagetools/CED")
import os
import EED_isotropic
import EED_anisotropic
import CED
import numpy as np


def find_best_parameters():
    best_similarity = 10e12
    best_parameters = []
    h1 = 1
    h2 = 1

    output_folder = "/home/ben/Desktop/imagetools/testing/churner"
    os.makedirs(output_folder, exist_ok=True)
    
    plicit_array = [0,1]
    alpha_array = [3*10**i for i in range(-1, 10)]
    tau_array = [4*10**i for i in range(0,10)]
    iter_array = [10**i for i in range(0,3+3)]
    rho_array = [i*2 +1 for i in range(4,81)]
    sigma_array = [i*2 + 1 for i in range(3,20)]
    contrast_array = [10**i for i in range(0, 10)]
   
    
    for plicit in plicit_array:
       for alpha in alpha_array:
           for tau in tau_array:
               for iter in iter_array:
                   for rho in rho_array:
                       for sigma in sigma_array:
                           for C in contrast_array:
                           
                              #anisotropic edge enhancing
                              EED_ani = EED_anisotropic.EED_anisotropic(image, plicit, alpha, tau, h1, h2, sigma, rho, C, iterations)
                        
                              #create difference image and compute MSE
                              diff = cv2.absdiff(processed_image, ground_truth)
                              similarity = (1.0/(image.shape[0]*image.shape[1])) * np.linalg.norm(diff)
                        
                              #store parameters if they're among the best 40	
                              if similarity < best_similarity:
                                  best_similarity = similarity
                                  best_parameters.append(("EED_ani", tau, sigma, C, iterations, similarity))
                            
                              # Sort and keep only top 40
                              best_parameters = sorted(best_parameters, key=lambda x: x[-1], reverse=True)[:200]
                              print("Better")
                        
                              #name and store processed image and difference
                        
                              name = "EED_ani" + "|" + str(plicit) + "|" + str(alpha) + "|" + str(tau) +"|" + str(sigma) + "|" + str(C) + "|" + str(iterations) + ".jpg"
                              output_res = os.path.join(output_folder, name)
                       	      output_diff = os.path.join(output_folder, "diff" + name)
                              cv2.imwrite(output_res, processed_image)
                              cv2.imwrite(output_diff, diff)
                              
                              
                              
                              #anisotropic edge coherence enhancing
                              CED = CED.CED(image, plicit, alpha, tau, h1, h2, sigma, rho, C, iterations)
                        
                              #create difference image and compute MSE
                              diff = cv2.absdiff(CED, ground_truth)
                              similarity = (1.0/(image.shape[0]*image.shape[1])) * np.linalg.norm(diff)
                        
                              #store parameters if they're among the best 40	
                              if similarity < best_similarity:
                                  best_similarity = similarity
                                  best_parameters.append(("CED", tau, sigma, C, iterations, similarity))
                            
                              # Sort and keep only top 40
                              best_parameters = sorted(best_parameters, key=lambda x: x[-1], reverse=True)[:200]
                              print("Better")
                        
                              #name and store processed image and difference
                        
                              name = "CED" + "|" + str(plicit) + "|" + str(alpha) + "|" + str(tau) +"|" + str(sigma) + "|" + str(C) + "|" + str(iterations) + ".jpg"
                              output_res = os.path.join(output_folder, name)
                       	      output_diff = os.path.join(output_folder, "diff" + name)
                              cv2.imwrite(output_res, processed_image)
                              cv2.imwrite(output_diff, diff)
                              
                              
                              
                              #isotropic edge enhancing
                              EED_iso = EED_isotropic.EED_isotropic(image, plicit, alpha, tau, h1, h2, sigma, C, iterations)
                        
                              #create difference image and compute MSE
                              diff = cv2.absdiff(processed_image, ground_truth)
                              similarity = (1.0/(image.shape[0]*image.shape[1])) * np.linalg.norm(diff)
                        
                              #store parameters if they're among the best 40	
                              if similarity < best_similarity:
                                  best_similarity = similarity
                                  best_parameters.append((EED_iso, tau, sigma, C, iterations, similarity))
                            
                              # Sort and keep only top 40
                              best_parameters = sorted(best_parameters, key=lambda x: x[-1], reverse=True)[:200]
                              print("Better")
                        
                              #name and store processed image and difference
                        
                              name = "EED_iso" + "|" + str(plicit) + "|" + str(alpha) + "|" + str(tau) +"|" + str(sigma) + "|" + str(C) + "|" + str(iterations) + ".jpg"
                              output_res = os.path.join(output_folder, name)
                       	      output_diff = os.path.join(output_folder, "diff" + name)
                              cv2.imwrite(output_res, processed_image)
                              cv2.imwrite(output_diff, diff)
                              

    return best_parameters
    
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_image_path> <ground_truth_image_path>")
    else:
        input_path = sys.argv[1]
        ground_truth_path = sys.argv[2]

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)	
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (300,200))
        ground_truth = cv2.resize(image, (300,200))

        if image is None or ground_truth is None:
            print("Error: Unable to read the images.")
            sys.exit(1)

        best_params = find_best_parameters()
        print("Best Parameters:", best_params)

