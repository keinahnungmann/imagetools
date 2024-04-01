import cv2
import sys
import os
import ced_raw
import numpy as np
from skimage.metrics import structural_similarity as ssim

def image_similarity(image1, image2):
    # Compute Structural Similarity Index (SSIM)
    return ssim(image1, image2)

def find_best_parameters():
    best_similarity = 10e12
    best_parameters = []
    h1 = 1
    h2 = 1

    output_folder = "/home/ben/Schreibtisch/imagetools/CED/test_numerical"
    os.makedirs(output_folder, exist_ok=True)
    
    tau_array = [i * 0.00002 for i in range(1, 40)]
    iter_array = [i for i in range(1,2)]
    rho_array = [i*2 +1 for i in range(4,40)]
    alpha_array = [round(i*0.0004,4) for i in range(1,100)]
    
    
    for iterations in iter_array:
    	for tau in tau_array:
            for rho in rho_array:
                for alpha in [1]:
                
                        #process image and compare to ground truth
                        processed_image = ced_raw.CED(image, tau, h1, h2, 3, rho, 1, iterations)
                        
                        #break if image is too dark
                        if np.mean(processed_image) < 13+2: 
                            print("break")
                            break
                       
                        #compare
                        #create difference image and compute MSE
                        diff = cv2.absdiff(processed_image, ground_truth)
                        similarity = (1.0/(image.shape[0]*image.shape[1])) * np.linalg.norm(diff)
                        
                        #store parameters if they're among the best 40
                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_parameters.append((tau, rho, iterations, similarity))
                            
                            # Sort and keep only top 40
                            best_parameters = sorted(best_parameters, key=lambda x: x[-1], reverse=True)[:40]
                            print("Better")
                        
                        #name and store processed image and difference
                        name = str(tau) +"|" + str(rho) + "|" + str(iterations) + "|" + str(alpha) +"reaction.jpg"
                        output_res = os.path.join(output_folder, name)
                       	output_diff = os.path.join(output_folder, "diff" + name)
                        cv2.imwrite(output_res, processed_image)
                        cv2.imwrite(output_diff, diff)
                        print("images saved")

    return best_parameters



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_image_path> <ground_truth_image_path>")
    else:
        input_path = sys.argv[1]
        ground_truth_path = sys.argv[2]

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (800, 200))
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.resize(ground_truth, (800, 200))

        if image is None or ground_truth is None:
            print("Error: Unable to read the images.")
            sys.exit(1)

        best_params = find_best_parameters()
        print("Best Parameters:", best_params)

