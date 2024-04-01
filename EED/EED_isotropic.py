import numpy as np
import cv2
import sys
import argparse
import scipy.sparse as sp
import os


        
        
def vectorize(image):
    
    #stack rows
    vector = [image[j, :] for j in range(image.shape[0])]
    vector = np.concatenate(vector)
    vector = vector.reshape(-1,1)
    return vector
    
    
def Weickertdiff(norm, C):
    return np.where(norm < 10e-8, 1, 1 - np.exp(-3.31488 / (norm)  / C)**8)) 
    
def Peronadiff(norm2, C):
    return 1/(1+(norm2/C**2))


def EED_isotropic(image, plicit, alpha, tau, h1, h2, sigma, C, iterations):
    
    image = cv2.split(image)[0]
    gray = image
    for _ in [0]:
    
        
             
        for k in range(iterations):  
            
            usigma = cv2.GaussianBlur(gray, (sigma, sigma), 0)
            reflect = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            h,w = gray.shape
            usigma_x = np.zeros((h, w))
            usigma_y = np.zeros((h, w))
            diff = np.zeros((h, w))
        
            #compute partial derivatives
            
            usigma_x = cv2.Sobel(usigma, cv2.CV_64F, 1, 0, ksize=3)
            usigma_y = cv2.Sobel(usigma, cv2.CV_64F, 0, 1, ksize=3)
           
      
            	
            
            #compute diffusivity          
            norm2 = usigma_x**2 + usigma_y**2
            diff = Weickertdiff(norm2, C)
                
            #mirror diffusivity at boundaries:
            diff = cv2.copyMakeBorder(diff, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            
            	
            #build system matrix
            data = []
            row_indices = []
            col_indices = []
        	
            #build matrix that contains weights of spatial discretisation
            for m in range(w*h):
            
             	
        	    #'m' denotes the current row
        	    #we want that <r_m, v(u_t)> = v(u_t+1)[m]  
        	    #hence r_m should contain all weights associated with the pixel (i,j)(m)
        	    
        	    #init: use m = (m // w)*w + m % w  = j*w + i
                    j = m // w  #y-coordinate
                    i = m % w   #x-coordinate
                    tm, ml, mm, mr, bm = np.zeros((3+2))
        	   
        	    #top middle: (i, j-1)  
                    tm = +tau/(2 * h2**2) * (diff[j+1, i+1] + diff[j, i+1])
                    
                    if j > 0:
                        j_tm = (j-1)*w + i 
                        row_indices.append(m)
                        col_indices.append(j_tm)
                        data.append(tm)
                    else: mm += tm
        	    
        	    #middle left: (i-1, j)
                    ml = +tau /(2 * h1**2) * (diff[j+1,i+1] + diff[j+1, i])
                    
                    if i > 0:
                        j_ml = j*w + i - 1
                        row_indices.append(m)
                        col_indices.append(j_ml)
                        data.append(ml)
                    else: mm += ml
        	
        	    #middle right: (i+1, j)
                    mr = +tau/(2 * h1**2) * (diff[j+1, i+2] + diff[j+1, i+1])
                    
                    if i < w-1:
                        j_mr = j*w + i + 1
                        row_indices.append(m)
                        col_indices.append(j_mr)
                        data.append(mr)
                    else: mm += mr
        	
        	    #bottom middle: (i, j+1)
                    bm = + tau / (2 * h2**2) * (diff[j+2,i+1] + diff[j+1, i+1])
                 
                    if j < h-1:
                        j_bm = (j+1)*w + i
                        row_indices.append(m)
                        col_indices.append(j_bm)
                        data.append(bm)
                    else: mm += bm
                        
                    #middle middle 
                    mm += - tm - ml - mr - bm
                    row_indices.append(m)
                    col_indices.append(m)
                    data.append(mm)
                
            tauA = sp.csr_matrix((data, (row_indices, col_indices)))
            I = sp.eye(w*h)
                
            if plicit > 0:
                if alpha == 0: 
                    gray += np.dot(tauA, vectorize(gray)).reshape(h,w)	#regular explicit scheme
                else:	
                    gray = 1/(alpha + tau)*(alpha * gray + tau * image) + alpha*np.dot(tauA, vectorize(gray)).reshape(h,w)
                        
            else:
                    
                if alpha > 0:
                    matrix = I - alpha/(tau + alpha)*tauA
                    vector = 1 / (alpha+tau) * (alpha*gray + tau*image)
                    gray = sp.linalg.cg(matrix, vectorize(vector), maxiter=100000, tol=10e-9)[0].astype(np.uint8).reshape(h,w)
                   
                else:
                    matrix = I - tauA
                    gray = sp.linalg.cg(matrix, vectorize(gray), maxiter=100000, tol=10e-9)[0].astype(np.uint8).reshape(h,w)
                        
                       
                
                
                
               
        	
        	     	
    return gray
    
    
    
def main(input_path, output_folder, *args):
    # Load the input image
    image = cv2.imread(input_path)



    if image is None:
        print("Error: Unable to read the input image.")
        return

    # Call the CED function with provided arguments
    processed_image = EED_isotropic(image, *args)
    name = "iEED" +"".join(str(elem) + "|" for elem in args)
    cv2.imshow(name, processed_image)
    cv2.waitKey(0)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed image to the output folder
    name = "CED" +"".join(str(elem) + "|" for elem in args)
    output_path = os.path.join(output_folder, name + ".jpg") 
    cv2.imwrite(output_path, processed_image)
    print("Processed image saved at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python script.py <input_image_path> <output_folder> <tau> <sigma> <rho> <C> <iterations>")
    else:
        input_path = sys.argv[1]
        output_folder = sys.argv[2]
        plicit = float(sys.argv[3])
        alpha = int(sys.argv[4])
        tau = float(sys.argv[4+1])
        sigma = int(sys.argv[4+2])
        C = float(sys.argv[7])
        iterations = int(sys.argv[8])

        main(input_path, output_folder, plicit, alpha, tau, 1, 1, sigma, C, iterations)

