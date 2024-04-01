import EED_anisotropic
import cv2
import sys
import os
import numpy as np
import scipy.sparse as sp

        
def vectorize(image):
    
    #stack rows
    vector = [image[j, :] for j in range(image.shape[0])]
    vector = np.concatenate(vector)
    vector = vector.reshape(-1,1)
    return vector
    
   
def contrast(image,  quantile):

    vec = sorted(image.flatten())
    C = int(quantile * len(vec))
    
    return C

def EED(image, plicit, alpha, tau, h1, h2, sigma, rho, quantile, iterations):
    
    curr = image
    RGB_image = cv2.split(image)
    h,w,_ = image.shape
    
    for z in range(iterations):

        #reduce to one channel if grayscale
        RGB = list(cv2.split(curr)) 
        if np.linalg.norm(cv2.absdiff(RGB[1],RGB[2]) + cv2.absdiff(RGB[0],RGB[1]))==0:
           RGB = [RGB[0]] 
        K = len(RGB)
           
        
        #blur and split channels
        usigma = [cv2.GaussianBlur(c, (sigma, sigma), 0) for c in RGB]
        
        #compute derivatives
        usigma_x = [cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) for u in usigma]
        usigma_y = [cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3) for u in usigma]
        
        #incorporate natural boundary conditions
        for i in range(w):
            for j in range(h):
               for k in range(K):
                    if (i==0 or j==0 or i==w-1 or j==h-1):
                        usigma_x[k][j,i] = usigma_y[k][j,i] = 0.0
                        
        #build structure tensor
        outer = np.zeros((3,h,w))
        for k in range(K):
            outer_k = [usigma_x[k]**2 , usigma_x[k]*usigma_y[k], usigma_y[k]**2]
            outer += outer_k
            
            
        #compute quantile of generalized smoothed gradient magnitude in first iteration
        #trace of the sum of outer products
        if z==0:
            gradmag = outer[0] + outer[2]
            C = contrast(gradmag, quantile)
        
        #derive diffusion tensor
        D = np.zeros((h, w, 2, 2))
        for i in range(w):
            for j in range(h):
                    
                    matrix = np.stack(([outer[0][j,i], outer[1][j,i]], [outer[1][j,i], outer[2][j,i]]))
                    eigenvalues, eigenvectors = np.linalg.eig(matrix)
                
                    #rearrange eigenvalues, -vectors so that the dominant pair is first
                    sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
                    eigenvalues = eigenvalues[sorted_indices]
                    v1,v2 = eigenvectors[sorted_indices]
                    
                    #normalize eigenvectors
                    norm1 = np.linalg.norm(v1)
                    v1 /= norm1 if norm1 > 0 else 1
                    norm2 = np.linalg.norm(v2)
                    v2 /= norm2 if norm2 > 0 else 1 
                
                    #compute diffusivity: 
                    edger = matrix[0,0] + matrix[1,1]
                    perona = 1/(1+edger**2/C**2)
                    weickert = 1 if edger == 0 else np.exp(-3.31488/((edger/C)**8))
                    
                    transform = np.vstack((v1,v2))
                    diag = np.vstack(([weickert, 0.0], [0.0, 1]))
                    D[j,i] = np.dot(transform.T, np.dot(diag, transform))
                 	 
         
         
        #diffuse each channel separately  
        
        for _ in [0]:     
            #Construct sparse system matrix
            #compute parameter beta = (1-2*alpha)sgn(J[j,i])alpha = 13/32
            beta = np.zeros((h,w))
            for i in range(w):
                for j in range(h):
                    beta[j,i] = (19/32)*(1 if outer[1][j,i] > 0 else -1 if outer[1][j,i] < 0 else 0.0)
                    
                    
            a = cv2.copyMakeBorder(D[:, :, 0, 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            b = cv2.copyMakeBorder(D[:, :, 1, 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            c = cv2.copyMakeBorder(D[:, :, 1, 1], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            
            #chose delta = 1/2(a+c)
            d = cv2.copyMakeBorder(abs(D[:,:,1,0]), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        
            data = []
            row_indices = []
            col_indices = []
        
            for m in range(w*h):

            			#determine indeces:
            			#theory:(i-1, j+1) / cv2:(j-1, i-1) / l sign+-+-
            			mm = 0.0
            			j = m // w
            			i = m % w
            			
            			#time savers 
            			cdiag = 1.0/(4*h1*h2)	#= 1/2*1/2*h1*h2, latter comes form diag directional deriv
            			chor = 1.0/2*h2**2  	#= 1/2*1/h1**2, latter comes from horiz. directional deriv
            			cver = 1.0/2*h1**2	#smlr.
            			
            			#top left: theory:(i-1,j+1) / cv2:(i-1,j-1) /
            			#weight = cdiag * (delta-b)(i-1/2,j+1/2) = 1/4*h1*h2* ((d-b)[i-1,j+1] + (d-b)[i,j])
            			tl = cdiag * (d[j, i+1] - b[j, i+1] + d[j+1,i] - b[j+1,i]) 
            			
            			if (i==0 or j==0):
            			    mm += tl
            			else:
            			    j_tl = (j-1)*w + i - 1  
            			    row_indices.append(m)
            			    col_indices.append(j_tl)
            			    data.append(tl)
                                
                                #top middle: (i,j+1) 
                                #weight = cver * (c-delta)(i,j+1/2) = 1/2*h2**2 * ((c-d)[i, j+1] + (c-d)[i,j])
            			tm = cver * (c[j,i+1] - d[j,i+1] + c[j+1,i+1] - d[j+1,i+1])
            			
            			if (j==0):
            			    mm += tm
            			else:
            			    j_tm = (j-1)*w + i    
            			    row_indices.append(m)
            			    col_indices.append(j_tm)
            			    data.append(tm)
                                
                                #top right: (i+1,j+1)
                                #weight = cdiag * (delta+b)(i+1/2,j+1/2)) = 1/4*h1*h2 * ((d+b)[i+1,j+1] +(d+b)[i,j])
            			tr = cdiag * (d[j,i+1] + b[j,i+1] + d[j+1,i] + b[j+1,i])
            			
            			if (i==w-1 or j==0):
            			    mm += tr
            			else:
            			    j_tr = (j-1)*w + i + 1   
            			    row_indices.append(m)
            			    col_indices.append(j_tr)
            			    data.append(tr)
                                
                                #center left (i-1,j)
                                #weight = chor * (a-delta)[i-1/2,j] = 1/2*h1**2 * ((a-d)[i-1,j] + (a-d)[i,j]
            			ml = chor * (a[j+1,i] - d[j+1,i] + a[j+1,i+1] - d[j+1,i+1])
            			
            			if (i==0):
            			    mm += ml
            			else:
            			    j_ml = j*w + i-1    
            			    row_indices.append(m)
            			    col_indices.append(j_ml)
            			    data.append(ml)
                                
                                #center right (i+1,j)
                                #weight = chor * (a-delta)[i+1/2,j] = 1/2*h1**2 * ((a-d)[i+1,j] + (a-d)[i,j])
            			mr = chor * (a[j+1,i+2] - d[j+1,i+2] + a[j+1,i+1] - d[j+1,i+1])
            			
            			if (i==w-1):
            			    mm += mr
            			else:
            			    j_mr = j*w + i +1    
            			    row_indices.append(m)
            			    col_indices.append(j_mr)
            			    data.append(mr)
                                
                                #bottom left (i-1,j-1)
                                #weight = cdiag * (delta+b)[i-1/2,j-1/2] = 1/4*h1*h2 * ((d+b)[i-1,j-1] + (d+b)[i,j])
            			bl = cdiag * (d[j+2,i+1] + b[j+2,i+1] + d[j+1,i] + b[j+1,i])
            			
            			if (i==0 or j==h-1):
            			    mm += bl
            			else:
            			    j_bl = (j+1)*w + i -1  
            			    row_indices.append(m)
            			    col_indices.append(j_bl)
            			    data.append(bl)
                               	
                                #bottom middle (i,j-1)
                                #weight = cver * (c-delta)[i,j-1/2] = 1/2*h2**2 * ((c-d)[i,j-1] + (c-d)[i,j])
            			bm = cver * (c[j+2,i+1] - d[j+2,i+1] + c[j+1,i+1] - d[j+1,i+1])
            			
            			if (j==h-1):
            			    mm += bm
            			else:
            			    j_bm = (j+1)*w + i    
            			    row_indices.append(m)
            			    col_indices.append(j_bm)
            			    data.append(bm)
                                
                                #bottom right
                                #weight = cdiag * (delta-b)[i+1/2,j-1/2]
            			br = cdiag * (d[j+2,i+1] - b[j+2,i+1] + d[j+1,i+2] - b[j+1,i+2])
            			
            			if (i==w-1 or j==h-1):
            			    mm += br
            			else:
            			    j_br = (j+1)*w + i +1    
            			    row_indices.append(m)
            			    col_indices.append(j_br)
            			    data.append(br)
            			
            			#center
            			mm += (-1)*(tl + tm +tr + ml + mr + bl + bm + br)
            			row_indices.append(m)
            			col_indices.append(m)
            			data.append(mm)
                    
            #build M = I - tau*A   
            A = sp.csr_matrix((data, (row_indices, col_indices)))
            I = sp.eye(w*h)
            for k in range(K):
                if alpha > 0:
                    matrix = I - alpha/(tau + alpha)*tau*A
                    vector = 1 / (alpha+tau) * (alpha*RGB[k] + tau*RGB_image[k])
                    RGB[k] = sp.linalg.cg(matrix, vectorize(vector), maxiter=10000, rtol=10e-12)[0].astype(np.uint8).reshape(h,w)
                   
                else:
                    matrix = I - tau*A
                    RGB[k] = sp.linalg.cg(matrix, vectorize(RGB[k]), maxiter=10000, rtol=10e-12)[0].astype(np.uint8).reshape(h,w)        
                            
            curr = cv2.merge(RGB) if K > 1 else cv2.merge((RGB[0], RGB[0], RGB[0]))	 	
                          
               	 	        
    return curr
    
    
    
def main(input_path, output_folder, *args):
    # Load the input image
    image = cv2.imread(input_path)
    image = cv2.resize(image, (600,600))	

    if image is None:
        print("Error: Unable to read the input image.")
        return

    # Call the CED function with provided arguments
    processed_image = EED(image, *args)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed image to the output folder
    name = "EED_ani" + "".join(str(elem) + " | " for elem in args)
    output_path = os.path.join(output_folder, name + ".jpg") 
    cv2.imshow(name, processed_image)
    cv2.waitKey(0)
    cv2.imwrite(output_path, processed_image)
    print("Processed image saved at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py <input_image_path> <output_folder> <plicit> <alpha> <tau> <sigma>  <quantile> <iterations> ")
    else:
        input_path = sys.argv[1]
        output_folder = sys.argv[2]
        plicit = int(sys.argv[3])
        alpha = float(sys.argv[4])
        tau = float(sys.argv[3+2])
        sigma = int(sys.argv[3+3])
        rho = int(sys.argv[7])
        quantile = float(sys.argv[8])
        iterations = int(sys.argv[9])
	
        main(input_path, output_folder, plicit, alpha, tau, 1, 1, sigma, rho, quantile, iterations)
        

