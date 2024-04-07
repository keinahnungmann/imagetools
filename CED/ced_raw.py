import cv2
import sys
import os
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

def vectorize(image):
    # Stack rows
    vector = [image[j, :] for j in range(image.shape[0])]
    vector = np.concatenate(vector)
    vector = vector.reshape(-1, 1)
    return vector

def contrast(image, quantile):
    vec = image.flatten()
    i = int(quantile * len(vec))
    if i == 0: i = 1
    C = vec[i-1]
    
    return C

def CED(image, plicit, alpha, tau, h1, h2, sigma, rho, quantile, iterations):
    # Initialize parameters
    C = 1
    RGB_image = cv2.split(image)
    K = len(RGB_image)
    curr = image
    h, w = image.shape[:2]
    
    for z in range(iterations):
    
        RGB = list(cv2.split(curr))
        # Preprocessing
        # Blur
        usigma = [cv2.GaussianBlur(c, (sigma, sigma), 0) for c in RGB_image]
        # Compute derivatives
        usigma_x = [cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3) for c in usigma]
        usigma_y = [cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3) for c in usigma]
        # Incorporate natural boundary conditions
        for i in range(w):
            for j in range(h):
                for k in range(K):
                    if (i==0 or j==0 or i==w-1 or j==h-1):
                        usigma_x[k][j,i] = usigma_y[k][j,i] = 0.0
        # Compute and add outer products
        J = np.zeros((3,h,w))
        for k in range(K):
                outer_k = [usigma_x[k]**2, usigma_x[k] * usigma_y[k], usigma_y[k]**2]
                J_k = [cv2.GaussianBlur(entry, (rho, rho), 0) for entry in outer_k]
                J += J_k
                
        # Preprocess for diffusion tensor computation
        if z == 0:
            coherence = np.zeros((h,w))
            for i in range(w):
                for j in range(h):
                    m = np.vstack(([J[0][j,i], J[1][j, i]], [J[1][j, i], J[2][j, i]]))
                    aux = np.sqrt((m[0,0] - m[1,1])**2 + 4*m[1,0]**2)
                    e0 = 0.5*(m[0,0] + m[1,1] + aux)
                    e1 = 0.5*(m[0,0] + m[1,1] - aux)
                    coherence[j,i] = abs(e1)
                
            C = contrast(coherence, quantile)
        
        # Derive diffusion tensor
        D = np.zeros((h, w, 2, 2))
        for i in range(w):
            for j in range(h):
                m = np.vstack(([J[0][j, i], J[1][j, i]], [J[1][j, i], J[2][j, i]]))
                eigenvalues, eigenvectors = np.linalg.eig(m)
                sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
                e0,e1 = eigenvalues[sorted_indices]
                v1,v2 = eigenvectors[sorted_indices]
                # Normalize eigenvectors
                norm1 = np.linalg.norm(v1)
                v1 /= norm1 if norm1 > 0 else 1
                norm2 = np.linalg.norm(v2)
                v2 /= norm2 if norm2 > 0 else 1 
                coh = abs(e0)
                weickert = np.exp(-3.31488/(coh/C)**8) if coh > 0 else 0.0
                lam1 = 1.0 - weickert
                lam2 = 1.0
                transform = np.vstack((v1, v2))
                diag = np.vstack(([lam1, 0.0], [0.0, lam2]))
                D[j, i] = np.dot(transform.T, np.dot(diag, transform)) 
                
        # Construct system matrix
        a = cv2.copyMakeBorder(D[:, :, 0, 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        b = cv2.copyMakeBorder(D[:, :, 1, 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        c = cv2.copyMakeBorder(D[:, :, 1, 1], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        d = cv2.copyMakeBorder(abs(D[:,:,1,0]), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
       
        data = []
        row_indices = []
        col_indices = []
        
        for m in range(w * h):
            mm = 0.0
            j = m // w
            i = m % w
            cdiag = 1.0/(4*h1*h2)
            chor = 1.0/2*h2**2
            cver = 1.0/2*h1**2
            
            tl = cdiag * (d[j, i+1] - b[j, i+1] + d[j+1,i] - b[j+1,i]) 
            if (i==0 or j==0):
                mm += tl
            else:
                j_tl = (j-1)*w + i - 1  
                row_indices.append(m)
                col_indices.append(j_tl)
                data.append(tl)

            tm = cver * (c[j,i+1] - d[j,i+1] + c[j+1,i+1] - d[j+1,i+1])
            if (j==0):
                mm += tm
            else:
                j_tm = (j-1)*w + i    
                row_indices.append(m)
                col_indices.append(j_tm)
                data.append(tm)

            tr = cdiag * (d[j,i+1] + b[j,i+1] + d[j+1,i] + b[j+1,i])
            if (i==w-1 or j==0):
                mm += tr
            else:
                j_tr = (j-1)*w + i + 1   
                row_indices.append(m)
                col_indices.append(j_tr)
                data.append(tr)

            ml = chor * (a[j+1,i] - d[j+1,i] + a[j+1,i+1] - d[j+1,i+1])
            if (i==0):
                mm += ml
            else:
                j_ml = j*w + i-1    
                row_indices.append(m)
                col_indices.append(j_ml)
                data.append(ml)

            mr = chor * (a[j+1,i+2] - d[j+1,i+2] + a[j+1,i+1] - d[j+1,i+1])
            if (i==w-1):
                mm += mr
            else:
                j_mr = j*w + i +1    
                row_indices.append(m)
                col_indices.append(j_mr)
                data.append(mr)

            bl = cdiag * (d[j+2,i+1] + b[j+2,i+1] + d[j+1,i] + b[j+1,i])
            if (i==0 or j==h-1):
                mm += bl
            else:
                j_bl = (j+1)*w + i -1  
                row_indices.append(m)
                col_indices.append(j_bl)
                data.append(bl)
                   	
            bm = cver * (c[j+2,i+1] - d[j+2,i+1] + c[j+1,i+1] - d[j+1,i+1])
            if (j==h-1):
                mm += bm
            else:
                j_bm = (j+1)*w + i    
                row_indices.append(m)
                col_indices.append(j_bm)
                data.append(bm)

            br = cdiag * (d[j+2,i+1] - b[j+2,i+1] + d[j+1,i+2] - b[j+1,i+2])
            if (i==w-1 or j==h-1):
                mm += br
            else:
                j_br = (j+1)*w + i +1    
                row_indices.append(m)
                col_indices.append(j_br)
                data.append(br)

            mm += (-1)*(tl + tm +tr + ml + mr + bl + bm + br)
            row_indices.append(m)
            col_indices.append(m)
            data.append(mm)

        A = sp.csr_matrix((data, (row_indices, col_indices)))
        I = sp.eye(w*h)
        
        def solve_cg(matrix, vector):
            return sp.linalg.cg(matrix, vector, maxiter=10000, tol=10e-6)[0].astype(np.uint8).reshape(h, w)

        def solve_cg_wrapper(k):
            nonlocal I
            if alpha > 0:
                matrix = I - alpha / (tau + alpha) * tau * A
                vector = 1 / (alpha + tau) * (alpha * RGB[k] + tau * RGB_image[k])
            else:
                matrix = I - tau * A
                vector = RGB[k]
            return solve_cg(matrix, vectorize(vector))

        # Parallelize the conjugate gradient solver loop
        RGB = Parallel(n_jobs=-1)(delayed(solve_cg_wrapper)(k) for k in range(K))
        curr = cv2.merge(RGB) if K > 1 else cv2.merge((RGB[0], RGB[0], RGB[0]))
        
    return curr

def main(input_path, output_folder, *args):
    # Load the input image
    image = cv2.imread(input_path)

    if image is None:
        print("Error: Unable to read the input image.")
        return

    processed = CED(image, *args)

    name = "CED" + "".join(str(elem) + "|" for elem in args)
    cv2.imshow(name, processed)
    cv2.waitKey(0)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed image to the output folder
    name = "CED" + "".join(str(elem) + "|" for elem in args)
    output_path = os.path.join(output_folder, name + ".jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python script.py <input_image_path> <output_folder> <tau> <sigma> <rho><iterations>")
    else:
        input_path = sys.argv[1]
        output_folder = sys.argv[2]
        plicit = int(sys.argv[3])
        alpha = float(sys.argv[4])
        tau = float(sys.argv[5])
        sigma = int(sys.argv[6])
        rho = int(sys.argv[7])
        quantile = float(sys.argv[8])
        iterations = int(sys.argv[9])

        main(input_path, output_folder, plicit, alpha, tau, 1, 1, sigma, rho, quantile, iterations)

