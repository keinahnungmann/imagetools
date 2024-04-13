import cv2
import sys
import os
import numpy as np
import scipy.sparse as sp


def vectorize(image):
    # stack rows
    vector = [image[j, :] for j in range(image.shape[0])]
    vector = np.concatenate(vector)
    vector = vector.reshape(-1, 1)
    
    return vector


def contrast(image, quantile):

    vec = sorted(image.flatten())
    i = int(quantile * len(vec))
    if i == 0:
    	i = 1
    	
    C = vec[i-1]
    
    return C


def EED(image, plicit, alpha, tau, h1, h2, sigma, quantile, iterations):
    # init
    h, w = image.shape[:2]  # image dimension
    RGB_image = list(cv2.split(image))  # original image channels
    K = len(RGB_image)
    print(K)
    curr = image  # iterators
    C = 1

    # main image processing
    for z in range(iterations):
        # init

        # reduces 'RGB' to one channel if image is grayscale
        # ---------------------------------------------------------------------#
        RGB = list(cv2.split(curr))
        # ---------------------------------------------------------------------#

        # compute regularized gradient with natural boundary conditions
        # ---------------------------------------------------------------------#
        usigma = [cv2.GaussianBlur(c, (sigma, sigma), 0) for c in RGB]
        usigma_x = [cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) for u in usigma]
        usigma_y = [cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3) for u in usigma]

        for i in range(w):
            for j in range(h):
                for k in range(K):
                    if (i == 0 or j == 0 or i == w - 1 or j == h - 1):
                        usigma_x[k][j, i] = usigma_y[k][j, i] = 0.0
        # ---------------------------------------------------------------------#

        # compute quantile of generalized smoothed gradient magnitude in first iteration
        # ---------------------------------------------------------------------#
        if z < 0:
            # add outer products
            outer = np.zeros((3, h, w))
            for k in range(K):
                outer_k = [usigma_x[k] ** 2, usigma_x[k] * usigma_y[k], usigma_y[k] ** 2]
                outer += outer_k

            # compute generalized gradient magnitude
            gradmag = outer[0] + outer[2]

            # derive contrast parameter
            C = contrast(gradmag, quantile)
        # ---------------------------------------------------------------------#

        # compute diffusion tensor
        # ---------------------------------------------------------------------#
        D = np.zeros((h, w, 3))
        for i in range(w):
            for j in range(h):
                # add outer products
                outer = np.zeros(3)
                for k in range(K):
                    outer_k = [usigma_x[k][j, i] ** 2, usigma_x[k][j, i] * usigma_y[k][j, i], usigma_y[k][j, i] ** 2]
                    outer += outer_k

                # build matrix, compute eigenpairs
                matrix = np.stack(([outer[0], outer[1]], [outer[1], outer[2]]))
                eigenvalues, eigenvectors = np.linalg.eig(matrix)

                # rearrange eigenvalues, -vectors so that the dominant pair is first
                sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
                eigenvalues = eigenvalues[sorted_indices]
                v1, v2 = eigenvectors[sorted_indices]

                # normalize eigenvectors
                norm1 = np.linalg.norm(v1)
                v1 /= norm1 if norm1 > 0 else 1
                norm2 = np.linalg.norm(v2)
                v2 /= norm2 if norm2 > 0 else 1

                # compute diffusivity:
                edger = matrix[0, 0] + matrix[1, 1]
                perona = 0.0
                #1 / (1 + edger ** 2 / C ** 2)
                weickert = 1 if edger == 0 else np.exp(-3.31488 / ((edger / C) ** 8))

                # compute diffusion tensor
                transform = np.vstack((v1, v2))
                diag = np.vstack(([0.0, 0.0], [0.0, 1.0]))
                diff = np.dot(transform.T, np.dot(diag, transform))

                # store in array
                D[j, i] = [diff[0, 0], diff[1, 0], diff[1, 1]]
        # ---------------------------------------------------------------------#


        # build system matrix
        # ---------------------------------------------------------------------#
        a = cv2.copyMakeBorder(D[:, :, 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        b = cv2.copyMakeBorder(D[:, :, 1], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        c = cv2.copyMakeBorder(D[:, :, 2], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        d = cv2.copyMakeBorder(abs(D[:, :, 1]), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        data = []
        row_indices = []
        col_indices = []

        for m in range(w * h):

            # determine indeces:
            mm = 0.0
            j = m // w
            i = m % w

            # time savers
            cdiag = 1.0 / (4 * h1 * h2)
            chor = 1.0 / 2 * h2 ** 2
            cver = 1.0 / 2 * h1 ** 2

            # ---------------------------------------------------------------------#
            tl = cdiag * (d[j, i + 1] - b[j, i + 1] + d[j + 1, i] - b[j + 1, i])

            if (i == 0 or j == 0):
                mm += tl
            else:
                j_tl = (j - 1) * w + i - 1
                row_indices.append(m)
                col_indices.append(j_tl)
                data.append(tl)

            # ---------------------------------------------------------------------#

            tm = cver * (c[j, i + 1] - d[j, i + 1] + c[j + 1, i + 1] - d[j + 1, i + 1])

            if (j == 0):
                mm += tm
            else:
                j_tm = (j - 1) * w + i
                row_indices.append(m)
                col_indices.append(j_tm)
                data.append(tm)

            # ---------------------------------------------------------------------#

            tr = cdiag * (d[j, i + 1] + b[j, i + 1] + d[j + 1, i] + b[j + 1, i])

            if (i == w - 1 or j == 0):
                mm += tr
            else:
                j_tr = (j - 1) * w + i + 1
                row_indices.append(m)
                col_indices.append(j_tr)
                data.append(tr)

            # ---------------------------------------------------------------------#

            ml = chor * (a[j + 1, i] - d[j + 1, i] + a[j + 1, i + 1] - d[j + 1, i + 1])

            if (i == 0):
                mm += ml
            else:
                j_ml = j * w + i - 1
                row_indices.append(m)
                col_indices.append(j_ml)
                data.append(ml)

            # ---------------------------------------------------------------------#

            mr = chor * (a[j + 1, i + 2] - d[j + 1, i + 2] + a[j + 1, i + 1] - d[j + 1, i + 1])

            if (i == w - 1):
                mm += mr
            else:
                j_mr = j * w + i + 1
                row_indices.append(m)
                col_indices.append(j_mr)
                data.append(mr)

            # ---------------------------------------------------------------------#

            # ---------------------------------------------------------------------#

            bl = cdiag * (d[j + 2, i + 1] + b[j + 2, i + 1] + d[j + 1, i] + b[j + 1, i])

            if (i == 0 or j == h - 1):
                mm += bl
            else:
                j_bl = (j + 1) * w + i - 1
                row_indices.append(m)
                col_indices.append(j_bl)
                data.append(bl)

            # ---------------------------------------------------------------------#

            bm = cver * (c[j + 2, i + 1] - d[j + 2, i + 1] + c[j + 1, i + 1] - d[j + 1, i + 1])

            if (j == h - 1):
                mm += bm
            else:
                j_bm = (j + 1) * w + i
                row_indices.append(m)
                col_indices.append(j_bm)
                data.append(bm)

            # ---------------------------------------------------------------------#

            br = cdiag * (d[j + 2, i + 1] - b[j + 2, i + 1] + d[j + 1, i + 2] - b[j + 1, i + 2])

            if (i == w - 1 or j == h - 1):
                mm += br
            else:
                j_br = (j + 1) * w + i + 1
                row_indices.append(m)
                col_indices.append(j_br)
                data.append(br)

            # ---------------------------------------------------------------------#

            mm += (-1) * (tl + tm + tr + ml + mr + bl + bm + br)
            row_indices.append(m)
            col_indices.append(m)
            data.append(mm)

        A = sp.csr_matrix((data, (row_indices, col_indices)))
        print("System matrix built")
        # ---------------------------------------------------------------------#

        # perform one diffusion step
        # ---------------------------------------------------------------------#
        I = sp.eye(w * h)

        for k in range(K):

            # implicit step with reaction term
            if alpha > 0:
                matrix = I - alpha / (tau + alpha) * tau * A
                vector = 1 / (alpha + tau) * (alpha * RGB[k] + tau * RGB_image[k])
                RGB[k] = sp.linalg.cg(matrix, vectorize(vector), maxiter=1000, rtol=10e-3)[0]
            # ---------------------------------------------------------------------#

            # implicit step without reaction term
            else:
                matrix = I - tau * A
                RGB[k] = sp.linalg.cg(matrix, vectorize(RGB[k]), maxiter=1000, rtol=10e-3)[0]
            # ---------------------------------------------------------------------#
   
            # rescale intensity, reshape
            RGB[k] = RGB[k].astype(np.uint8).reshape(h, w)
            # ---------------------------------------------------------------------#

        curr = cv2.merge(RGB) if K > 1 else cv2.cvtColor(RGB[0], cv2.COLOR_GRAY2BGR)
        
    return curr


def main(input_path, output_folder, *args):
    # Load the input image
    image = cv2.imread(input_path)

    if image is None:
        print("Error: Unable to read the input image.")
        return

    # Call the EED function with provided arguments
    processed_image = EED(image, *args)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed image to the output folder
    name = "EED_ani" + "".join(str(elem) + " | " for elem in args)
    output_path = os.path.join(output_folder, name + ".jpg")
    cv2.imshow(name, processed_image)
    cv2.waitKey(0)
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved at: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py <input_image_path> <output_folder> <plicit> <alpha> <tau> <sigma>  <quantile> <iterations> ")
    else:
        input_path = sys.argv[1]
        output_folder = sys.argv[2]
        plicit = int(sys.argv[3])
        alpha = float(sys.argv[4])
        tau = float(sys.argv[3 + 2])
        sigma = int(sys.argv[3 + 3])
        quantile = float(sys.argv[7])
        iterations = int(sys.argv[8])

        main(input_path, output_folder, plicit, alpha, tau, 1, 1, sigma, quantile, iterations)
