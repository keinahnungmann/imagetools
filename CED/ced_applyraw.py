import ced_raw

import cv2
import sys
import os

def main(input_path, output_folder, *args):
    # Load the input image
    image = cv2.imread(input_path)



    if image is None:
        print("Error: Unable to read the input image.")
        return

    # Call the CED function with provided arguments
    processed_image = ced_raw.CED(image, *args)
    name = "CED" +"".join(str(elem) + "|" for elem in args)
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
        tau = float(sys.argv[3])
        sigma = int(sys.argv[4])
        rho = int(sys.argv[5])
        C = float(sys.argv[6])
        alpha = float(sys.argv[7])
        iterations = int(sys.argv[7])

        main(input_path, output_folder, tau, 1, 1, sigma, rho, C, alpha, iterations)

