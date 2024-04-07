import cv2
import os

def canny_edge_detection(image, threshold1=50, threshold2=100):
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def process_images(folder_path, output_folder=None, threshold1=50, threshold2=100):
    if output_folder is None:
        output_folder = folder_path
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                edges = canny_edge_detection(image, threshold1, threshold2)
                output_filename = os.path.splitext(filename)[0] + "_canny" + os.path.splitext(filename)[1]
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    folder_path = "./"
    process_images(folder_path, threshold1=50, threshold2=100)

