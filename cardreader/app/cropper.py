import os
import sys 
import cv2 
import numpy as np

sys.path.append('/home/ben/Desktop/imagetools/EED')

if __name__ == "__main__":
    image = cv2.imread("/home/ben/Desktop/imagetools/cardreader/cards/grids/-1.jpg",cv2.IMREAD_GRAYSCALE)
    
    output_path = "/home/ben/Desktop/imagetools/testing/contours"
    # Apply adaptive thresholding
    _,thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("canny", thresh)
    cv2.waitKey(0)
    file_path = os.path.join(output_path, "canny.jpg")
    cv2.imwrite(file_path, thresh)
    
    
    aspect_ratios = [86/58, 82/55]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #FINDING 4-POINT CONTOURS BECAUSE THOSE MOST OFTEN RESEMBLE A RECTANGLE / CARD SHAPE
    tbd = list()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            tbd.append(approx)
            
            
    for c in contours:
 
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        # Calculate the aspect ratio of the contour's bounding rectangle
        if w!=0 and h!=0:
            contour_aspect_ratio = max(w, h) / min(w, h)
            # Check if the aspect ratio matches the specified ratio
            outer = contour_aspect_ratio - aspect_ratios[0] < 0.02
            inner = contour_aspect_ratio - aspect_ratios[1] < 0.02
            if outer or inner:  # Adjust the threshold as needed#
                print(cv2.arcLength(c,True))
                pts = cv2.boxPoints(rect)
                pts = np.int0(pts)
                image = cv2.drawContours(image, [pts], 0, (0, 255, 0), 6)
        
    cv2.imshow("", image)
    cv2.waitKey(0)
    file_path = os.path.join(output_path, "contours.jpg")
    cv2.imwrite(file_path, image)

