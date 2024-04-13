import cv2
import os
import numpy as np
import sys 
sys.path.append("/home/ben/Desktop/imagetools/EED")
from EED_gradmag import EED
from utils import four_point_transform

#Takes card image, returns card image without boundary
#Loaded image size will always be approx. 707x1031
if __name__ == "__main__":
    
    image = cv2.imread("/home/ben/Desktop/imagetools/cardreader/cards/single/35.jpg", cv2.IMREAD_GRAYSCALE)
    #image = EED(image, 0, 32.0, 4.0, 1, 1, 11, 0.9, 10)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("diffused", image)
    cv2.waitKey(0)
    #thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 10)
    #thresh = cv2.erode(thresh, (3,3), iterations=1)
    thresh = cv2.Canny(image, 150,150)
    cv2.imshow("thresh", cv2.resize(thresh, (350,500)))
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #find desired contour
    for c in contours:
    
        
        
        rect = cv2.minAreaRect(c)
        w,h = rect[1]
        
        #filter small contours
        if w > 400 and h > 700:
        
            
            #filter by aspect ratio
            aspect_ratio = 82/55
            aspect = max(w,h)/min(w,h)
            
            if abs(aspect - aspect_ratio) < 0.01:
            
                pts = cv2.boxPoints(rect)
                card = four_point_transform(image, pts)
                card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow("card", card)
                cv2.waitKey(0)
                
    image = cv2.drawContours(image, contours,  0, (0, 255, 0), 6)            
    cv2.imshow("image", image)
    cv2.waitKey(0)            
                
    
   
