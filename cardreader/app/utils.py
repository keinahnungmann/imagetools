import numpy as np
import cv2

# Courtesy of https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin((diff ** 2).sum(axis=0))]
    rect[3] = pts[np.argmax((diff ** 2).sum(axis=0))]
    return rect

def four_point_transform(image, pts):

    (tl, tr, br, bl) = pts
    widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthT = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    heightR = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightL = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    
    maxheight = int(max(heightR, heightL))
    maxwidth = int(max(widthT, widthB))
    minheight = int(min(heightR, heightL))
    minwidth = int(min(widthT, widthB))
    
    dst = np.array([
        [0, 0],
        [widthT-1, 0],
        [widthB-1, heightR-1],
        [0, heightL-1]], dtype= 'float32')
    M = cv2.getPerspectiveTransform(pts	, dst)
    warped = cv2.warpPerspective(image, M, (maxwidth, maxheight))
    return warped

# -----------------------------------------------------------------------------

#Takes card image, returns card image without boundary
#Loaded image size will always be approx. 707x1031
def remove_boundary(image):
    
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #find desired contour
    for c in contours:
        
        rect = cv2.minAreaRect(c)
        w,h = rect[1]
        
        #filter small contours
        if w < 680 and w > 500 and h < 1000 and h > 800:
        
            #filter by aspect ratio
            aspect_ratio = 82/55
            aspect = max(w,h)/min(w,h)
            
            if abs(aspect - aspect_ratio) < 0.05:
            
                pts = cv2.boxPoints(rect)
                card = four_point_transform(image, pts)
                card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return card

