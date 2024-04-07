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

def cropminarearect(image, c):
    # init
    rect = cv2.minAreaRect(c)
    
    # rotate image to that minarearect is axes-aligned
    # -----------------------------------------------------------------------------

    angle = rect[2]
    matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
    
    # rotate image
    img_rot = cv2.warpAffine(image, matrix, (3508, 2552))
    
    # rotate contour
    c_hom = np.hstack((c.reshape(4, 2), np.ones((4, 1), dtype=np.uint64)))
    c_rot = np.dot(matrix, c_hom.T).T
    
    # then crop boundingrect
    x, y, w, h = cv2.boundingRect(c_rot)
    return image[y: y+h, x: x+w]

