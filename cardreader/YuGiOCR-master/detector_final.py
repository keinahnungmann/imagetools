import cv2
import os
import numpy as np
import sys
from utils import four_point_transform
from deskew import determine_skew
from skimage.transform import rotate
import json
import difflib
from PIL import Image
import argparse
import pytesseract
#include <opencv2/ximgproc.hpp>


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to image")
    parser.add_argument("--tesseract", help="path to tesseract.exe")
    parser.add_argument("--visualize", help="boolean, visualize intermediate steps", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image) or not args.image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        print("Not a valid image path!")
        exit(0)

    if not os.path.exists(args.tesseract) or not args.tesseract.lower().endswith(("tesseract")):
        print("Not a valid tesseract executable!")
        exit(0)

    pytesseract.pytesseract.tesseract_cmd = args.tesseract

    f = open("./cardinfo.php", "rb")
    card_data = json.loads(f.read())
    f.close()

    card_names = [i["name"].upper() for i in card_data["data"]]
    edition = ["1st Edition", ""]
    img = cv2.imread(args.image)

    #MAIN IMAGE PREPROCESSING
    #Resizing to 800x800
    ratio = img.shape[0] / 800.
    img_resized = cv2.resize(img, (800, 800))
    coef_y = img.shape[0] / img_resized.shape[0]
    coef_x = img.shape[1] / img_resized.shape[1]
    
    #grayscale -> bluring -> canny thresholding -> dilation

    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (9,9), 0)
    thresh = cv2.Canny(blur, 50, 100)
    output_folder = "/home/ben/Schreibtisch/cardreader"
    output_path = os.path.join(output_folder, "4x4_thresh.jpg")
    cv2.imwrite(output_path, thresh)
    dilated = cv2.dilate(thresh, np.ones((7,7), dtype=np.int8))

    if args.visualize:
        cv2.imshow("blurred", blur)
        cv2.imshow("thresholded", thresh)
        cv2.imshow("dilated", dilated)
        cv2.waitKey(0)

    #CONTOUR EXTRACTION
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #FINDING 4-POINT CONTOURS BECAUSE THOSE MOST OFTEN RESEMBLE A RECTANGLE / CARD SHAPE
    tbd = list()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            tbd.append(approx)


    new_contours = list()

    #ITERATING THROUGH 4-POINT CONTOURS AND FINDING THE CARDS AMONG THEM
    for c in tbd:
        x,y,w,h = cv2.boundingRect(c)

        if (w >= 100 and h >= 100): 
            warped = four_point_transform(img_resized, c.reshape((4,2)))
            warped_w = warped.shape[0]
            warped_h = warped.shape[1]
            #print("Warped w: {}, Warped h: {}, ratio: {}, scaled ratio: {}".format(warped_w, warped_h, warped_h/warped_w, (warped_h*coef_y) / (warped_w*coef_x)))
            if (1.15 < (warped_h*coef_y) / (warped_w*coef_x) < 1.4):
                if args.visualize:
                    cv2.imshow("warped", warped)
                    cv2.waitKey(0)

                #DETERMINE ANGLE FOR DESKEWING
                angle = determine_skew(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

                if angle != 0 and angle is not None:
                    warped = rotate(warped, angle, resize=True)
                    
                #CROP THE TEXT BOX APPROXIMATELLY AND GET THE TEXT
                croppedname = warped[int(warped.shape[1]//(20)):int(warped.shape[1]//7), int(warped.shape[0]*0.059):int(warped.shape[0]*0.8)]
                name_roi = cv2.resize(croppedname, (800,70))
                
                croppedbooster = warped[int(warped.shape[1]*57/86.0):int(warped.shape[0]*65/86.0),  int(warped.shape[0]*40/59.0):int(warped.shape[0]*55/59.0)] ##
                booster_roi = cv2.resize(croppedbooster, (800,200))
                croppededition = warped[int(warped.shape[1]*81/86.0):int(warped.shape[0]*84/86.0),  int(warped.shape[0]*0.17):int(warped.shape[0]*0.4)] ##excellent
                edition_roi = cv2.resize(croppededition, (800, 200))
                
                #enhance scans
                booster_roi = cv2.cvtColor(booster_roi, cv2.COLOR_BGR2GRAY)
                
                
                         
                edition_roi = cv2.GaussianBlur(edition_roi, (1,1), 0) #23
                edition_roi = cv2.erode(edition_roi, np.ones((3,3), dtype=np.int8))

                if args.visualize:
                    cv2.imshow("extracted name", name_roi)
                    cv2.imshow("extracted booster", booster_roi)
                    cv2.imshow("extracted edition", edition_roi)
                    cv2.waitKey(0)

                name_roi = Image.fromarray((name_roi).astype(np.uint8))
                booster_roi = Image.fromarray((booster_roi).astype(np.uint8))
                edition_roi = Image.fromarray((edition_roi).astype(np.uint8))

                name = pytesseract.image_to_string(name_roi, config="--psm 7")
                print(name)
                booster = pytesseract.image_to_string(booster_roi, config="--psm 7")
                print(booster)
                edition = pytesseract.image_to_string(edition_roi, config="--psm 7")
                print(edition)

                #IF TEXT HAS BEEN FOUND, MEMORIZE THE CONTOUR AND THE TEXT
                name = difflib.get_close_matches(name.upper(), name, n=1)
                print(name)
                booster  = difflib.get_close_matches(booster.upper(), booster, n=1)
                print(booster)
                edition = difflib.get_close_matches(edition.upper(), edition, n=1)
            



    
