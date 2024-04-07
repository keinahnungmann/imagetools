import cv2
import os
import numpy as np
import sys
sys.path.append('/home/ben/Desktop/imagetools/EED')
import EED_gradmag
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

    #parse arguments
    #------------------------------------------------------------------------------#
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
    #------------------------------------------------------------------------------#
    
    
    
    #init
    #------------------------------------------------------------------------------#
    pytesseract.pytesseract.tesseract_cmd = args.tesseract

    f = open("./cardinfo.php", "rb")
    card_data = json.loads(f.read())
    f.close()

    card_names = [i["name"].upper() for i in card_data["data"]]
    edition = ["1st Edition", ""]
    img = cv2.imread(args.image)
    H,W,K = img.shape
    print(W)
    print(H)
    
    output_path = "/home/ben/Desktop/imagetools/cardreader/cards/single"
    #------------------------------------------------------------------------------#



    #MAIN IMAGE PREPROCESSING
    #------------------------------------------------------------------------------#
    
    #Resize, thresh, dilate and extract contours
    #------------------------------------------------------------------------------#
    img_resized = cv2.resize(img, (319, 438))
    blur = cv2.GaussianBlur(img_resized, (1,1), 0)
    gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny(gray_img, 50, 100)
    thresh = cv2.merge((thresh, thresh, thresh))
    diffused = EED_gradmag.EED(thresh, 0, 0.1, 4, 1, 1, 3, 0.9, 1)
    diffused = cv2.cvtColor(diffused, cv2.COLOR_BGR2GRAY)
    #diffused = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)	
    cv2.imshow("diffused", diffused)
    cv2.waitKey(0)
    
    contours, hierarchy = cv2.findContours(diffused, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #------------------------------------------------------------------------------#


    #determine all 4 point contours
    #------------------------------------------------------------------------------#
    tbd = list()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            tbd.append(approx)
    #------------------------------------------------------------------------------#


    #determine cards among 4 point contours
    #------------------------------------------------------------------------------#
    new_contours = list()
    l = 1
    for c in tbd:
        print("contours found")
            
        x,y,w,h = cv2.boundingRect(c)
	#3x3 cards on image of size (638, 877) give the measures:
	#------------------------------------------------------------------------------#
        if (w >= (1/6) * 319 and h >= (5/28) * 438): 
        
            #crop and save to file
            #------------------------------------------------------------------------------#
            card = img[int(8*y) : int(8*(y + h)), int(8*x) : int(8*(x + w))]

            file_name = args.image.split("/")[-1].split(".")[0] + "." + str(l)
            l += 1
            file_path = os.path.join(output_path, file_name + ".jpg")
            
            cv2.imwrite(file_path, card)
            print("image saved")
            cv2.imshow("saved:", card)
            cv2.waitKey(0)
            #------------------------------------------------------------------------------# 
             
            #crop card attributes
            #------------------------------------------------------------------------------#
            
            #init
            warped_h = card.shape[0]		#card width
            warped_w = card.shape[1]		#card height
            
            #name: upper:y = 3mm; lower:y = 10mm; left:x = 4mm; right:x = 48mm
            #------------------------------------------------------------------------------#
            name_t = int((3/86) * warped_h)
            name_b = int((10/86) * warped_h)
            name_l = int((4/58) * warped_w)
            name_r = int((45/58) * warped_w)
            
            croppedname = card[name_t:name_b, name_l:name_r]
            print(croppedname.shape)
            cv2.imshow("name:", croppedname)
            #name_roi = Image.fromarray(croppedname.astype(np.uint8))
            #------------------------------------------------------------------------------#
            
            #booster: upper:y = 65mm; lower:y = 57mm; left:x = 40mm; right: x = 55mm
            #------------------------------------------------------------------------------#
            booster_t = int((61/86) * warped_h)
            booster_b = int((64/86) * warped_h)
            booster_l = int((42/59) * warped_w)
            booster_r = int((53/59) * warped_w)
            
            cropped_booster = card[booster_t:booster_b, booster_l:booster_r]
            cv2.imshow("booster", cropped_booster)
            #booster_roi = Image.fromarray(cropped_booster.astype(np.uint8))
            #------------------------------------------------------------------------------#
            
            #edition:
            #edition bottom at bottom:
            #upper:y = 81mm; lower:y = 84mm; left:x = 10mm; right:x = 32mm 
            edition_t = int((81/86) * warped_h)
            edition_b = int((84/86) * warped_h)
            edition_l = int((10/58) * warped_w)
            edition_r = int((32/86) * warped_w)
            
            cropped_edition = card[edition_t:edition_b, edition_l:edition_r]
            cv2.imshow("edition", cropped_edition)
            #editionb_roi = Image.fromarray(cropped_edition.astype(np.uint8))
            
            #edition at booster height // "SPEED DUEL"
            #upper,lower: cf. booster, left, right:7-29mm
            edition_t = int((57/86) * warped_h)
            edition_b = int((65/86) * warped_h)
            edition_l = int((7/58) * warped_w)
            edition_r = int((29/58) * warped_w)
            
            cropped_edition = card[edition_t:edition_b, edition_l:edition_r]
            cv2.imshow("edition1", cropped_edition)
            #editiont_roi = Image.fromarray(cropped_edition.astype(np.uint8))
            #------------------------------------------------------------------------------#

            




