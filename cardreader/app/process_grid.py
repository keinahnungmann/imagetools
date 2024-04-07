import cv2
import os
import numpy as np
import sys
sys.path.append('/home/ben/Desktop/imagetools/EED')
sys.path.append('/home/ben/Desktop/imagetools/CED')
from EED_gradmag import EED
from ced_raw import CED

from yugiohcard import YuGiOhCard
from utils import four_point_transform
from deskew import determine_skew
from skimage.transform import rotate
import json
import difflib
from PIL import Image
import argparse
import pytesseract

#slice
#for all cards in sliced
#-----------------------------------------------------------------------------#
#arguments: image of 9/10 cards, output folder
#Crops cards and stores them in output folder
#returns: number of cards cropped

def process_grid(image_path):
    
    #------------------------------------------------------------------------------#
    lib = load_lib()
    
    #------------------------------------------------------------------------------#
    
    #extract cards from card grid
    images_path = slice_grid(image_path)[0]
    struct_dir = '/home/ben/Desktop/imagetools/cardreader/cards/structs'
    #iterate over all cards/card paths in images_path
    for card in img_generator(images_path):
        
        yugiohcard = YuGiOhCard(card)
        yugiohcard.set_attributes(lib)
        yugiohcard.save_to_file(struct_dir)


def img_generator(directory):
    for filename in os.listdir(directory):
        picture = filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff',
        '.bmp', '.gif'))
        if picture:
            yield os.path.join(directory, filename)

def slice_grid(image_path):

    #initialize
    output_path = '/home/ben/Desktop/imagetools/cardreader/cards/single'
    w_res = 2552//2
    h_res = 3508//2
    img = cv2.imread(image_path)
    
    #PREPROCESSING: Resize, thresh and diffuse for cleaner edges
    #------------------------------------------------------------------------------#
    img_resized = cv2.resize(img, (w_res, h_res))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    print("initiate diffusion")
    #diffused = EED(gray, 0, 32.0, 4.0, 1.0, 1.0, 3, 1.0, 4)
    #cv2.imshow("diffused" , diffused)
    #cv2.waitKey(0)		
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("",thresh)
    cv2.waitKey(0)
    #------------------------------------------------------------------------------#
    #CONTOUR EXTRACTION:
    #------------------------------------------------------------------------------#
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tbd = list()
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05* peri, True)
        if len(approx) == 4:
            tbd.append(approx)
    #------------------------------------------------------------------------------#



    #determine cards among 4 point contours
    #------------------------------------------------------------------------------#
    new_contours = list()
    l = 1
    for c in tbd:
        print("contours found")
            
        #x,y,w,h = cv2.boundingRect(c)
        cropper = cv2.minAreaRect(c)
        w,h = cropper[1]	#width / height
        
	#3x3 cards on image of size (638, 877) give the measures:
	#------------------------------------------------------------------------------#
        least_straight = (w >= (1/4) * w_res and h >= (1/4) * h_res)
        most_straight = (w <= (1/3) * w_res and h <= (1/3) * h_res)
        straight = least_straight and most_straight
        
        least_crooked = (h >= (1/4) * w_res and w >= (1/4) * h_res)
        most_crooked = (h <= (1/3) * w_res and w <= (1/3) * h_res)
        crooked = least_crooked and most_crooked
        #crooked = (w >= (1/3) * w_res and h >= (1/7) * h_res)
        if (straight or crooked	): #or crooked
            
            #crop card
            #------------------------------------------------------------------------------#
            pts = cv2.boxPoints(cropper)
            print(pts)	
            card = four_point_transform(img, 2*pts)
            if straight:
                card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #card = img[int(8*y) : int(8*(y + h)), int(8*x) : int(8*(x + w))]
            #------------------------------------------------------------------------------#
            
            #store card 
            #------------------------------------------------------------------------------#
            sliced_name = image_path.split("/")[-1].split(".")[0] + str(l) + '.jpg'
            file_path = os.path.join(output_path, sliced_name)
            cv2.imwrite(file_path, card)
            
            print("image saved")
            
            l += 1
            #------------------------------------------------------------------------------#
            
    if l<9:
        print("Less than 9 cards detected")
                
    return (output_path, l-1)    
      
      
#Takes image_path, creates new yugioh card structure with only image attr. set, stores to file         
def card_from_image(image_path): 
    
    #------------------------------------------------------------------------------#         
    card = yugiohcard(image_path)
    card.init_card()
    #------------------------------------------------------------------------------#
    
    return card.save_to_file()     
    
   
def load_lib():
    #------------------------------------------------------------------------------#    
    with open("cardinfo.php", "r") as file:
    	data = json.load(file)

    names = [entry["name"] for entry in data["data"]]
    boosters = []
    for entry in data['data']:
    	# Check if the entry has 'card_sets' key and it's not empty
    	if 'card_sets' in entry and entry['card_sets']:
        	# Extract the name of the first booster pack (assuming there is at least one)
        	boosters.append(entry['card_sets'][0]['set_name'])
    	else:
        	# If 'card_sets' key is missing or empty, add a placeholder value
        	boosters.append("Unknown")
        	
        	
        	
    editions = ["1st Edition", "1te Edition", "LIMITIERTE AUFLAGE"]
    editions.append("LIMITED EDITION")
    editions.append("SPEED DUEL")
    editions.append("SPEED DUELL")
    
    lib = (names, boosters, editions)
    
    return lib






    
           
