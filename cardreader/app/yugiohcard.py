import cv2
import pytesseract
import difflib
import os
import sys
sys.path.append("/home/ben/Desktop/imagetools/EED")
from EED_gradmag import EED
from utils import remove_boundary


class YuGiOhCard:

    def __init__(self, image_path):
    
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.name = ""
        self.booster = ""
        self.edition = ""
        self.condition = ""
        self.number = 1
        self.language = "Deutsch"
        
        
    
        
    def save_to_file(self, file_dir):
        
        #------------------------------------------------------------------------------#
        filename = os.path.join(file_dir, f"{self.name}.txt")
        with open(filename, 'w') as file:
            file.write(f"Name: {self.name}\n")
            file.write(f"Booster: {self.booster}\n")
            file.write(f"Edition: {self.edition}\n")
            file.write(f"Condition: {self.condition}\n")
            file.write(f"Language: {self.language}\n")
            file.write(f"Number: {self.number}\n")
            file.write(f"Image: {self.image_path}\n")
        #------------------------------------------------------------------------------#
        
        
        
    @classmethod
    def load_from_file(cls, filename):
    
        #------------------------------------------------------------------------------#
        with open(filename, 'r') as file:
            attributes = {}
            for line in file:
                key, value = line.strip().split(': ')
                attributes[key.lower()] = value
            return cls(**attributes)
        #------------------------------------------------------------------------------#
       
       
       
    #extracts cards attributes from image and sets them
    def set_attributes(self, lib):
        
        #------------------------------------------------------------------------------#
        names,boosters,editions = lib
        self.set_name(names)
        self.set_booster(boosters)
        self.set_edition(editions)
        
        
        
    def set_name(self, names): 
    
        #------------------------------------------------------------------------------#
        x = [2/82, 2/55]
        
        
        y = [8/82, 47/55]
        
        self.name = self.read_area(x, y, names)
        #------------------------------------------------------------------------------#
        
        
    def set_booster(self, boosters): 
    
        #------------------------------------------------------------------------------#
        x = [59/82, 41/55]
        y = [62/82, 45/55]
        
        self.booster = self.read_area(x, y, boosters)
        #------------------------------------------------------------------------------#
        
    def set_edition(self, editions):
    
        #------------------------------------------------------------------------------#
        edition1 = self.get_edition1(editions)
        edition2 = self.get_edition2(editions)
        
        self.edition = edition2 + edition1
        #------------------------------------------------------------------------------#
        
        
   
    def get_edition1(self, editions): 
        
        #------------------------------------------------------------------------------#
        x = [59/82, 5/55]
        y = [62/82, 26/55]
        
        return self.read_area(x, y, editions)
    
    def get_edition2(self, editions):
    
        #------------------------------------------------------------------------------#
        x = [80/82, 9/55]
        y = [82/82, 29/55]
        
        return self.read_area(x, y, editions)
        
    
    
    
    #reads text in specified area (x = top left, y = bottom right, cv2-coordinates)
    def read_area(self,x,y,data): 
    
        #------------------------------------------------------------------------------#
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        inner = remove_boundary(gray)
        cv2.imshow("inner", inner)
        cv2.waitKey(0)
        h,w = inner.shape
        
        top = int(x[0]*h)
        bottom = int(y[0]*h)
        left = int(x[1]*w)
        right = int(y[1]*w)
        
        roi = inner[top : bottom, left : right]
        #roi = EED(roi, 0, 32.0, 4.0, 1, 1, 3, 0.7, 1)
        text = pytesseract.image_to_string(roi, config=r'--oem 1 --psm 6').lower()
        text = difflib.get_close_matches(text, data, n=1)
        #------------------------------------------------------------------------------#
        
        return text

