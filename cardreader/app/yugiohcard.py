import cv2
import pytesseract
import difflib
import os


class YuGiOhCard:

    def __init__(self, image_path):
    
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.w = self.image.shape[1]
        self.h = self.image.shape[0]
        self.condition = ""
        self.number	 = 1
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
        x = [3/86, 4/58]
        
        
        y = [14/86, 45/58]
        
        self.name = self.read_area(x, y, names)
        #------------------------------------------------------------------------------#
        
        
    def set_booster(self, boosters): 
    
        #------------------------------------------------------------------------------#
        x = [61/86, 38/58]
        y = [67/86, 53/58]
        
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
        x = [61/86, 7/58]
        y = [67/86, 29/58]
        
        return self.read_area(x, y, editions)
    
    def get_edition2(self, editions):
    
        #------------------------------------------------------------------------------#
        x = [81/86, 10/58]
        y = [84/86, 32/58]
        
        return self.read_area(x, y, editions)
        
    
    
    
    #reads text in specified area (x = top left, y = bottom right, cv2-coordinates)
    def read_area(self,x,y,data): 
    
        #------------------------------------------------------------------------------#
        top = int(x[0]*self.h)
        bottom = int(y[0]*self.h)
        left = int(x[1]*self.w)
        right = int(y[1]*self.w)
        
        roi = self.image[top : bottom, left : right]
        cv2.imshow("roi", roi)
        cv2.waitKey(0)
        
        print(data)
        text = pytesseract.image_to_string(roi, config=r'--oem 1 --psm 6')
        text = difflib.get_close_matches(text, data, n=1)
        print(text)
        #------------------------------------------------------------------------------#
        
        return text

