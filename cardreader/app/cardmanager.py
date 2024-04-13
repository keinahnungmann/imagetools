import sys
import os
import numpy as np
import scipy as sp
import cv2
import json
from yugiohcard import YuGiOhCard


def read_jsons(json_files):
    keys = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data:
                keys.append(key.lower())
    return keys
    
    
if __name__ == "__main__":

    #Lade Namen library
    ger = "/home/ben/Desktop/imagetools/cardreader/data/de.json"
    eng = "/home/ben/Desktop/imagetools/cardreader/data/en.json"
    names = [ger, eng]
    
    namen = read_jsons(names)
    
    printcode = "/home/ben/Desktop/imagetools/cardreader/data/printcode.json"
    boosters = read_jsons([printcode])
    print(boosters)
    
    editions = list(["1. edition", "1. auflage", "limited edition", "limitierte edition"])
    editions.append("speed duell")
    editions.append("speed duel")
    
    
    #Lade Edelstein Ritter, gecroppt ohne Rand	 
    file_path = '/home/ben/Desktop/imagetools/cardreader/cards/single/2-032.jpg'
    
    yugiohcard = YuGiOhCard(file_path)
    yugiohcard.set_name(namen)
    yugiohcard.set_booster(boosters)
    yugiohcard.set_edition(editions)
    yugiohcard.save_to_file("/home/ben/Desktop/imagetools/cardreader/cards/single")
