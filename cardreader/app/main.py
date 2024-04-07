
import sys
sys.path.append('/home/ben/Desktop/imagetools/cardreader/app')
from process_grid import img_generator
from process_grid import process_grid	


if __name__ == "__main__":
    
    grids_path = sys.argv[1]
    
    for grid in img_generator(grids_path):
        
        process_grid(grid)
