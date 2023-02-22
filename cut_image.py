import argparse
import os
#import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument("--image", default="cat.0", type=str)
parser.add_argument("--path", default="./sample/", type=str)
parser.add_argument("--col", default=2, type=int)
parser.add_argument("--row", default=2, type=int)

args = parser.parse_args()
#image = mpimg.imread(args.path + args.image + '.jpg')
image = Image.open(args.path + args.image + '.jpg')
image = np.array(image)

_row, _col, _ = image.shape

if _row % 2 == 1:
    image = image[:-1,:,:]
if _col % 2 == 1:
    image = image[:,:-1,:]

_row = _row // 2
_col = _col // 2

images = [image[:_row,:_col,:], image[:_row,_col:,:], image[_row:,:_col,:], image[_row:,_col:,:]]
random.shuffle(images)
for i in range(4): 
    if np.random.rand() >= 0.5:
        images[i] = images[i][:,::-1,:]
    if np.random.rand() >= 0.5:
        images[i] = images[i][::-1,:,:]
    if np.random.rand() >= 0.5:
        images[i] = np.rot90(images[i], 1, axes = (0, 1))
    
    np.save(f'./save/{args.image}_{i}.npy', images[i])

    # save_image = Image.fromarray(images[i])
    # save_image.save(f'./save/{args.image}_{i}.jpg','JPEG')
