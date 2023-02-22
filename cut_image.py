import argparse
import os
from PIL import Image
import numpy as np
import random

parser = argparse.ArgumentParser()

# 이미지 이름을 넣습니다.
parser.add_argument("--image", default="cat.0", type=str)
parser.add_argument("--path", default="./sample/", type=str)
parser.add_argument("--nums", default=2, type=int)

args = parser.parse_args()
image = Image.open(args.path + args.image + '.jpg')
image = np.array(image)

_row, _col, _ = image.shape

if args.nums == 2:
    if _row % 2 == 1:
        image = image[:-1,:,:]
    if _col % 2 == 1:
        image = image[:,:-1,:]

elif args.nums == 3:
    if _row % 3 == 1:
        image = image[:-1,:,:]
    elif _row % 3 == 2:
        image = image[:-2,:,:]
    if _col % 3 == 1:
        image = image[:,:-1,:]
    elif _col % 3 == 2:
        image = image[:,:-2,:]

_row = _row // args.nums
_col = _col // args.nums

if args.nums == 2:
    images = [image[:_row,:_col,:], image[:_row,_col:,:], image[_row:,:_col,:], image[_row:,_col:,:]]

elif args.nums == 3:
    images = [image[:_row,:_col,:], image[:_row,_col:_col*2,:], image[:_row,_col*2:,:], 
    image[_row:_row*2,:_col,:], image[:_row:_row*2,_col:_col*2,:], image[_row:_row*2,_col*2:,:],
    image[_row*2:,:_col,:], image[_row*2:,_col:_col*2,:], image[_row*2:,_col*2:,:]]

random.shuffle(images)

for i in range(args.nums * args.nums): 
    if np.random.rand() >= 0.5:
        images[i] = images[i][:,::-1,:]
    if np.random.rand() >= 0.5:
        images[i] = images[i][::-1,:,:]
    if np.random.rand() >= 0.5:
        images[i] = np.rot90(images[i], 1, axes = (0, 1))
    
    np.save(f'./save/{args.image}_{i}.npy', images[i])