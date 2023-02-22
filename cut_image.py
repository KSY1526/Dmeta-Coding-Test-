import argparse
import os
from PIL import Image
import numpy as np
import random

parser = argparse.ArgumentParser()

# 이미지 이름
parser.add_argument("--image", default="cat.0", type=str)
# 이미지가 존재하는 경로
parser.add_argument("--path", default="./sample/", type=str)
# 이미지를 분할하는 크기
parser.add_argument("--nums", default=2, type=int)

args = parser.parse_args()
image = Image.open(args.path + args.image + '.jpg')
image = np.array(image)

_row, _col, _ = image.shape

# 이미지 분할 시 남는 부분 미리 제거
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


# 이미지를 분할합니다.
_row = _row // args.nums
_col = _col // args.nums

if args.nums == 2:
    images = [image[:_row,:_col,:], image[:_row,_col:,:], image[_row:,:_col,:], image[_row:,_col:,:]]

elif args.nums == 3:
    images = [image[:_row,:_col,:], image[:_row,_col:_col*2,:], image[:_row,_col*2:,:], 
    image[_row:_row*2,:_col,:], image[:_row:_row*2,_col:_col*2,:], image[_row:_row*2,_col*2:,:],
    image[_row*2:,:_col,:], image[_row*2:,_col:_col*2,:], image[_row*2:,_col*2:,:]]

# 이미지 저장 순서를 랜덤하게 섞어 이미지 이름으로 위치를 추정할 수 없게 합니다.
random.shuffle(images)

# mirroring, flipping, rotation을 50% 확률로 모든 이미지에 적용합니다.
for i in range(args.nums * args.nums): 
    if np.random.rand() >= 0.5:
        images[i] = images[i][:,::-1,:]
    if np.random.rand() >= 0.5:
        images[i] = images[i][::-1,:,:]
    if np.random.rand() >= 0.5:
        # np.rot90(img, 1, axes = (0, 1)) : 90도 회전 (2번째 인자에 따라 180, 270 응용 가능)
        images[i] = np.rot90(images[i], 1, axes = (0, 1))
    
    # 넘파이 값으로 저장 (PIL를 이용하여 저장하면 일부 오차가 생김을 확인)
    np.save(f'./save/{args.image}_{i}.npy', images[i])