import argparse
import os
from PIL import Image
import numpy as np
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--image", default="cat.0", type=str)
parser.add_argument("--path", default="./sample/", type=str)
parser.add_argument("--col", default=2, type=int)
parser.add_argument("--row", default=2, type=int)

args = parser.parse_args()

num = args.col * args.row
images = []

for i in range(num):
    image = np.load(f'./save/{args.image}_{i}.npy')
    images.append(image)

_row, _col, _ = images[0].shape

if _col > _row:
    _row, _col = _col, _row
    images[0] = np.rot90(images[0], 1, axes = (0, 1))

for i in range(1, num):
    if images[i].shape[0] != _row:
        images[i] = np.rot90(images[i], 1, axes = (0, 1))


def _transform(image, i):
    if i == 0:
        return image
    elif i == 1:
        return image[:,::-1,:]
    elif i == 2:
        return image[::-1,:,:]
    elif i == 3:
        return image[::-1,::-1,:]

def get_measure(image_0, image_1, image_2, image_3):
    _sum = 0
    _sum += np.sum(np.abs(image_0[:,-1,:].astype('i') - image_1[:,0,:].astype('i')))
    _sum += np.sum(np.abs(image_0[-1].astype('i') - image_2[0].astype('i')))
    _sum += np.sum(np.abs(image_1[-1].astype('i') - image_3[0].astype('i')))
    _sum += np.sum(np.abs(image_2[:,-1,:].astype('i') - image_3[:,0,:].astype('i')))

    return _sum

_min = sys.maxsize
new_images = []

idxs = [(1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)]

for idx in idxs:
    for x0 in range(4):
        image_0 = _transform(images[0],x0)
        for x1 in range(4):
            image_1 = _transform(images[idx[0]],x1)
            for x2 in range(4):
                image_2 = _transform(images[idx[1]],x2)
                for x3 in range(4):
                    image_3 = _transform(images[idx[2]],x3)
                    _sum = get_measure(image_0, image_1, image_2, image_3)
                    if _min > _sum:
                        _min = _sum
                        new_images = [image_0, image_1, image_2, image_3]

tem1 = np.concatenate((new_images[0],new_images[1]),axis=1)
tem2 = np.concatenate((new_images[2],new_images[3]),axis=1)
new_image = np.concatenate((tem1, tem2),axis=0)

real_image = Image.open(args.path + args.image + '.jpg')
real_image = np.array(real_image)

def img_check(real_image, new_image):
    if real_image.shape[0] == new_image.shape[0]:
        for i in range(4):
            if np.min(np.abs(real_image.astype('i') - _transform(new_image, i).astype('i'))) == 0:
                save_image = Image.fromarray(_transform(new_image, i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

    else:
        new_image = np.rot90(new_image, 1, axes = (0, 1))
        for i in range(4):
            if np.max(np.abs(real_image.astype('i') - _transform(new_image, i).astype('i'))) == 0:
                save_image = Image.fromarray(_transform(new_image, i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

_row_real, _col_real, _ = real_image.shape
if _row_real % 2 == 1:
    real_image = real_image[:-1,:,:]
if _col_real % 2 == 1:
    real_image = real_image[:,:-1,:]

print(img_check(real_image, new_image))
