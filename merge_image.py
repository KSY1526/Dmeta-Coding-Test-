import argparse
import os
from PIL import Image
import numpy as np
import sys
from itertools import permutations

parser = argparse.ArgumentParser()

parser.add_argument("--image", default="cat.0", type=str)
parser.add_argument("--path", default="./sample/", type=str)
parser.add_argument("--nums", default=2, type=int)

args = parser.parse_args()

num = args.nums * args.nums
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

_min = sys.maxsize
new_images = []

images = [i.astype('i') for i in images]

if args.nums == 2:

    def get_measure(image_0, image_1, image_2, image_3):
        _sum = 0
        _sum += np.sum(np.abs(image_0[:,-1,:] - image_1[:,0,:]))
        _sum += np.sum(np.abs(image_0[-1] - image_2[0]))
        _sum += np.sum(np.abs(image_1[-1] - image_3[0]))
        _sum += np.sum(np.abs(image_2[:,-1,:] - image_3[:,0,:]))

        return _sum

    idxs = [i for i in permutations([1,2,3], 3)]

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


elif args.nums == 3:

    def get_measure(image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8):
        _sum = 0
        _sum += np.sum(np.abs(image_0[:,-1,:] - image_1[:,0,:]))
        _sum += np.sum(np.abs(image_1[:,-1,:] - image_2[:,0,:]))
        _sum += np.sum(np.abs(image_3[:,-1,:] - image_4[:,0,:]))
        _sum += np.sum(np.abs(image_4[:,-1,:] - image_5[:,0,:]))
        _sum += np.sum(np.abs(image_6[:,-1,:] - image_7[:,0,:]))
        _sum += np.sum(np.abs(image_7[:,-1,:] - image_8[:,0,:]))

        _sum += np.sum(np.abs(image_0[-1] - image_3[0]))
        _sum += np.sum(np.abs(image_1[-1] - image_4[0]))
        _sum += np.sum(np.abs(image_2[-1] - image_5[0]))
        _sum += np.sum(np.abs(image_3[-1] - image_6[0]))
        _sum += np.sum(np.abs(image_4[-1] - image_7[0]))
        _sum += np.sum(np.abs(image_5[-1] - image_8[0]))

        return _sum
    
    idxs = [i for i in permutations([1,2,3,4,5,6,7,8], 8)]
    print(len(idxs))
    cnt = 0
    for idx in idxs:
        for x0 in range(4):
            image_0 = _transform(images[0],x0)
            for x1 in range(4):
                image_1 = _transform(images[idx[0]],x1)
                for x2 in range(4):
                    image_2 = _transform(images[idx[1]],x2)
                    for x3 in range(4):
                        image_3 = _transform(images[idx[2]],x3)
                        for x4 in range(4):
                            image_4 = _transform(images[idx[3]],x4)
                            for x5 in range(4):
                                image_5 = _transform(images[idx[4]],x5)
                                for x6 in range(4):
                                    image_6 = _transform(images[idx[5]],x6)
                                    for x7 in range(4):
                                        image_7 = _transform(images[idx[6]],x7)
                                        for x8 in range(4):
                                            image_8 = _transform(images[idx[7]],x8)
                                            _sum = get_measure(image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8)
                                            if _min > _sum:
                                                _min = _sum
                                                new_images = [image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        cnt += 1
        print(cnt)

    tem1 = np.concatenate((new_images[0],new_images[1],new_images[2]),axis=1)
    tem2 = np.concatenate((new_images[3],new_images[4],new_images[5]),axis=1)
    tem2 = np.concatenate((new_images[6],new_images[7],new_images[8]),axis=1)
    new_image = np.concatenate((tem1, tem2, tem3),axis=0)

real_image = Image.open(args.path + args.image + '.jpg')
real_image = np.array(real_image)

def img_check(real_image, new_image):
    if real_image.shape[0] == new_image.shape[0]:
        for i in range(4):
            if np.min(np.abs(real_image - _transform(new_image, i))) == 0:
                save_image = Image.fromarray(_transform(new_image.astype(np.uint8), i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

    else:
        new_image = np.rot90(new_image, 1, axes = (0, 1))
        for i in range(4):
            if np.max(np.abs(real_image - _transform(new_image, i))) == 0:
                save_image = Image.fromarray(_transform(new_image.astype(np.uint8), i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

_row_real, _col_real, _ = real_image.shape

if args.nums == 2:
    if _row_real % 2 == 1:
        image = image[:-1,:,:]
    if _col_real % 2 == 1:
        image = image[:,:-1,:]

elif args.nums == 3:
    if _row_real % 3 == 1:
        image = image[:-1,:,:]
    elif _row_real % 3 == 2:
        image = image[:-2,:,:]
    if _col_real % 3 == 1:
        image = image[:,:-1,:]
    elif _col_real % 3 == 2:
        image = image[:,:-2,:]

print(img_check(real_image, new_image))
