import argparse
import os
from PIL import Image
import numpy as np
import sys
from itertools import permutations

parser = argparse.ArgumentParser()

# 이미지 이름
parser.add_argument("--image", default="cat.0", type=str)
# 이미지가 존재하는 경로
parser.add_argument("--path", default="./sample/", type=str)
# 이미지를 분할하는 크기
parser.add_argument("--nums", default=2, type=int)

args = parser.parse_args()

# num : 총 분할된 개수
num = args.nums * args.nums
# images : 분할 이미지 담는 배열
images = []
for i in range(num):
    image = np.load(f'./save/{args.image}_{i}.npy')
    images.append(image)

# 이미지를 가로가 긴 형태로 통일합니다.
# 1) 일반적으로 가로가 긴 형태의 이미지가 많이 존재하며
# 2) rotation한 것은 shape만 다시 맞춰주면 mirroring, flipping을 이용해 같은 이미지을 복원할 수 있습니다.
# (반대 방향으로 rotation 해도 mirroring, flipping 하면 원상 복구 됨. 어짜피 mirroring, flipping은 적용 됬을 수도 있어서.) 
_row, _col, _ = images[0].shape

if _col > _row:
    _row, _col = _col, _row
    images[0] = np.rot90(images[0], 1, axes = (0, 1))

for i in range(1, num):
    if images[i].shape[0] != _row:
        images[i] = np.rot90(images[i], 1, axes = (0, 1))


def _transform(image, i):
    """
    i 값에 따라 이미지를 mirroring, flipping 하여 돌려주는 함수입니다.
    """    
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

# 연산이 가능하도록 int32로 형변환 해줍니다.
images = [i.astype('i') for i in images]

# 2 * 2 분할인 경우
if args.nums == 2:

    def get_measure(image_0, image_1, image_2, image_3):
        """
        분할된 이미지들의 edge간 차이를 measure로 이용합니다.
        edge간 차이가 작을 수록 이미지가 자연스럽다고 판단합니다.
        edge간 차이의 합으로 정의한 measure를 구해주는 함수입니다.
        """        
        _sum = 0
        _sum += np.sum(np.abs(image_0[:,-1,:] - image_1[:,0,:]))
        _sum += np.sum(np.abs(image_0[-1] - image_2[0]))
        _sum += np.sum(np.abs(image_1[-1] - image_3[0]))
        _sum += np.sum(np.abs(image_2[:,-1,:] - image_3[:,0,:]))

        return _sum

    # 이미지 붙어있는 순서를 순열을 이용하여 구했습니다.
    # 원순열 성질을 이용하면 겹치는 연산을 제거할 수 있습니다. (0을 고정하고 나머지 값 배정하기, 4! => 3!)
    # [[0,1], [2,3]], [[0,1], [3,2]], [[0,2], [1,3]], [[0,2], [3,1]], [[0,3], [1,2]], [[0,3], [2,1]]
    idxs = [i for i in permutations([1,2,3], 3)]

    # for x0 in range(4) => x0는 _transform에서 mirroring, flipping 여부를 선택하는 값입니다.
    # 반복문을 이용해 모든 mirroring, flipping (적용/비적용) 경우를 구하려고 합니다.
    for idx in idxs:
        for x0 in range(4):
            image_0 = _transform(images[0],x0)
            for x1 in range(4):
                image_1 = _transform(images[idx[0]],x1)
                for x2 in range(4):
                    image_2 = _transform(images[idx[1]],x2)
                    for x3 in range(4):
                        image_3 = _transform(images[idx[2]],x3)
                        # get_measure : 해당방식으로 병합된 이미지의 measure 구하는 함수.
                        _sum = get_measure(image_0, image_1, image_2, image_3)
                        # 만약 measure가 최솟값이라면 (작을 수록 좋음)
                        if _min > _sum:
                            _min = _sum
                            new_images = [image_0, image_1, image_2, image_3]

    # measure가 최소인 경우 new_images로 저장되어있습니다.
    # new_images를 이용해 measure가 최소인 이미지를 합쳐줍니다.
    tem1 = np.concatenate((new_images[0],new_images[1]),axis=1)
    tem2 = np.concatenate((new_images[2],new_images[3]),axis=1)
    new_image = np.concatenate((tem1, tem2),axis=0)

# 3*3 분할인 경우
elif args.nums == 3:
    """
    모든 케이스를 확인하려면 다음과 같은 시간복잡도를 가집니다.
    9 * 7! * 4^9 * (get_measure 함수 연산시간) = (11,890,851,840) * (get_measure 함수 연산시간)
    1) 한 개를 제외한 8개의 분할 이미지의 위치를 정해야 합니다. 
    이 때 가운데 이미지를 구하는 경우의 수도 존재합니다. (9 * 7!)
    2) 모든 분할 이미지 마다 mirroring, flipping(적용/비적용) 경우를 따져야합니다.
    => 연산 시간이 너무 많이 걸립니다. 일부 개선된다 해도 부담되는 정도입니다.
    즉 빠른 시간 내 작업이 불가능합니다.
    """   
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
    _set = set([0,1,2,3,4,5,6,7,8])
    # 가운데 값 먼저 정하기.
    for _mid in range(9):
        for x0 in range(4):
            image_0 = _transform(images[_mid],x0)
        idxs = [i for i in permutations(list(_set - set([_mid])), 8)]
        for idx in idxs:
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

# 정답 이미지를 다운로드 받습니다. (평가로만 사용)
real_image = Image.open(args.path + args.image + '.jpg')
real_image = np.array(real_image)

def img_check(real_image, new_image):
    """
    예측 이미지가 실제 이미지와 일치하는지 확인하는 함수
    Returns:
        bool: (True, False)
    """
    # 만약 가로, 세로 shape이 일치하면
    if real_image.shape[0] == new_image.shape[0]:
        # shape만 맞으면 mirroring, flipping을 통해 rotation 없이 복구할 수 있습니다.
        for i in range(4):
            # 모든 넘파이 값이 일치한다면 == 정답
            if np.min(np.abs(real_image - _transform(new_image, i))) == 0:
                save_image = Image.fromarray(_transform(new_image.astype(np.uint8), i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

    # 만약 가로, 세로 shape이 일치하지 않으면
    else:
        # 90도 회전을 하여 shape을 맞춰줍니다.
        new_image = np.rot90(new_image, 1, axes = (0, 1))
        # shape만 맞으면 mirroring, flipping을 통해 rotation 없이 복구할 수 있습니다.
        for i in range(4):
            if np.max(np.abs(real_image - _transform(new_image, i))) == 0:
                save_image = Image.fromarray(_transform(new_image.astype(np.uint8), i))
                save_image.save(f'./save/{args.image}_success.jpg','JPEG')
                return True

        save_image = Image.fromarray(new_image)
        save_image.save(f'./save/{args.image}_fail.jpg','JPEG')
        return False

# 이미지 분할 시 남는 부분 미리 제거
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

# 예측 이미지가 실제 이미지와 일치하는지 확인하는 함수 실행
print(img_check(real_image, new_image))
