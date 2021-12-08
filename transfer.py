import numpy as np
import os
import torch
import cv2


def load_data():
    pass

def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst

def enhance(video, num=4):
    res = [[] for i in range(num)]
    flag = 128 - 128 % num
    for i, image in enumerate(video):
        if i == flag:
            break
        # print(image.shape)
        image = image[:, 105:375, 0:3]
        # print(image.shape)
        image = cv2.resize(image, (256, 256))
        # print(image.shape)
        # print(image.shape)
        res[i % num].append(image)

    return np.array(res)


path = '/home/xiaoguojian/theoreticalAcademic/DG-20_32'

new_path = '/home/xiaoguojian/theoreticalAcademic/DG-20_32_noisy'
folders = os.listdir(path)
#
# for folder in folders:
#     folders_path = path + '/' + folder
#     files = os.listdir(folders_path)
#     new_folders_path = new_path + '/' + folder
#     os.mkdir(new_folders_path)
#     i = 1
#     for file in files:
#         name = folders_path + '/' + file
#         data = np.load(name)
#         res = enhance(data)
#         for r in res:
#             temp = r.transpose(3, 0, 1, 2)
#             np.save(new_folders_path + '/' + str(i), temp)
#             i += 1
#             print(temp.shape)
#     print(folder + " finish")


for folder in folders:
    folders_path = path + '/' + folder
    files = os.listdir(folders_path)
    new_folders_path = new_path + '/' + folder
    os.mkdir(new_folders_path)
    i = 1
    for file in files:
        name = folders_path + '/' + file
        data = np.load(name)
        frames = data.transpose(1, 2, 3, 0)
        a = np.random.rand() * 2 + 0.3
        seq = []
        for frame in frames:
            # print(frame.shape)
            # cv2.imshow("123",frame)
            # cv2.waitKey(2)
            f = imgBrightness(frame, a, 3)
            res = f.transpose(2, 0, 1)
            temp = [f[:, :, 0], f[:, :, 1], f[:, :, 2], frame[:, :, 3]]
            # print(res.shape)
            seq.append(temp)
        res = np.array(seq)
        result = res.transpose(1, 0, 2, 3)

        np.save(new_folders_path + '/' + str(i), result)
        i += 1
        # print(res.shape)
    print(folder + " finish")

# data = np.load("rgbd_stream.npy")
# print(data.shape)
# for d in data:
#     res = d.transpose(2,0,1)
#     print(res.shape)
