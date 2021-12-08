import numpy as np
import torch
from torch import nn, optim
import os
from dnn import DNN
import cv2


def load_data():
    pass


def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst


# path = "F:/BaiduNetdiskDownload/trainData"
# folders = os.listdir(path)
# print(folders)

# data = np.load("G:/new_gesture/c/5.npy")
data = np.load("C:/Users/xiaooo/Desktop/1.npy")
frames = data.transpose(1, 2, 3, 0)

a = np.random.rand() * 2 + 0.3
seq = []
for frame in frames:
    print(frame.shape)
    cv2.imshow("123",frame)
    cv2.waitKey(2)
    f = imgBrightness(frame[:, :, 0:3], a, 3)
    cv2.imshow("123", f)
    cv2.waitKey(2)
    res = f.transpose(2, 0, 1)

    temp = [f[:, :, 0], f[:, :, 1], f[:, :, 2], frame[:, :, 3]]
    # print(frame.shape)
    # cv2.imshow("123",f)
    # cv2.waitKey(2)
    seq.append(temp)
res = np.array(seq)
result = res.transpose(1, 0, 2, 3)

# frame = data.transpose(1, 2, 0)

# for i in range(256):
#     for j in range(256):
#         frame[i][j] = np.std(frame[i][j])
# print(frame[0][0])
# var = np.zeros((256, 256))

# for i in range(256):
#     for j in range(256):
#         total = []
#         for frame in data:
#             total.append(frame[i][j])
#         # print(np.var(total), np.std(total))
#         num = np.std(total)
#         if num > 100:
#             var[i][j] = 1

# cv2.imshow("123", var)
# cv2.waitKey()
# print(total)
