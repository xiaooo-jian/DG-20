import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
import pandas as pd
# Import OpenCV for easy image rendering
import cv2
import os


def read_bag(path, name="rgbd_stream"):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    colorizer = rs.colorizer(2)

    # cv2.namedWindow("Stream", cv2.WINDOW_AUTOSIZE)

    rgbd_stream = []
    try:
        # while True:
        for i in range(128):
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # print(depth_color_image.shape,color_image.shape)

            # images = np.vstack((color_image, depth_color_image))  #  效果展示
            # cv2.imshow("!23",images)
            # cv2.waitKey(1)

            depth, _, _ = cv2.split(depth_color_image)  # b,g,r are equal
            # print(np.count_nonzero(b),np.count_nonzero(g),np.count_nonzero(r))

            rgbd = np.zeros((270, 480, 4), dtype=np.uint8)
            rgbd[:, :, 0] = color_image[:, :, 0]
            rgbd[:, :, 1] = color_image[:, :, 1]
            rgbd[:, :, 2] = color_image[:, :, 2]
            rgbd[:, :, 3] = depth
            rgbd_stream.append(rgbd)
            # print(rgbd.shape)

            # images = color_image + depth_color_image
            # cv2.imshow("Stream", images)
            # cv2.waitKey(1)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     cv2.destroyAllWindows()
            #     break
        np.save(name, rgbd_stream)
    finally:
        pass


result = []


def count_bag(path, name="rgbd_stream"):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, path)
    # config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    colorizer = rs.colorizer(2)

    frameNum = 0
    fistFrame = pipeline.wait_for_frames().get_frame_number()
    temp = []
    while 1:
        frames = pipeline.wait_for_frames()
        curFrame = frames.get_frame_number()
        if curFrame == fistFrame:
            temp.append(frameNum)
            break
        frameNum += 1
    result.append(temp)

def batch_trans(folders_root_path):
    folders = os.listdir(folders_root_path)  # 数据集根目录
    folders_root_path = folders_root_path + '/'
    for folder in folders:
        files = os.listdir(folders_root_path + folder)  # 单个手势数据集根目录
        files_root_path = folders_root_path + folder + '/'
        for index, file in enumerate(files):
            print(files_root_path, index)
            read_bag(files_root_path + file, files_root_path + str(index))
            # count_bag(files_root_path + file, files_root_path + str(index))


if __name__ == '__main__':
    # path = "./test.bag"
    # read_bag(path)

    path = "G:/new_gesture"
    batch_trans(path)
    # count_bag("G:/new_gesture/pull/20210607_105117.bag")
    data = pd.DataFrame(result)
    data.to_excel("frames.xlsx")
