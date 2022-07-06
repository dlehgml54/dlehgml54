import shutil
import cv2
import numpy as np
import tarfile
import os

# setting #
tar_file_location = "D:\DH_Model\object_detection\dataset\VOCtrainval_11-May-2012.tar"
unzip_location = "D:\DH_Model\object_detection\dataset/"
# ======= #

# unzip #
ap = tarfile.open(tar_file_location)
ap.extractall(unzip_location)
ap.close()

os.mkdir(unzip_location + "refine")

# get train set #
img_name_list = []
f = open(unzip_location + "VOCdevkit\VOC2012\ImageSets\Segmentation\\train.txt", 'r')
for i in f:
    img_name_list.append(i[:-1])
f.close()

origin_path = unzip_location + "VOCdevkit\VOC2012\\"

refine_location = unzip_location + "refine"

os.mkdir(refine_location + '\\train')
os.mkdir(refine_location + '\\train/JPEGImages')
os.mkdir(refine_location + '\\train/SegmentationClass')
os.mkdir(refine_location + '\\train/SegmentationObject')

path_jpg = refine_location + '/train/JPEGImages/'
path_cld = refine_location + '/train/SegmentationClass/'
path_obj = refine_location + '/train/SegmentationObject/'
for i in range(len(img_name_list)):
    if i % 20 == 0:
        print(f"{i} / {len(img_name_list)}")
    shutil.copy(origin_path + 'JPEGImages/' + img_name_list[i] + '.jpg', path_jpg + str(i) + '.jpg')
    shutil.copy(origin_path + 'SegmentationClass/' + img_name_list[i] + '.png', path_cld + str(i) + '.png')
    shutil.copy(origin_path + 'SegmentationObject/' + img_name_list[i] + '.png', path_obj + str(i) + '.png')

# get val set #
img_name_list = []
f = open(unzip_location + "VOCdevkit\VOC2012\ImageSets\Segmentation\\val.txt", 'r')
for i in f:
    img_name_list.append(i[:-1])
f.close()

origin_path = unzip_location + "VOCdevkit\VOC2012\\"

refine_location = unzip_location + "refine"

os.mkdir(refine_location + '\\val')
os.mkdir(refine_location + '\\val/JPEGImages')
os.mkdir(refine_location + '\\val/SegmentationClass')
os.mkdir(refine_location + '\\val/SegmentationObject')

path_jpg = refine_location + '/val/JPEGImages/'
path_cld = refine_location + '/val/SegmentationClass/'
path_obj = refine_location + '/val/SegmentationObject/'
for i in range(len(img_name_list)):
    if i % 20 == 0:
        print(f"{i} / {len(img_name_list)}")
    shutil.copy(origin_path + 'JPEGImages/' + img_name_list[i] + '.jpg', path_jpg + str(i) + '.jpg')
    shutil.copy(origin_path + 'SegmentationClass/' + img_name_list[i] + '.png', path_cld + str(i) + '.png')
    shutil.copy(origin_path + 'SegmentationObject/' + img_name_list[i] + '.png', path_obj + str(i) + '.png')

# make train target to 1channel #
os.mkdir(refine_location + '/train/refine_cls')
img_list = []
color_list = {(0, 0, 0): 0, (192, 224, 224): 21}  # background 0, Border 21
for i in range(1464):
    img_list.append(cv2.imread(refine_location + '/train/SegmentationClass/' + str(i) + '.png'))

color_val = 1
img_count = 0
for img in img_list:
    if img_count % 20 == 0:
        print(f"{img_count} / {len(img_list)}")
    cls = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            color_b = img[i, k, 0]
            color_g = img[i, k, 1]
            color_r = img[i, k, 2]
            color_total = color_b, color_g, color_r
            if color_total not in color_list:
                color_list[color_total] = color_val
                color_val += 1
            cls[i, k] = color_list[color_total]
    cv2.imwrite(refine_location + '/train/refine_cls/' + str(img_count) + '.png', cls)
    img_count += 1

# make val target to 1channel #
os.mkdir(refine_location + '/val/refine_cls')
img_list = []
for i in range(1449):
    img_list.append(cv2.imread(refine_location + '/val/SegmentationClass/' + str(i) + '.png'))

color_val = 1
img_count = 0
for img in img_list:
    if img_count % 20 == 0:
        print(f"{img_count} / {len(img_list)}")
    cls = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            color_b = img[i, k, 0]
            color_g = img[i, k, 1]
            color_r = img[i, k, 2]
            color_total = color_b, color_g, color_r
            cls[i, k] = color_list[color_total]
    cv2.imwrite(refine_location + '/val/refine_cls/' + str(img_count) + '.png', cls)
    img_count += 1