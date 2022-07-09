import numpy as np
import cv2


def load_data(width,height,train_size=1464, val_size=1449):
    X_train = np.zeros((train_size,width,height,3))
    X_val = np.zeros((val_size,width,height,3))
    Y_train = np.zeros((train_size,width,height))
    Y_val = np.zeros((val_size,width,height))

    for i in range(train_size):
        i = str(i)
        img = cv2.imread("D:\DH_Model\object_detection\dataset\\refine\\train\JPEGImages\\"+i+'.jpg')
        cls = cv2.imread("D:\DH_Model\object_detection\dataset\\refine\\train\\refine_cls\\"+i+'.png')[:,:,0]

        img = img.astype(np.float32)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        cls = cv2.resize(cls, (width, height), interpolation=cv2.INTER_NEAREST).reshape(width, height)
        X_train[int(i)] = img
        Y_train[int(i)] = cls
    mean = X_train.reshape(-1,3).mean(axis=0)
    std = X_train.reshape(-1,3).std(axis=0)
    X_train = (X_train - mean)/std

    for i in range(val_size):
        i = str(i)
        img = cv2.imread("D:\DH_Model\object_detection\dataset\\refine\\val\JPEGImages\\"+i+'.jpg')
        cls = cv2.imread("D:\DH_Model\object_detection\dataset\\refine\\val\\refine_cls\\"+i+'.png')[:,:,0]

        img = img.astype(np.float32)
        img = cv2.resize(img,(width, height), interpolation=cv2.INTER_NEAREST)
        cls = cv2.resize(cls, (width, height), interpolation=cv2.INTER_NEAREST).reshape(width, height)
        X_val[int(i)] = img
        Y_val[int(i)] = cls
    mean = X_val.reshape(-1,3).mean(axis=0)
    std = X_val.reshape(-1,3).std(axis=0)
    X_val = (X_val - mean)/std
    return X_train, Y_train, X_val, Y_val


def make_train_batch(img, cls, batch_size, width, height):
    rand_num = np.random.randint(low=0, high=len(img), size=batch_size)

    img_batch = np.zeros((batch_size,width,height,3))
    cls_batch = np.zeros((batch_size,width,height))
    count = 0
    for iL in rand_num:
        img_temp = img[iL]
        cls_temp = cls[iL]

        # -- 이미지 좌우 반전 -- #
        k = np.random.randint(0, 1)
        if k == 1:
            img_temp = cv2.flip(img_temp, 1)
            cls_temp = cv2.flip(cls_temp, 1)

        img_batch[count] = img_temp
        cls_batch[count] = cls_temp
        count += 1
    return img_batch, cls_batch


def make_test_batch(img, cls, test_batch_size, it, width, height):
    if (test_batch_size > len(img) - it):
        test_batch_size = len(img) - it

    iCount = 0
    Image_batch = np.zeros((test_batch_size,width,height,3))
    cls_batch = np.zeros((test_batch_size,width,height))

    for i in range(it, it + test_batch_size):
        img_temp = img[i]
        cls_temp = cls[i]

        Image_batch[iCount] = img_temp
        cls_batch[iCount] = cls_temp
        iCount += 1

    return Image_batch, cls_batch