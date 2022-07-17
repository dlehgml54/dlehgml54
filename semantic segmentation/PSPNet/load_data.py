import torch
from torch.utils.data import DataLoader, Dataset
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

class TensorData(Dataset):
    # 외부에 있는 데이터를 가져오기 위해 외부에서 데이터가 들어올 수 있도록, x_data, y_data 변수를 지정
    def __init__(self, x_data, y_data):

        #들어온 x는 tensor형태로 변환
        self.x_data = torch.FloatTensor(x_data)
        # tensor data의 형태는 (배치사이즈, 채널사이즈, 이미지 너비, 높이)의 형태임
        # 따라서 들어온 데이터의 형식을 permute함수를 활용하여 바꾸어주어야함.
        self.x_data = self.x_data.permute(0,3,1,2)  # 인덱스 번호로 바꾸어주는 것 # 이미지 개수, 채널 수, 이미지 너비, 높이
        self.y_data = torch.LongTensor(y_data) # float tensor / long tensor 로 숫자 속성을 정해줄 수 있음
        self.len = self.y_data.shape[0]

    # x,y를 튜플형태로 바깥으로 내보내기
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len