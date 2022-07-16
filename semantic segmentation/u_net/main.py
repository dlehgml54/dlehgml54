import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import load_data
import network
import os

# ==== cuda check ==== #
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    if 'Y' != input('Warning > CUDA is not available want resume (Y/N)? > '):
        exit('User Stop')
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# ==================== #


# ==== setting ==== #
num_class = 20+2  # cls + ground : 0 + border : 21
Height = 224
Width = 224
batch_size = 16
test_batch_size = 10
learning_rate = 1e-3
saving_epoch = 3
epochs = 1000
restore_point = 0
# ================= #

# ==== load data ==== #
print('== data loading ==')
color_dict = {0: (0, 0, 0), 21: (192, 224, 224), 1: (0, 0, 128), 2: (128, 128, 192), 3: (128, 64, 0), 4: (0, 0, 192), 5: (128, 0, 64), 6: (0, 128, 128), 7: (128, 0, 128), 8: (128, 0, 0), 9: (0, 128, 192), 10: (0, 192, 128), 11: (128, 128, 64), 12: (128, 0, 192), 13: (0, 128, 64), 14: (128, 128, 128), 15: (0, 128, 0), 16: (0, 0, 64), 17: (0, 192, 0), 18: (128, 128, 0), 19: (0, 64, 0), 20: (0, 64, 128)}
X_train, Y_train, X_val, Y_val = load_data.load_data(Width, Height)
train_data = load_data.TensorData(X_train, Y_train)
test_data = load_data.TensorData(X_val, Y_val)
train_loader = DataLoader(train_data, batch_size=test_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
print('== data loaded ==')
# =================== #

# ==== record parameter ===== #
f = open('accuracy.txt', 'a+')
f.write(f"==== LR : {learning_rate}  /  Batch_size : {batch_size} ====\n")
f.close()
# =========================== #


model = network.PSPNet().to(DEVICE)
# optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# model restore #
PATH = './model'

if not os.path.exists(PATH):
    os.makedirs(PATH)

if restore_point != 0:
    print('load weight')
    model.load_state_dict(torch.load(PATH + '/' + str(restore_point) +'.pth'))
    print('weight loaded')
# ============== #

for epoch in range(restore_point+1,epochs+1):
    network.train(model,train_loader,optimizer,epoch,DEVICE,color_dict)
    if (epoch % saving_epoch) == 0:
        torch.save(model.state_dict(), PATH + '/' + str(epoch) + '.pth')
        train_iou = network.test(model,train_loader,epoch,DEVICE,color_dict,'train',len(X_train))
        test_iou = network.test(model,test_loader,epoch,DEVICE,color_dict,'test',len(X_val))
        print(f"Train IoU : {train_iou} / Test IoU : {test_iou}")
        f = open('accuracy.txt', 'a+')
        f.write(f"epoch : {epoch}  /  Train IoU : {train_iou}  /  Test IoU : {test_iou}\n")
        f.close()