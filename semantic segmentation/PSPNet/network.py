import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2


class PSPNet(nn.Module):
    def __init__(self, batch_size=10, pretrained=True):
        super(PSPNet,self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.resnet_head = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool)
        self.resnet_layer1, self.resnet_layer2, self.resnet_layer3, self.resnet_layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for name, module in self.resnet_layer3.named_modules():
            if 'conv2' in name:
                module.dilation, module.padding, module.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in name:
                module.stride = 1

        for name, module in self.resnet_layer4.named_modules():
            if 'conv2' in name:
                module.dilation, module.padding, module.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in name:
                module.stride = 1

        self.batch_size = batch_size
        self.h, self.w = 224,224


        self.n_class = 22

        self.p_conv_1 = nn.Conv2d(2048,1, kernel_size=(1,1), stride=(1,1))
        self.p_conv_2 = nn.Conv2d(2048,1, kernel_size=(1,1), stride=(1,1))
        self.p_conv_3 = nn.Conv2d(2048,1, kernel_size=(1,1), stride=(1,1))
        self.p_conv_4 = nn.Conv2d(2048,1, kernel_size=(1,1), stride=(1,1))

        self.conv1 = nn.Conv2d(512, 4096, kernel_size=(1,1))
        self.conv2 = nn.Conv2d(4096, 4096, kernel_size=(1,1))
        self.conv3 = nn.Conv2d(4096, self.n_class, kernel_size=(1,1))

        self.aux_conv1 = nn.Conv2d(1024,256,kernel_size=(3,3),padding=(1,1),bias=False)
        self.aux_bn1 = nn.BatchNorm2d(256)
        self.aux_relu = nn.ReLU(inplace=True)
        self.aux_dropout = nn.Dropout2d(0.1)

        self.upsampling0 = nn.ConvTranspose2d(1024, self.n_class, kernel_size=(8,8),stride=(8,8),bias=False)
        self.upsampling1 = nn.ConvTranspose2d(2052, self.n_class, kernel_size=(8,8),stride=(8,8),bias=False)

    def forward(self,x, train_mode):
        x = self.resnet_head(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x_aux = self.resnet_layer3(x)
        print(x_aux.shape)
        x = self.resnet_layer4(x_aux)

        res_featuremap_shape = x.shape

        p_1 = F.avg_pool2d(x,kernel_size=x.shape[3])
        p_2 = F.avg_pool2d(x,kernel_size=x.shape[3]//2)
        p_3 = F.avg_pool2d(x,kernel_size=x.shape[3]//3)
        p_4 = F.avg_pool2d(x,kernel_size=x.shape[3]//6)

        p_1 = self.p_conv_1(p_1)
        p_2 = self.p_conv_2(p_2)
        p_3 = self.p_conv_3(p_3)
        p_4 = self.p_conv_4(p_4)

        p_1 = F.interpolate(p_1,(res_featuremap_shape[2],res_featuremap_shape[2]),mode='bilinear',align_corners=True)
        p_2 = F.interpolate(p_2,(res_featuremap_shape[2],res_featuremap_shape[2]),mode='bilinear',align_corners=True)
        p_3 = F.interpolate(p_3,(res_featuremap_shape[2],res_featuremap_shape[2]),mode='bilinear',align_corners=True)
        p_4 = F.interpolate(p_4,(res_featuremap_shape[2],res_featuremap_shape[2]),mode='bilinear',align_corners=True)

        x = torch.concat((x,p_1,p_2,p_3,p_4),1)
        x = self.upsampling1(x)
        if train_mode:
            x_aux = self.upsampling0(x_aux)
            return x_aux,x
        else:
            return x



def train(model,train_loader,optimizer,epoch,DEVICE,color_dict):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        aux,main_loss = model(data,train_mode=True)
        loss = 0.4 * F.cross_entropy(aux,target,ignore_index=21) + F.cross_entropy(main_loss,target,ignore_index=21)
        loss.backward()
        optimizer.step()
        if(batch_idx % 5 == 0):
            print(f"Train [{epoch}] : [{batch_idx}/{len(train_loader)}] ({100 * batch_idx / len(train_loader):.0f}%) \tLoss:{loss.item():.6f}")


def test(model,test_loader,epoch,DEVICE,color_dict,mode,data_count,boundary=21):
    model.eval()
    count = 0
    test_iou_score = 0
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data,train_mode=False)
            pred = output.argmax(dim=1)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()

            target_img = target[0].copy()
            
            # boundary의 target값과 pred값을 0으로 만들어 IoU에 포함시키지 않음
            target[target == boundary] = 0
            pred[target == boundary] = 0   

            intersection = np.logical_and(pred, target)
            union = np.logical_or(pred, target)
            test_iou_score += np.sum(intersection) / np.sum(union) * len(pred)

            img_bgr = np.zeros((224, 224, 3))
            cls_bgr = np.zeros((224, 224, 3))

            pred = pred[0]
            if len(np.unique(pred)) != 1:  # model 이 0만 predict 하면 이미지를 출력하지 않음
                for w in range(224):
                    for h in range(224):
                        img_bgr[w, h] = color_dict[pred[w, h]]
                        cls_bgr[w, h] = color_dict[target_img[w, h]]
                if(mode=='train'):
                    cv2.imwrite(f'./img/train/{count}_p.png', img_bgr)
                    cv2.imwrite(f'./img/train/{count}_c.png', cls_bgr)
                if(mode=='test'):
                    cv2.imwrite(f'./img/test/{count}_p.png', img_bgr)
                    cv2.imwrite(f'./img/test/{count}_c.png', cls_bgr)
            if(batch_idx % 5 == 0):
                print(f"Test_{mode}_data [{epoch}] : [{batch_idx}/{len(test_loader)}] ({100 * batch_idx / len(test_loader):.0f}%)")
            count += 1
        test_iou_score = test_iou_score / data_count * 100

        return test_iou_score