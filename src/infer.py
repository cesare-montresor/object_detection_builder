import matplotlib.pyplot as plt
import numpy as np
import time 
import os
import yaml
import cv2

import torch
import torchvision.transforms.v2 as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import evaluate  # train,
from kitty_dataset import KittiDataset

# http://vision.cs.stonybrook.edu/~lasot/

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
model_path = 'output/1711007600148/model.pth'
imgnet_mean = (0.485, 0.456, 0.406)
imgnet_std = (0.229, 0.224, 0.225)

def main():
    params = {}

    
    model = torch.load(model_path)
    
    dataloader = getDataLoader(params, False)
    inference_model(model, dataloader)

@torch.no_grad
def inference_model(model, data_loader):
    model.to(dtype=dtype, device=device)
    model.eval()
    

    for x_images, y_labels in data_loader:
        x_images_gpu = x_images.to(dtype=dtype, device=device)
        
        y_hat = model(x_images_gpu)
        y_hat_cpu = y_hat.detach().cpu().numpy()
        for img, lbl, lbl_hat in zip(x_images, y_labels, y_hat_cpu):
            img_np = img.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = ((img_np * imgnet_std) + imgnet_mean)
            img_np = np.transpose(img_np, (2, 0, 1))
            img_np = (img_np * 255).astype(np.int32)

            lbl = np.array(lbl)
            c,h,w = img_np.shape
            scale = np.array([w,h,w,h])
            
            if lbl[0] > 0.5:
                x1,y1,x2,y2 = lbl[1:] * scale
                p1 = (int(x1),int(y1))
                p2 = (int(x2),int(y2))
                print('GT',p1,p2)
                img_np = cv2.rectangle(img_np, p1, p2, (0,255,0), thickness=10)
            
            if lbl_hat[0] > 0.5:
                x1,y1,x2,y2 = lbl_hat[1:] * scale
                p1 = (int(x1),int(y1))
                p2 = (int(x2),int(y2))    
                print('PRED',p1,p2)
                img_np = cv2.rectangle(img_np, p1, p2, (0,255,255), thickness=10)
            
            p1,p2 = (200,200),(250,250)
            img_np = cv2.rectangle(img_np, p1, p2, (0,255,255), thickness=10)

            plt.imshow(np.transpose(img_np, (1, 2, 0)))
            plt.show(block=True)





def getTransforms(params):
    img_size = (370, 1224)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(imgnet_mean, imgnet_std)    
    ])
    return transform


def getDataLoader(params, train=True, download=False):
    batch_size = params.get('batch_size', 32)
    transform = getTransforms(params)
    dataset = KittiDataset(root='./datasets', train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                            collate_fn=collate_batch, )  # , pin_memory_device=device)
    return dataloader


def collate_batch(batch):
    x_images = [data[0] for data in batch]
    y_labels = [data[1][0] for data in batch]

    image_sizes = [torch.as_tensor((img.shape[2], img.shape[1], img.shape[2], img.shape[1])) for img in
                   x_images]  # w,h,w,h
    y_bbox = [torch.as_tensor(label.get('bbox', [0, 0, 0, 0])) for label in y_labels]
    y_class = [torch.as_tensor([1 if label['type'] == 'Car' else 0]) for label in y_labels]

    x_images = torch.stack(x_images)
    y_class = torch.stack(y_class)
    y_bbox = torch.stack(y_bbox) / torch.stack(image_sizes)

    y_labels = torch.cat((y_class, y_bbox), dim=1)

    return x_images, y_labels


if __name__ == '__main__': main()
