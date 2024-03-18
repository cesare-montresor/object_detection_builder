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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
session_id = '1710758143091'
output_dir = './output/'

def main():
    params = {}

    session_dir = output_dir + session_id + '/'
    
    model = torch.load(session_dir+'model.pth')
    
    dataloader = getDataLoader(params, False)
    inference_model(model, dataloader)
    
    return 

def inference_model(model, data_loader):
    model.to(dtype=dtype, device=device)
    model.eval()

    for x_images, y_labels in data_loader:
        x_images_gpu = x_images.to(dtype=dtype, device=device)
        
        y_hat = model(x_images_gpu)
        for img, lbl, lbl_hat in zip(x_images, y_labels, y_hat):
            img_np = img.cpu().numpy()
            lbl = np.array(lbl)
            h,w,c = img_np.shape
            scale = np.array([h,w,h,w])
            
            if lbl[0] > 0.5:
                coords = lbl[1:] * scale
                y1,x1,hb,wb = coords
                y1,x1 = int(y1),int(x1)
                y2,x2 = int(hb+y1),int(wb+x1)
                cv2.rectangle(img_np, (y1,x1), (y2,x2), (0,255,0))
            
            if lbl_hat[0] > 0.5:
                coords = lbl_hat[1:] * scale
                y1,x1,hb,wb = coords
                y1,x1 = int(y1),int(x1)
                y2,x2 = int(hb+y1),int(wb+x1)
                cv2.rectangle(img_np, (y1,x1), (y2,x2), (0,255,255))

            imshow(img_np)

def train_model(params, model, data_loader):
    model.to(dtype=dtype, device=device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()

    lr = params.get('lr', 0.001)
    lr_max = params.get('lr_max', lr*10)
    num_epochs = params.get("num_epochs", 1)
    

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr
        # momentum=params.get('momentum',0.9)
    )
    num_batches = len(data_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr_max, total_steps=num_epochs*num_batches)

    for _ in range(num_epochs):
        for x_images, y_labels in data_loader:
            x_images = x_images.to(dtype=dtype, device=device)
            y_labels = y_labels.to(dtype=dtype, device=device)

            optimizer.zero_grad()
            y_hat = model(x_images)
            loss_class = criterion_class(y_hat[:, 0], y_labels[:, 0])
            loss_bbox = criterion_bbox(y_hat[:, 1:], y_labels[:, 1:])
            loss = loss_class + loss_bbox
            loss.backward()

            optimizer.step()
            scheduler.step()
    return model



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def getTransforms(params):
    img_size = (370, 1224)
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
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
                   x_images]  # h,w,h,w
    y_bbox = [torch.as_tensor(label.get('bbox', [0, 0, 0, 0])) for label in y_labels]
    y_class = [torch.as_tensor([1 if label['type'] == 'Car' else 0]) for label in y_labels]

    x_images = torch.stack(x_images)
    y_class = torch.stack(y_class)
    y_bbox = torch.stack(y_bbox) / torch.stack(image_sizes)

    y_labels = torch.cat((y_class, y_bbox), dim=1)

    return x_images, y_labels


if __name__ == '__main__': main()
