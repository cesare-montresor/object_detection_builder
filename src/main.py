import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights


from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def test():
    
    
    classes = ('tree', 'sky', 'car', 'sign', 'road', 'pedestrian', 'fence', 'pole', 'sidewalk', 'bicyclist')
    
    download_ds = False

    

    transform = getTransforms(params)
    trainset = torchvision.datasets.Kitti(root='./datasets', train=True, download=download_ds, transform=transform)
    valset = torchvision.datasets.Kitti(root='./datasets', train=False, download=download_ds, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2, collate_fn=lambda x:x)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x:x)

    dataiter = iter(trainloader)
    batch = next(dataiter)

    images = [data[0] for data in batch]
    labels = [data[1][0]['bbox'] for data in batch]
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    pass


def main():
    params = [
        {"name": "lr",             "value_type": "float",   "type": "range",  "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "batch_size",     "value_type": "int",     "type": "range",  "bounds": [16, 128]},
        {"name": "num_epoch",      "value_type": "int",     "type": "range",  "bounds": [1, 30]},
        {"name": "drop_out",       "value_type": "float",   "type": "range",  "bounds": [0.0,0.9]},
        {"name": "resnet_size",    "value_type": "int",     "type": "choice", "values": [18,34,50,101,152], "is_ordered":True},
        {"name": "fc_hidden_size", "value_type": "int",     "type": "choice", "values": [18,34,50,101,152], "is_ordered":True},
        # {"name": "step_size ", "type": "range", "bounds": [0.0,0.9]},
        
    ]

    best_parameters, values, experiment, model = optimize(
        parameters=params,
        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

def train_evaluate(params):
    trainloader = getDataLoader(params)
    valloader = getDataLoader(params,False)

    model = build_model(params)
    model_trained = train_model(params, model, trainloader)

    return evaluate(net=model_trained,data_loader=valloader, dtype=dtype, device=device)


def train_model(params, model, data_loader):
    model.to(dtype=dtype, device=device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    
    lr = params.get('lr',0.001)
    num_epochs = params.get("num_epochs",3)
    step_size = params.get("step_size",30)

    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr
        # momentum=params.get('momentum',0.9)
    )


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size)
    

    for _ in range(num_epochs):
        for batch in data_loader:
            x_image = [data[0] for data in batch]
            image_sizes = [torch.as_tensor((img.shape[2],img.shape[1],img.shape[2],img.shape[1])) for img in x_image] #h,w,h,w
            y_bbox = [torch.as_tensor(data[1][0].get('bbox',[0,0,0,0])) for data in batch]
            y_class = [torch.as_tensor(1 if data[1][0]['type']=='Car' else 0) for data in batch]

            x_image = torch.stack(x_image).to(dtype=dtype, device=device)
            y_class = torch.stack(y_class).to(dtype=dtype, device=device)
            y_bbox = torch.stack(y_bbox).to(device=device) / torch.stack(image_sizes).to(device=device)

            optimizer.zero_grad()

            y_hat = model(x_image)
            y_hat_class = y_hat[:,0]
            y_hat_bbox = y_hat[:,1:]
            loss_class = criterion_class(y_hat_class, y_class)
            loss_bbox = criterion_bbox(y_hat_bbox, y_bbox)
            loss = loss_class + loss_bbox
            loss.backward()

            optimizer.step()
            scheduler.step()
    return model


def build_model(params):
    dropout = params.get('dropout',0.1)
    resnet_size = params.get('resnet_size',18)
    fc_hidden_size = params.get('fs_hidden_size',64) # Hidden layer size; you can optimize this as well

    if resnet_size == 18:  model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if resnet_size == 34:  model = resnet34(weights=ResNet34_Weights.DEFAULT)
    if resnet_size == 50:  model = resnet50(weights=ResNet50_Weights.DEFAULT)
    if resnet_size == 101: model = resnet101(weights=ResNet101_Weights.DEFAULT)
    if resnet_size == 152: model = resnet152(weights=ResNet152_Weights.DEFAULT)
    
    
    in_features = model.fc.in_features
    print(in_features)                                    

    model.fc = nn.Sequential(
        nn.Linear(in_features, fc_hidden_size), # attach trainable classifier
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_hidden_size, 5),
        nn.Sigmoid()
    )

    return model # return untrained model



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def getTransforms(params):
    img_size = (370, 1224)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def getDataLoader(params, train=True, download=False):
    batch_size = params.get('batch_size',32)
    transform = getTransforms(params)
    dataset = torchvision.datasets.Kitti(root='./datasets', train=train, download=download, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda x:x)
    return dataloader











if __name__ == '__main__': main()