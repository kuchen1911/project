import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch
from torcheval.metrics import MulticlassAUPRC
from torchmetrics import Accuracy
from torch import nn
import lightning as L

from matplotlib import pyplot as plt

import cv2
import os

from torchvision.transforms import v2
import torchvision
from torchvision.models import resnet50
from torchvision.models import efficientnet_v2_s

import PIL

from tqdm import tqdm

#device = torch.device('mps:0')
model = efficientnet_v2_s(weights='DEFAULT')#.to(device)
num_classes = 10

class CarModel(torch.nn.Module):
    def __init__(self, no_input, no_output):
        super(CarModel, self).__init__()
        
        self.no_output = no_output
        self.no_input = no_input

        self.linear1 = torch.nn.Linear(self.no_input, 256)
        self.activation = torch.nn.Tanh()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(256, self.no_output)
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        
        return nn.functional.softmax(x)

car_model = CarModel(no_input=model.classifier[1].in_features, no_output=num_classes)#.to(device)

model.classifier = car_model

model = torch.load('weight0804')
model.eval()

car_type_mappa = {
 'Coupe': 0,
 'Sedan': 1,
 'Cab': 2,
 'Convertible': 3,
 'SUV': 4,
 'Minivan': 5,
 'Hatchback': 6,
 'Other': 7,
 'Van': 8,
 'Wagon': 9
}

inverse_mappa = dict(zip(car_type_mappa.values(), car_type_mappa.keys()))

def predict(image_id):
    img_name = os.path.join('pics', image_id)
        
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformation = v2.Compose([
        v2.ToTensor(),
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = transformation(image)
    image = image.to(device)
    
    pred = model(image.reshape(1, 3, 224, 224))
    pred_type = np.argmax(model(image.reshape(1, 3, 224, 224)).cpu().detach().numpy())
    
    return inverse_mappa[pred_type]
    
    
    
    
