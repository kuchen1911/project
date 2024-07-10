import telebot
import logging
from my_token import token



#from model import CarModel, predict, inverse_mappa, binary_predict
#import torch
#from torchvision.models import efficientnet_v2_s
#import cv2
#import os
#from copy import deepcopy
#
#
#logger = telebot.logger
#telebot.logger.setLevel(logging.DEBUG)
#
#
#bot = telebot.TeleBot(token)
#
## init car model
##device = torch.device('mps:0')
#model = efficientnet_v2_s(weights='DEFAULT')#.to(device)
#num_classes = 10
#
#car_model = CarModel(no_input=model.classifier[1].in_features, no_output=num_classes)#.to(device)
#
#model.classifier = car_model
#
#model = torch.load('weight0804')
#model = model#.to(device)
#model.eval()
#
## binary model
#binary_model = efficientnet_v2_s(weights='DEFAULT')#.to(device)
#binary_model = binary_model.eval()


logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)


bot = telebot.TeleBot(token)

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

model = torch.load('efficientnet_v2_s')
model.eval()
#model = model.to(device)

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
    image_id = image_id + '.jpg'
    img_name = os.path.join('/Users/nnprazdnikov/Desktop/Project/bots/pics', image_id)
    
    logger.info(f"PATH IS {img_name}")
        
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformation = v2.Compose([
        v2.ToTensor(),
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = transformation(image)
    image = image#.to(device)
    
    pred = model(image.reshape(1, 3, 224, 224))
    pred_type = np.argmax(model(image.reshape(1, 3, 224, 224)).cpu().detach().numpy())
    
    return inverse_mappa[pred_type]

@bot.message_handler(commands=["start"])
def get_text_messages(message):
        bot.send_message(
            message.from_user.id,
            "Привет, чем я могу тебе помочь? Можешь загрузить фото /photo",
        )
        
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_id = bot.get_file(message.photo[len(message.photo) - 1].file_id).file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    with open(f"pics/{file_id}.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    
    
    #binary = binary_predict(deepcopy(image), binary_model)
    #
    #car_type = False
    #
    #if binary[0] in (436, 817, 511, 717):
    #    car_type = True
    #
    #pred_type = predict(deepcopy(image), model, inverse_mappa)
    #logger.info(f"PREDICTED TYPE IS {pred_type}, {binary}")
    #
    #msg = f"ЭТО {pred_type}" if car_type else f"Я думаю, что это скорее всего не машина, а {binary[1]} (PRED TYPE {binary[0]})"
    #
    #bot.send_message(
    #        message.from_user.id,
    #        msg,
    #    )
    
    
    pred_type = predict(file_id)
    logger.info(f"PREDICTED TYPE IS {pred_type}")
    
    bot.send_message(
            message.from_user.id,
            f"ЭТО {pred_type}",
        )
    
    


#@bot.message_handler(commands=["photo"])
#def handle_docs_photo(message):
#    if message.text == "Загрузи фото".lower() or "Загрузи фото".upper():
#        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
#        downloaded_file = bot.download_file(file_info.file_path)
#
#        src = "../nikita/" + file_info.file_path
#        with open(src, "wb") as new_file:
#            new_file.write(downloaded_file)
#
#        bot.reply_to(message, "Пожалуй, я сохраню это")


bot.infinity_polling()

