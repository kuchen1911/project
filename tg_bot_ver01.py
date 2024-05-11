import telebot
import logging
from my_token import token
from model import CarModel, predict, inverse_mappa, binary_predict
import torch
from torchvision.models import efficientnet_v2_s
import cv2
import os
from copy import deepcopy


logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)


bot = telebot.TeleBot(token)

# init car model
#device = torch.device('mps:0')
model = efficientnet_v2_s(weights='DEFAULT')#.to(device)
num_classes = 10

car_model = CarModel(no_input=model.classifier[1].in_features, no_output=num_classes)#.to(device)

model.classifier = car_model

model = torch.load('weight0804')
model = model#.to(device)
model.eval()

# binary model
binary_model = efficientnet_v2_s(weights='DEFAULT')#.to(device)
binary_model = binary_model.eval()


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
        
    img_name = os.path.join('pics', file_id+'.jpg')
        
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    binary = binary_predict(deepcopy(image), binary_model)
    
    car_type = False
    
    if binary[0] in (436, 817, 511, 717):
        car_type = True
    
    pred_type = predict(deepcopy(image), model, inverse_mappa)
    logger.info(f"PREDICTED TYPE IS {pred_type}, {binary}")
    
    msg = f"ЭТО {pred_type}" if car_type else f"Я думаю, что это скорее всего не машина, а {binary[1]} (PRED TYPE {binary[0]})"
    
    bot.send_message(
            message.from_user.id,
            msg,
        )

@bot.message_handler(commands=["photo"])
def handle_docs_photo(message):
    if message.text == "Загрузи фото".lower() or "Загрузи фото".upper():
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = "../nikita/" + file_info.file_path
        with open(src, "wb") as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Пожалуй, я сохраню это")


bot.infinity_polling()
