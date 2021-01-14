# training validation test

import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'D:/iqiyi/PYTORCH/flower_photos/flower_photos'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
mkfile('D:/iqiyi/PYTORCH/flower_photos/train')
for cla in flower_class:
    mkfile('D:/iqiyi/PYTORCH/flower_photos/train/'+cla)

mkfile('D:/iqiyi/PYTORCH/flower_photos/val')
for cla in flower_class:
    mkfile('D:/iqiyi/PYTORCH/flower_photos/val/'+cla)

split_rate = 0.1
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'D:/iqiyi/PYTORCH/flower_photos/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'D:/iqiyi/PYTORCH/flower_photos/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")












