import torch
import cv2
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time
import random
from PIL import Image

#   device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from policy import solarize, color, cutout, contrast, autocontrast, equalize, invert, rotate, translatey, translatex, sharpness, shearx, sheary, posterize, brightness
#   数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(31),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((31, 31)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


#data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = os.getcwd()


# image_path = 'D:/iqiyi/PYTORCH/flower_photos/'
image_path_cifar = "D:/iqiyi/PYTORCH/cifar-10-python"

batch_size = 32

validate_dataset = datasets.ImageFolder(root=image_path_cifar + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()
#print(test_image[0].size(),type(test_image[0]))
#print(test_label[0],test_label[0].item(),type(test_label[0]))


#显示图像，之前需把validate_loader中batch_size改为4
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes=10, init_weights=True)

net.to(device)
#损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
#优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
#训练参数保存路径
save_path = './AlexNet.pth'
#训练过程中最高准确率
best_acc = 0.0


# design the operation dict:
operate_dict = {0: [['invert', 0.1, 7], ['contrast', 0.2, 6]], 1: [['rotate', 0.7, 2], ['translatex', 0.3, 9]],
                2: [['sharpness', 0.8, 1], ['sharpness', 0.9, 3]], 3: [['sheary', 0.5, 8], ['translatey', 0.7, 9]],
                4: [['autocontrast', 0.5, 8], ['equalize', 0.9, 2]], 5: [['sheary', 0.2, 7], ['posterize', 0.3, 7]],
                6: [['color', 0.4, 3], ['brightness', 0.6, 7]], 7: [['sharpness', 0.3, 9], ['brightness', 0.7, 9]],
                8: [['equalize', 0.6, 5], ['equalize', 0.5, 1]], 9: [['contrast', 0.6, 7], ['sharpness', 0.6, 5]],
                10: [['color', 0.7, 7], ['translatex', 0.5, 8]], 11: [['equalize', 0.3, 7], ['autocontrast', 0.4, 8]],
                12: [['translatey', 0.4, 3], ['sharpness', 0.2, 6]], 13: [['brightness', 0.9, 6], ['color', 0.2, 8]],
                14: [['solarize', 0.5, 2], ['invert', 0, 0.3]], 15: [['equalize', 0.2, 0], ['autocontrast', 0.6, 0]],
                16: [['equalize', 0.2, 8], ['equalize', 0.6, 4]], 17: [['color', 0.9, 9], ['equalize', 0.6, 6]],
                18: [['autocontrast', 0.8, 4], ['solarize', 0.2, 8]], 19: [['brightness', 0.1, 3], ['color', 0.7, 0]],
                20: [['solarize', 0.4, 5], ['autocontrast', 0.9, 3]], 21: [['translatey', 0.9, 9], ['translatey', 0.7, 9]],
                22: [['autocontrast', 0.9, 2], ['solarize', 0.8, 3]], 23: [['equalize', 0.8, 8], ['invert', 0.1, 3]],
                24: [['translatey', 0.7, 9], ['autocontrast', 0.9, 1]]}
#   开始进行训练和测试，训练一轮，测试一轮
for epoch in range(60):
    # train
    net.train()    #    训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    t1 = time.perf_counter()

    # preprocess all pics
    if not os.path.exists(image_path_cifar + "/train" + str(epoch)):
        os.mkdir(image_path_cifar + "/train" + str(epoch))
    for name in os.listdir(image_path_cifar + "/train"):
        for file in os.listdir(image_path_cifar + "/train/" + name):
            jpg = Image.open(image_path_cifar + "/train/" + name+'/'+file)
            operate_id = random.randint(0, 24)
            operation = operate_dict[operate_id]
            jpg = locals()[operation[0][0]](jpg, operation[0][1], operation[0][2])
            jpg = locals()[operation[1][0]](jpg, operation[1][1], operation[1][2])
            jpg.save(image_path_cifar + "/train" + str(epoch) + '/' +file)
    # done



    train_dataset = datasets.ImageFolder(root=image_path_cifar + "/train",
                                     transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()    #测试过程中不需要dropout，使用所有的神经元
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
print('Finished Training')





