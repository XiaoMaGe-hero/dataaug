import pickle
import numpy as np
import os
import shutil
import cv2


# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
#
# data_root = "D:/iqiyi/PYTORCH/cifar-10-python"
# store_root = "D:/iqiyi/PYTORCH/cifar-10-python/train"
# for i in range(1, 6):
#     name = "data_batch_" + str(i)
#     dict = unpickle(data_root + "/" + name)
#     for k in range(10000):
#         img = np.reshape(dict[b"data"][k], (3, 32, 32))
#         img = img.transpose((1, 2, 0))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         if not os.path.exists(store_root+"/"+str(dict[b"labels"][k])):
#             os.mkdir(store_root+"/"+str(dict[b"labels"][k]))
#         # mkdir
#         cv2.imwrite(store_root+"/"+str(dict[b"labels"][k]) + "/"+str(dict[b"filenames"][k])+'.jpg', img)

# test_data = "D:/iqiyi/PYTORCH/cifar-10-python/test_batch"
# store_root = "D:/iqiyi/PYTORCH/cifar-10-python/test"
# test_dict = unpickle(test_data)
# for i in range(10000):
#     img = np.reshape(test_dict[b'data'][i], (3, 32, 32))
#     img = np.transpose(img, (1, 2, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if not os.path.exists(store_root + "/" + str(test_dict[b"labels"][i])):
#         os.mkdir(store_root + "/" + str(test_dict[b"labels"][i]))
#     cv2.imwrite(store_root + "/" + str(test_dict[b"labels"][i]) + "/" + str(test_dict[b"filenames"][i]) + '.jpg', img)

labeled = "D:/iqiyi/PYTORCH/cifar-10-python/train/"
unlabeled = "D:/iqiyi/PYTORCH/cifar-10-python/unlabel/"
val = "D:/iqiyi/PYTORCH/cifar-10-python/val/"
for i in range(10):
    list = os.listdir(unlabeled + str(i))
    for k in range(500, len(list)):
        shutil.move(unlabeled + str(i) + "/" + list[k], val + str(i) + "/" + list[k])

