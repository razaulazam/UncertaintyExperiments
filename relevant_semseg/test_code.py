import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from deepLabv3 import DeepLab
from transform_deepLabv3 import RandomCropsTrain


image_transform = RandomCropsTrain(crop_size=(10, 10), patch_size=5)

image = np.random.random((20, 20, 3))*255.
label = np.random.randint(0, 33, size=(20, 20))

image_one = np.random.random((20, 20, 3))*255.
label_one = np.random.randint(0, 33, size=(20, 20))

image_t, label_t = image_transform(image, label)




fig, (ax1, ax2) = plt.subplots(1, 2)
ax2.imshow(image/255)
ax1.imshow(np.transpose(image_t, (1, 2, 0)))
plt.show()

print(label_t)
print(label_one_t)

    

    

