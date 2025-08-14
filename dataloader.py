import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# 对数据做归一化（-1，1）
transform = transforms.Compose([
    transforms.ToTensor(),          # 0-1规划，channel,high,wisth,
    transforms.Normalize((0.5,),(0.5,))
])

train_ds = torchvision.datasets.MNIST("data",
                                     train=True,
                                     transform=transform,
                                     download=False)

dataloader = torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)

# def gen_img_plot(model,test_input):
#     prediction = np.squeeze(model(test_input).detach().cpu().numpy())
#     fig = plt.figure(figsize=(4,4))
#     for i in range(prediction.shape[0]):
#         plt.subplot(4,4,i+1)
#         plt.imshow((prediction[i]+1)/2)    # 要么是0~1的float，要么是0~255的uint
#         plt.axis('off')
#     plt.show()

if __name__ == '__main__':
    imgs,_ = next(iter(dataloader))
    print(f"img_shape:{imgs.shape}")
