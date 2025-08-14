import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
        输入数据：(1,28,28)的图片数据

        卷积1：-》(64,14,14)
            leaky relu
        卷积2：-》(128,7,7)
            leaky relu
            batchnorm2d
        卷积3：-》(256,3,3)
            leaky relu
            batchnorm2d
        卷积4：-》(1,1,1)
            Sigmoid


        输出数据：(1,1,1)的判别结果
    """
    def __init__(self,img_channel=1,out_channel=1):
        super().__init__()
        self.img_channel = img_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.img_channel,64,4,stride=2,padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,1,kernel_size=3,stride=1,padding=0),
        )
        # self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        out = self.conv4(x)
        # out = self.sigmoid(out)
        # print(out.shape)
        return self.dropout(out)

class Generator(nn.Module):
    """
        输入数据：(100,1,1)

        卷积1：-》(256,4,4)
            bn2
            relu
        卷积2：-》(128,7,7)
            bn2
            relu
        卷积3：-》(64,16,16)
            bn2
            relu
        卷积4：-》(1,28,28)
            Sigmoid()

        输出数据：(1,28,28)
    """
    def __init__(self,in_channel=100,out_channel=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel,256,kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
        )
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        # print(out.shape)

        return out

if __name__ == '__main__':
    gen = Generator()
    dis = Discriminator()
    dis_in = torch.randn((64,1,28,28))
    dis_out = dis(dis_in)
    gen_in = torch.randn((64,100,1,1))
    gen_out = gen(gen_in)
    print(torch.max(dis_out),torch.min(dis_out))
