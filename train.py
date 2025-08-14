import torch
import torchvision
from model import Generator, Discriminator
from dataloader import dataloader
import numpy as np
import matplotlib.pyplot as plt
import tensorboardX
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_epoch = 10

gen = Generator().to(device)
dis = Discriminator().to(device)

d_optim = torch.optim.Adam(dis.parameters(),lr=1e-4)
g_optim = torch.optim.Adam(gen.parameters(),lr=1e-4)

loss_fn = torch.nn.BCEWithLogitsLoss()

log_dir = "log/1/"
os.makedirs(log_dir,exist_ok=True)
writer = tensorboardX.SummaryWriter(log_dir)

save_path = "generated images/1/"
os.makedirs(save_path,exist_ok=True)

def gen_img_plot(model,test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(prediction.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)    # 要么是0~1的float，要么是0~255的uint
        plt.axis('off')
    plt.show()


# 添加标签平滑和梯度惩罚
def gradient_penalty(discriminator, real, fake, device):
    batch_size = real.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)

    # 计算判别器输出
    d_out = discriminator(interpolated)

    # 计算梯度
    grad = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True
    )[0]

    grad = grad.view(grad.size(0), -1)
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

test_input = torch.randn(16, 100, 1, 1,device=device)

D_loss = []
G_loss = []

global_step = 0
# 编写训练循环
for epoch in range(max_epoch):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)

    for step, (img, _) in enumerate(dataloader):
        global_step += 1
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, 1, 1, device=device)

        d_optim.zero_grad()
        real_output = dis(img)  # 判别器输入真实图片，real_output对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output))
        d_real_loss.backward()

        gen_img = gen(random_noise)

        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output))
        d_fake_loss.backward()

        # # 梯度惩罚 (WGAN-GP)
        # gp = gradient_penalty(dis, img, gen_img, device)

        d_loss = d_real_loss + d_fake_loss
        # d_loss = d_real_loss + d_fake_loss + 10*gp
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                         torch.ones_like(fake_output))
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

        if step % 100 == 0:
            print(f"epoch:{epoch+1}/{max_epoch}",end=' ')
            print(f"step:{step}/{len(dataloader)}",end=' ')
            print(f"生成器损失：{g_loss.item():.4f}, 判别器损失{d_loss.item():.4f}",end='\n')

            writer.add_scalar("生成器损失", g_loss, global_step=global_step)
            writer.add_scalar("判别器损失", d_loss, global_step=global_step)

            gen_img_fixed_seed = gen(test_input)

            torchvision.utils.save_image(gen_img_fixed_seed,os.path.join(save_path,"epoch{:03d}_batch{:03d}.png".format(epoch,step)))

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        writer.add_scalar("生成器损失-epoch", g_epoch_loss, global_step=epoch)
        writer.add_scalar("判别器损失-epoch", d_epoch_loss, global_step=epoch)
        print(f'Epoch{epoch+1} 训练完毕')
        # gen_img_plot(gen, test_input)
