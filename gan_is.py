import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torchvision.utils as vutils
from swd import swd
os.environ["OMP_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lr_d = 0.000013#Discriminatorの学習率
lr_g = 0.0002#Generatorの学習率
nz = 50#random noisze z 潜在変数の次元
main_folder = "./D000013G0002_swd_LeakyReLU_dropout01_nz50"
print(main_folder)
os.makedirs(main_folder, exist_ok=True)
os.makedirs(os.path.join(main_folder, "generated_images"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "real_images"), exist_ok=True)
save_path = os.path.join(main_folder, "loss.png")
save_path_sw = os.path.join(main_folder, "sw.png")

# random seed 設定
torch.manual_seed(1111)
np.random.seed(1111)
random.seed(1111)

batch_size = 64 #一度に学習するデータ量

#変換器の作成
transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),  # torch.Tensor へ変換
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 正規化する
 
#訓練データのダウンロードと変換設定
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
#訓練データのローダの作成
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
nc = 3

class Generator(nn.Module):
  def __init__(self, nz):
    super(Generator, self).__init__()
    self.nz = nz
    self.nf = 64
    self.main = nn.Sequential(
        nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(self.nf * 8),
        nn.LeakyReLU(0.2, inplace = True),
        #nn.ReLU(),
        nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf * 4),
        nn.LeakyReLU(0.2, inplace = True),
        #nn.ReLU(),
        nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf * 2),
        nn.LeakyReLU(0.2, inplace = True),
        #nn.ReLU(),
        nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf),
        nn.LeakyReLU(0.2, inplace = True),
        #nn.ReLU(),
        nn.ConvTranspose2d(self.nf, nc, 4, 2, 1, bias=False),
        nn.Tanh() 
        #nn.Sigmoid()

    )
  def forward(self, input):
    output = self.main(input)
    return output

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.nf = 64
    self.main = nn.Sequential(
        nn.Conv2d(nc, self.nf, 4, 2, 1, bias = False),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 2),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 4),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 8),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Dropout(0.1),
        nn.Conv2d(self.nf * 8, 1, 4, 1, 0, bias = False),
        nn.Sigmoid()
    )
  def forward(self, input):
    output = self.main(input)
    return output.view(-1, 1).squeeze(1)


fixed_noise = torch.randn(batch_size, nz, 1, 1, device = device)#正規分布

netG = Generator(nz).to(device)
netD = Discriminator().to(device)

optimizerD = optim.Adam(netD.parameters(), lr = lr_d, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr_g, betas=(0.5, 0.999))

criterion = nn.BCELoss()
"""
BCE(x, y) = -(y*log(x) + (1-y)*log(1-x))
x cnn output
y ground truth label
"""

#ネットワークを訓練する関数
def train_sw(netD, netG, criterion, optimizerD, optimizerG, n_epoch, batch):
    netD.train()  # ネットワークを訓練状態へ切り替える
    netG.train()  # ネットワークを訓練状態へ切り替える
    D_loss = []
    G_loss = []
    sw = []

    for epoch in range(n_epoch):
        errD_loss = 0
        errG_loss = 0
        sw_epoch = 0
        for i, data in enumerate(trainloader, 0):
            if data[0].to(device).size()[0] != batch:
              #一番最後はbatch_sizeに満たない場合は無視する
              break

            # Discriminatorの学習
            # 本物を見分ける
            optimizerD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size()[0]

            label = torch.ones(batch_size).to(device)
            #label = (0.7*torch.ones(batch_size)).to(device)+ (1/200)*torch.randint(0,100,(batch_size)).float().to(device)#soft label 0.7 ~ 1.2
            output = netD(real)
            errD_real = criterion(output, label)#label = 1だと、BCEは-log(x)になるのでx = 1(本物を本物にしたい)に近くなると嬉しい 定義の式と一致
            errD_real.backward()
            
            # 偽物を見分ける
            noise = torch.randn(batch_size, nz, 1, 1, device=device)#正規分布
            fake = netG(noise)
            
            label = torch.zeros(batch_size).to(device)
            #label = (1/1000)*torch.randint(0,300,(batch_size)).float().to(device)#soft label 0.0 ~ 0.3
            output = netD(fake.detach())#勾配がGに伝わらないようにdetach()して止める
            
            errD_fake = criterion(output, label)#label = 0だと、BCEは-log(1-x)になるのでx = 0(偽物を偽物にしたい)に近くなると嬉しい 定義の式と一致
            errD_fake.backward()
            errD = errD_real + errD_fake
            errD_loss = errD_loss + errD.item()
            optimizerD.step()#これでGのパラメータは更新されない

            # Generatorの学習
            optimizerG.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = netD(fake)

            errG = criterion(output, label)#label = 1だと、BCEは-log(x)になるのでx = 1(偽物を本物にしたい、騙したい)に近くなると嬉しい
            #定義の式とは少し異なる(実装上の問題)
            errG_loss = errG_loss + errG.item()
            errG.backward()
            optimizerG.step()

            sw_epoch += swd(real, fake, device=device).item()
            print("[{0:d}/{1:d}][{2:d}/{3:d}] Loss_D: {4:.4f} Loss_G: {5:.4f}".format(epoch+1, n_epoch, i, len(trainloader), errD.item(), errG.item()))
  
        fake = netG(fixed_noise)
    
        joined_real = torchvision.utils.make_grid(real, nrow=8, padding=3)
        joined_fake = torchvision.utils.make_grid(fake, nrow=8, padding=3)
        vutils.save_image(joined_fake.detach(), os.path.join(main_folder, "generated_images/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_real, os.path.join(main_folder, "real_images/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

        D_loss.append(errD_loss/len(trainloader))
        G_loss.append(errG_loss/len(trainloader))
        sw.append(sw_epoch/len(trainloader))

                      
    print('Finished Training')
    return D_loss, G_loss, sw

# 損失の変遷を表示する関数
def show_loss(D_loss, G_loss, save_path):
    fig = plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    x = [i for i in range(len(D_loss))]
    plt.plot(x, D_loss, label='D_loss')
    plt.plot(x, G_loss, label='G_loss')
    fig.legend()
    fig.savefig(save_path)
    #fig.show()

def show_sw(sw, save_path_sw):
    fig = plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("swd")
    x = [i for i in range(len(sw))]
    plt.plot(x, sw, label='swd')
    fig.legend()
    fig.savefig(save_path_sw)
    #fig.show()


D_loss, G_loss, sw = train_sw(netD, netG, nn.BCELoss(), optimizerD, optimizerG, n_epoch = 100, batch=batch_size)
show_loss(D_loss, G_loss, save_path) # 損失の変遷を表示する
show_sw(sw, save_path_sw)


