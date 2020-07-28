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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_class = 10#ラベルの種類
nz = 100#潜在変数の次元
nc = 3#生成する画像のチャンネル
batch_size = 64 #一度に学習するデータ量
epoch_number = 200#epoch数

lr_d = 0.00002#Discriminatorの学習率
lr_g = 0.0002#Generatorの学習率
main_folder = "./D00002G0002_conditional_epoch200"
os.makedirs(main_folder, exist_ok=True)
os.makedirs(os.path.join(main_folder, "generated_images"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "real_images"), exist_ok=True)
save_path = os.path.join(main_folder, "loss.png")

# random seed 設定
torch.manual_seed(1111)
np.random.seed(1111)
random.seed(1111)


#変換器の作成
transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),  # torch.Tensor へ変換
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 正規化する
 
#訓練データのダウンロードと変換設定
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
#訓練データのローダの作成
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)


class Generator(nn.Module):
  def __init__(self, nz):
    super(Generator, self).__init__()
    self.nz = nz
    self.nf = 64
    self.main = nn.Sequential(
        nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(self.nf * 8),
        nn.LeakyReLU(0.2, inplace = True),
        nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf * 4),
        nn.LeakyReLU(0.2, inplace = True),
        nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf * 2),
        nn.LeakyReLU(0.2, inplace = True),
        nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(self.nf),
        nn.LeakyReLU(0.2, inplace = True),
        nn.ConvTranspose2d(self.nf, nc, 4, 2, 1, bias=False),
        nn.Tanh() 

    )
  def forward(self, input):
    output = self.main(input)
    return output

class Discriminator(nn.Module):
  def __init__(self, nc):
    super(Discriminator, self).__init__()
    self.nc = nc
    self.nf = 64
    self.main = nn.Sequential(
        nn.Conv2d(self.nc, self.nf, 4, 2, 1, bias = False),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 2),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 4),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias = False),
        nn.BatchNorm2d(self.nf * 8),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(self.nf * 8, 1, 4, 1, 0, bias = False),
        nn.Sigmoid()
    )
  def forward(self, input):
    output = self.main(input)
    return output.view(-1, 1).squeeze(1)


def onehot_encode(label, device, n_class=n_class):
    eye = torch.eye(n_class, device=device)

    return eye[label].view(-1, n_class, 1, 1)

def concat_image_label(image, label, device, n_class=n_class):
    B, C, H, W = image.size()

    oh_label = onehot_encode(label, device, n_class)
    oh_label = oh_label.expand(B, n_class, H, W)

    return torch.cat((image, oh_label), dim=1)

def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device, n_class)

    return torch.cat((noise, oh_label), dim=1)



fixed_noise = torch.randn(batch_size, nz, 1, 1, device = device)#正規分布
fixed_label = torch.randint(10, (batch_size, ), dtype=torch.long, device=device)#fake生成用のラベル
fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)#確認用のノイズとラベルを連結

netG = Generator(nz+10).to(device)
netD = Discriminator(nc+10).to(device)

optimizerD = optim.Adam(netD.parameters(), lr = lr_d, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr_g, betas=(0.5, 0.999))

criterion = nn.MSELoss()
"""
MSE(x, y) = (x-y)**2
x cnn output
y ground truth label
"""

#ネットワークを訓練する関数
def train(netD, netG, criterion, optimizerD, optimizerG, n_epoch, batch):
    netD.train()  # ネットワークを訓練状態へ切り替える
    netG.train()  # ネットワークを訓練状態へ切り替える
    D_loss = []
    G_loss = []
    for epoch in range(n_epoch):
        errD_loss = 0
        errG_loss = 0
        for i, data in enumerate(trainloader, 0):
            if data[0].to(device).size()[0] != batch:
              #一番最後はbatch_sizeに満たない場合は無視する
              break

            real_image = data[0].to(device)
            real_label = data[1].to(device)

            real_image_label = concat_image_label(real_image, real_label, device)#リアル画像とリアルラベルを生成 (n, c+10, h, w)

            sample_size = real_image.size()[0]#バッチサイズ
            noise = torch.randn(sample_size, nz, 1, 1, device=device)
            fake_label = torch.randint(10, (sample_size, ), dtype=torch.long, device=device)#fake生成用のラベル
            fake_noise_label = concat_noise_label(noise, fake_label, device)#ノイズとラベルを連結 (n, nz+10) ラベルを加えた潜在変数

            real_target = torch.ones(sample_size,device=device)#GAN Lossのターゲット
            fake_target = torch.zeros(sample_size,device=device)#GAN Lossのターゲット

            # Discriminatorの学習
            # 本物を見分ける
            optimizerD.zero_grad()

            output = netD(real_image_label)
            errD_real = criterion(output, real_target)#label = 1だと、BCEは-log(x)になるのでx = 1(本物を本物にしたい)に近くなると嬉しい
            errD_real.backward()
           
            # 偽物を見分ける
            fake = netG(fake_noise_label)#ノイズとラベルから画像を生成
            
            fake_image_label = concat_image_label(fake, fake_label, device)#生成した画像とラベルを連結
            output = netD(fake_image_label.detach())#勾配がGに伝わらないようにdetach()して止める
            errD_fake = criterion(output, fake_target)#label = 0だと、BCEは-log(1-x)になるのでx = 0(偽物を偽物にしたい)に近くなると嬉しい
            errD_fake.backward()

            errD = errD_real + errD_fake
            errD_loss = errD_loss + errD.item()
            optimizerD.step()#これでGのパラメータは更新されない

            # Generatorの学習
            optimizerG.zero_grad()
            
            output = netD(fake_image_label)

            errG = criterion(output, real_target)#label = 1だと、BCEは-log(x)になるのでx = 1(偽物を本物にしたい、騙したい)に近くなると嬉しい
            #実際の式とは少し異なる
            errG_loss = errG_loss + errG.item()
            errG.backward()
            optimizerG.step()
            print("[{0:d}/{1:d}][{2:d}/{3:d}] Loss_D: {4:.4f} Loss_G: {5:.4f}".format(epoch+1, n_epoch, i+1, len(trainloader), errD.item(), errG.item()))
  
        fake_image = netG(fixed_noise_label)
    
        joined_real = torchvision.utils.make_grid(real_image, nrow=8, padding=3)
        joined_fake = torchvision.utils.make_grid(fake_image, nrow=8, padding=3)
        
        vutils.save_image(joined_fake.detach(), os.path.join(main_folder, "generated_images/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_real, os.path.join(main_folder, "real_images/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

        D_loss.append(errD_loss/len(trainloader))
        G_loss.append(errG_loss/len(trainloader))
                   
    print('Finished Training')
    return D_loss, G_loss

# 損失の変遷を表示する関数
def show_loss(D_loss, G_loss, save_path):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    x = [i for i in range(len(D_loss))]
    plt.plot(x, D_loss, label='D_loss')
    plt.plot(x, G_loss, label='G_loss')
    plt.legend()
    plt.savefig(save_path)
    
if __name__  == "__main__":
    D_loss, G_loss = train(netD, netG, nn.BCELoss(), optimizerD, optimizerG, n_epoch = epoch_number, batch=batch_size)
    show_loss(D_loss, G_loss, save_path) # 損失の変遷を表示する 