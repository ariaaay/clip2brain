import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from PIL import Image
from tqdm import tqdm

import os
from util.util import *

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

preprocess = transforms.Compose([transforms.Resize(255), transforms.ToTensor()])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size):
        return input.view(input.size(0), size, 1, 1)


class Autoencoder(nn.Module):
    def __init__(self, im_dim=255, h_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1
        )
        im_dim = int((im_dim - 1) / 2 + 1)
        # print(im_dim)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        im_dim = int((im_dim - 3) / 2 + 1)
        # print(im_dim)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        im_dim = int((im_dim - 3) / 2 + 1)
        # print(im_dim)

        self.f_dim = im_dim
        self.fc = nn.Linear(self.f_dim * self.f_dim * 32, h_dim)

        self.cf = nn.Linear(h_dim, self.f_dim * self.f_dim * 32)
        self.vnoc3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2
        )
        self.vnoc2 = nn.ConvTranspose2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=2
        )
        self.vnoc1 = nn.ConvTranspose2d(
            in_channels=8, out_channels=3, kernel_size=3, stride=2
        )

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        h = x

        # print(x.shape)
        x = self.cf(x)
        # print(x.shape)
        x = x.view(x.shape[0], 32, self.f_dim, self.f_dim)
        # print(x.shape)

        x = F.relu(self.vnoc3(x))
        # print(x.shape)
        x = F.relu(self.vnoc2(x))
        # print(x.shape)
        x = F.relu(self.vnoc1(x))
        # print(x.shape)

        return x, h


if __name__ == "__main__":
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # Load Images
    sf_dir = "../genStimuli/surfaceNormal/"
    all_images_paths = [
        sf_dir + name
        for name in os.listdir(sf_dir)
        if ".jpg" in name or ".JPEG" in name
    ]
    print("Number of surface normal images: {}".format(len(all_images_paths)))
    for epoch in range(num_epochs):
        for p in tqdm(all_images_paths):
            img = Image.open(p)
            input = Variable(preprocess(img).unsqueeze_(0)).to(device)
            # ===================forward=====================
            output = model(input)[0]
            loss = criterion(output, input)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.data))
        # if epoch % 10 == 0:
        #     # pic = to_img(output.cpu().data)
        #     pic = output.cpu().data
        #     save_image(pic, './dc_img/image_{}.png'.format(epoch))

        torch.save(model.state_dict(), "../outputs/models/conv_autoencoder.pth")
        if loss < 0.01:
            break
