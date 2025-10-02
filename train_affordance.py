import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import json


class AffordanceDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = os.path.join(data_dir, 'train' if is_train else 'test')
        self.rgb_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_rgb.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_depth.npy')))
        self.afford_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_affordance.npy')))
        self.angle_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_angles.npy')))
        self.transform = transforms.ToTensor()
        self.num_angle_classes = 36  # 每10度一类，0-35对应0°-350°

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = self.transform(rgb)  # (3,H,W)

        depth = np.load(self.depth_paths[idx])
        depth = torch.tensor(depth).unsqueeze(0).float()  # (1,H,W)

        afford = np.load(self.afford_paths[idx])
        afford = torch.tensor(afford).unsqueeze(0).float()  # (1,H,W)

        angle = np.load(self.angle_paths[idx])
        # 离散化角度：0-360度 -> 0-35类
        angle_class = torch.tensor((angle / 10).astype(int) % self.num_angle_classes).long()  # (H,W)

        x = torch.cat([rgb, depth], dim=0)  # (4,H,W)
        return x, afford.squeeze(0), angle_class.squeeze(0)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=37):  # 36角度类 + 1 affordance
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.final = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.upconv3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


def train_model():
    data_dir = './data/affordance_v5'
    train_dataset = AffordanceDataset(data_dir, is_train=True)
    val_dataset = AffordanceDataset(data_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet().cuda()
    criterion_afford = nn.BCEWithLogitsLoss()
    criterion_angle = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    for epoch in range(50):  # 增加epoch
        model.train()
        train_loss = 0.0
        for x, afford, angle in train_loader:
            x, afford, angle = x.cuda(), afford.cuda(), angle.cuda()
            pred = model(x)  # (B, 37, H, W)
            pred_afford = pred[:, 0]  # (B, H, W)
            pred_angle = pred[:, 1:]  # (B, 36, H, W)

            loss_afford = criterion_afford(pred_afford, afford)
            loss_angle = criterion_angle(pred_angle, angle)
            loss = loss_afford + loss_angle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, afford, angle in val_loader:
                x, afford, angle = x.cuda(), afford.cuda(), angle.cuda()
                pred = model(x)
                pred_afford = pred[:, 0]
                pred_angle = pred[:, 1:]

                loss_afford = criterion_afford(pred_afford, afford)
                loss_angle = criterion_angle(pred_angle, angle)
                loss = loss_afford + loss_angle
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/affordance_model.pth')
            print("Saved best model")

    print("Training complete. Best model saved to ./models/affordance_model.pth")


if __name__ == '__main__':
    train_model()
