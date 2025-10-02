import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import glob
import os
from torchvision import transforms

# 复制数据集类和模型类
class AffordanceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, is_train=False):  # 测试用test
        self.data_dir = os.path.join(data_dir, 'test')
        self.rgb_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_rgb.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_depth.npy')))
        self.afford_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_affordance.npy')))
        self.angle_paths = sorted(glob.glob(os.path.join(self.data_dir, '*_angles.npy')))
        self.transform = transforms.ToTensor()
        self.num_angle_classes = 36

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = self.transform(rgb)
        depth = np.load(self.depth_paths[idx])
        depth = torch.tensor(depth).unsqueeze(0).float()
        afford = np.load(self.afford_paths[idx])
        afford = torch.tensor(afford).unsqueeze(0).float()
        angle = np.load(self.angle_paths[idx])
        angle_class = torch.tensor((angle / 10).astype(int) % self.num_angle_classes).long()
        x = torch.cat([rgb, depth], dim=0)
        return x, afford.squeeze(0), angle_class.squeeze(0)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=37):
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

def evaluate_model():
    data_dir = './data/affordance_v5'
    test_dataset = AffordanceDataset(data_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet().cuda()
    model.load_state_dict(torch.load('./models/affordance_model.pth'))
    model.eval()

    criterion_afford = nn.BCEWithLogitsLoss()
    criterion_angle = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_afford_acc = 0.0
    total_angle_acc = 0.0
    num_samples = 0

    with torch.no_grad():
        for x, afford, angle in test_loader:
            x, afford, angle = x.cuda(), afford.cuda(), angle.cuda()
            pred = model(x)
            pred_afford = pred[:, 0]
            pred_angle = pred[:, 1:]

            loss_afford = criterion_afford(pred_afford, afford)
            loss_angle = criterion_angle(pred_angle, angle)
            loss = loss_afford + loss_angle
            total_loss += loss.item()

            # 计算准确率
            pred_afford_bin = (torch.sigmoid(pred_afford) > 0.5).float()
            afford_acc = (pred_afford_bin == afford).float().mean()
            total_afford_acc += afford_acc.item()

            pred_angle_class = torch.argmax(pred_angle, dim=1)
            angle_acc = (pred_angle_class == angle).float().mean()
            total_angle_acc += angle_acc.item()

            num_samples += 1

    avg_loss = total_loss / num_samples
    avg_afford_acc = total_afford_acc / num_samples
    avg_angle_acc = total_angle_acc / num_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Affordance Accuracy: {avg_afford_acc:.4f}")
    print(f"Angle Accuracy: {avg_angle_acc:.4f}")

if __name__ == '__main__':
    evaluate_model()
