import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
import os
from torchvision import transforms

# 复制模型类
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

def visualize_predictions(scene_id=0, data_dir='./data/affordance_v5/test'):
    """可视化模型预测的可供性热力图"""
    # 加载模型
    model = UNet().cuda()
    model.load_state_dict(torch.load('./models/affordance_model.pth', weights_only=True))
    model.eval()

    # 找到对应的文件
    rgb_pattern = os.path.join(data_dir, f'scene_{scene_id:04d}_rgb.png')
    depth_pattern = os.path.join(data_dir, f'scene_{scene_id:04d}_depth.npy')
    afford_pattern = os.path.join(data_dir, f'scene_{scene_id:04d}_affordance.npy')
    angle_pattern = os.path.join(data_dir, f'scene_{scene_id:04d}_angles.npy')

    if not os.path.exists(rgb_pattern):
        print(f"Scene {scene_id} not found in {data_dir}")
        return

    # 加载数据
    rgb = Image.open(rgb_pattern).convert('RGB')
    rgb_np = np.array(rgb)
    transform = transforms.ToTensor()
    rgb_tensor = transform(rgb)

    depth = np.load(depth_pattern)
    depth_tensor = torch.tensor(depth).unsqueeze(0).float()

    # 真实标签
    true_afford = np.load(afford_pattern)
    true_angle = np.load(angle_pattern)

    # 准备输入
    x = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).cuda()

    # 预测
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()
        pred_afford = torch.sigmoid(torch.tensor(pred[0])).numpy()
        pred_angle_class = np.argmax(pred[1:], axis=0)

    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # RGB Image
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB Image")
    axes[0, 0].axis('off')

    # GT Affordance
    im1 = axes[0, 1].imshow(true_afford, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title("GT Affordance")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Predicted Affordance
    im2 = axes[0, 2].imshow(pred_afford, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title("Pred Affordance")
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Affordance Error
    diff_afford = np.abs(pred_afford - true_afford)
    im3 = axes[0, 3].imshow(diff_afford, cmap='Reds', vmin=0, vmax=0.1)
    axes[0, 3].set_title("Affordance Error")
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # GT Angle
    im4 = axes[1, 0].imshow(true_angle, cmap='hsv', vmin=0, vmax=35)
    axes[1, 0].set_title("GT Angle")
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Pred Angle
    im5 = axes[1, 1].imshow(pred_angle_class, cmap='hsv', vmin=0, vmax=35)
    axes[1, 1].set_title("Pred Angle")
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Angle Error
    diff_angle = np.abs(pred_angle_class - true_angle) % 36
    diff_angle = np.minimum(diff_angle, 36 - diff_angle)  # minimal angle diff
    im6 = axes[1, 2].imshow(diff_angle, cmap='Reds', vmin=0, vmax=5)
    axes[1, 2].set_title("Angle Error")
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Overlay
    overlay = rgb_np.copy().astype(np.float32) / 255.0
    heatmap_colored = plt.cm.hot(pred_afford)[..., :3]
    mask = pred_afford > 0.1
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * heatmap_colored[mask]

    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("Overlay")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(f'./vis/scene_{scene_id:04d}_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"Scene {scene_id} 预测结果:")
    print(f"可供性 MAE: {np.mean(diff_afford):.4f}")
    print(f"角度 MAE: {np.mean(diff_angle):.2f} 类")
    print(f"可供性像素准确率 (>0.5): {(np.abs(pred_afford - true_afford) < 0.1).mean():.4f}")
    print(f"角度像素准确率: {(diff_angle == 0).mean():.4f}")

if __name__ == '__main__':
    # 可视化测试集中的前几个场景
    import glob
    test_rgb_files = glob.glob('./data/affordance_v5/test/scene_*_rgb.png')
    scene_ids = []
    for f in test_rgb_files[:5]:  # 前5个
        scene_id = int(f.split('_')[-2])
        scene_ids.append(scene_id)

    for scene_id in sorted(scene_ids):
        try:
            visualize_predictions(scene_id)
        except Exception as e:
            print(f"Error visualizing scene {scene_id}: {e}")
            continue