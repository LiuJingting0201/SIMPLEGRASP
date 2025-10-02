# -*- coding: utf-8 -*-
"""
可供性数据可视化工具
用于验证和查看收集的可供性热力图数据
"""

import numpy as np
import cv2
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

def load_scene_data(data_dir, scene_id):
    """加载场景数据"""
    scene_name = f"scene_{scene_id:04d}"
    data_path = Path(data_dir)
    
    # 加载图像
    rgb_path = data_path / f"{scene_name}_rgb.png"
    rgb_image = cv2.imread(str(rgb_path))
    if rgb_image is not None:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # 加载深度图
    depth_path = data_path / f"{scene_name}_depth.npy"
    depth_image = np.load(str(depth_path)) if depth_path.exists() else None
    
    # 加载可供性数据
    affordance_path = data_path / f"{scene_name}_affordance.npy"
    affordance_map = np.load(str(affordance_path)) if affordance_path.exists() else None
    
    # 加载角度数据
    angles_path = data_path / f"{scene_name}_angles.npy"
    angle_map = np.load(str(angles_path)) if angles_path.exists() else None
    
    # 加载元数据
    meta_path = data_path / f"{scene_name}_meta.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    return rgb_image, depth_image, affordance_map, angle_map, metadata

def visualize_affordance_heatmap(rgb_image, affordance_map, angle_map, metadata, save_path=None):
    """可视化可供性热力图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始RGB图像
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("原始RGB图像")
    axes[0, 0].axis('off')
    
    # 2. 可供性热力图
    affordance_vis = axes[0, 1].imshow(affordance_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title("可供性热力图 (成功概率)")
    axes[0, 1].axis('off')
    plt.colorbar(affordance_vis, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. 角度地图
    angle_vis = axes[0, 2].imshow(angle_map, cmap='hsv', vmin=-1, vmax=metadata['num_angles']-1)
    axes[0, 2].set_title("最佳抓取角度")
    axes[0, 2].axis('off')
    plt.colorbar(angle_vis, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. 热力图叠加在RGB上
    overlay = rgb_image.copy().astype(np.float32) / 255.0
    
    # 创建热力图掩码
    heatmap_mask = affordance_map > 0.1  # 只显示成功概率>10%的区域
    heatmap_colored = plt.cm.hot(affordance_map)[..., :3]  # 获取RGB颜色
    
    # 叠加热力图
    overlay[heatmap_mask] = 0.6 * overlay[heatmap_mask] + 0.4 * heatmap_colored[heatmap_mask]
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("热力图叠加效果")
    axes[1, 0].axis('off')
    
    # 5. 采样点分布
    axes[1, 1].imshow(rgb_image)
    
    if 'candidates' in metadata:
        candidates = metadata['candidates']
        results = metadata['results']
        
        # 绘制成功和失败的采样点
        for i, (candidate, success) in enumerate(zip(candidates, results)):
            u, v, angle_idx = candidate
            color = 'green' if success else 'red'
            marker = 'o' if success else 'x'
            axes[1, 1].scatter(u, v, c=color, marker=marker, s=10, alpha=0.7)
    
    axes[1, 1].set_title(f"采样点分布 (绿色=成功, 红色=失败)")
    axes[1, 1].axis('off')
    
    # 6. 统计信息
    axes[1, 2].axis('off')
    if metadata:
        stats_text = f"""数据统计信息:
        
场景ID: {metadata['scene_id']}
图像尺寸: {metadata['image_size'][0]}x{metadata['image_size'][1]}
抓取角度数: {metadata['num_angles']}
采样点数量: {metadata['num_candidates']}
总体成功率: {metadata['success_rate']:.2%}
        
可供性地图统计:
非零像素数: {np.sum(affordance_map > 0)}
最大成功概率: {np.max(affordance_map):.2%}
平均成功概率: {np.mean(affordance_map[affordance_map > 0]):.2%}
        
高质量区域 (>50%):
像素数量: {np.sum(affordance_map > 0.5)}
占比: {np.sum(affordance_map > 0.5) / np.sum(affordance_map > 0) * 100:.1f}%
        """
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def visualize_angle_arrows(rgb_image, affordance_map, angle_map, metadata, threshold=0.3):
    """在高可供性区域绘制抓取角度箭头"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.imshow(rgb_image)
    
    # 找到高可供性区域
    high_affordance = affordance_map > threshold
    y_coords, x_coords = np.where(high_affordance)
    
    if len(x_coords) > 0:
        # 为了避免箭头过密，进行采样
        step = max(1, len(x_coords) // 50)  # 最多显示50个箭头
        sampled_indices = np.arange(0, len(x_coords), step)
        
        grasp_angles = np.array(metadata['grasp_angles_rad'])
        
        for idx in sampled_indices:
            x, y = x_coords[idx], y_coords[idx]
            angle_idx = angle_map[y, x]
            
            if angle_idx >= 0:  # 有效角度
                angle = grasp_angles[angle_idx]
                affordance = affordance_map[y, x]
                
                # 箭头长度和颜色基于可供性值
                arrow_length = 15 * affordance  # 长度与成功概率成正比
                
                # 计算箭头方向
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)
                
                # 绘制箭头
                ax.arrow(x, y, dx, dy, 
                        head_width=3, head_length=3,
                        fc='yellow', ec='red', 
                        alpha=0.8, linewidth=1.5)
    
    ax.set_title(f"抓取方向可视化 (阈值>{threshold:.1%})")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset(data_dir):
    """分析整个数据集的统计信息"""
    data_path = Path(data_dir)
    
    # 找到所有场景
    meta_files = list(data_path.glob("scene_*_meta.json"))
    
    if not meta_files:
        print(f"❌ 在 {data_dir} 中没有找到数据文件")
        return
    
    print(f"📊 分析数据集: {data_dir}")
    print(f"找到 {len(meta_files)} 个场景")
    print("=" * 60)
    
    success_rates = []
    total_candidates = 0
    total_successes = 0
    
    for meta_file in sorted(meta_files):
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        scene_id = metadata['scene_id']
        success_rate = metadata['success_rate']
        num_candidates = metadata['num_candidates']
        
        success_rates.append(success_rate)
        total_candidates += num_candidates
        total_successes += int(num_candidates * success_rate)
        
        print(f"场景 {scene_id:04d}: {success_rate:.2%} ({num_candidates} 个样本)")
    
    print("=" * 60)
    print(f"数据集统计:")
    print(f"  总场景数: {len(meta_files)}")
    print(f"  总采样点: {total_candidates}")
    print(f"  总成功数: {total_successes}")
    print(f"  平均成功率: {np.mean(success_rates):.2%}")
    print(f"  成功率标准差: {np.std(success_rates):.2%}")
    print(f"  成功率范围: {np.min(success_rates):.2%} - {np.max(success_rates):.2%}")
    
    # 绘制成功率分布
    plt.figure(figsize=(10, 6))
    plt.hist(success_rates, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('成功率')
    plt.ylabel('场景数量')
    plt.title('抓取成功率分布')
    plt.axvline(np.mean(success_rates), color='red', linestyle='--', 
                label=f'平均值: {np.mean(success_rates):.2%}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可供性数据可视化工具")
    parser.add_argument("--data_dir", type=str, default="data/affordance_dataset",
                       help="数据集路径")
    parser.add_argument("--scene_id", type=int, default=0,
                       help="要可视化的场景ID")
    parser.add_argument("--analyze", action="store_true",
                       help="分析整个数据集")
    parser.add_argument("--save", type=str, default=None,
                       help="保存可视化结果的路径")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.data_dir)
    else:
        # 加载和可视化单个场景
        rgb, depth, affordance, angles, metadata = load_scene_data(args.data_dir, args.scene_id)
        
        if rgb is None or affordance is None:
            print(f"❌ 无法加载场景 {args.scene_id} 的数据")
            return
        
        print(f"🎨 可视化场景 {args.scene_id}")
        
        # 主要可视化
        visualize_affordance_heatmap(rgb, affordance, angles, metadata, args.save)
        
        # 角度箭头可视化
        visualize_angle_arrows(rgb, affordance, angles, metadata)

if __name__ == "__main__":
    main()