import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse

def get_scene_ids(data_dir):
    """获取指定目录中的所有场景ID"""
    ids = set()
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return []

    for fname in os.listdir(data_dir):
        if fname.startswith("scene_") and fname.endswith("_rgb.png"):
            parts = fname.split("_")
            if len(parts) >= 3:
                sid = parts[1]
                ids.add(sid)
    return sorted(list(ids))

def load_scene_data(data_dir, scene_id):
    """加载场景的所有数据"""
    base_path = os.path.join(data_dir, f"scene_{scene_id}")

    data = {}
    try:
        # 加载图像和数组
        rgb_path = f"{base_path}_rgb.png"
        depth_path = f"{base_path}_depth.npy"
        afford_path = f"{base_path}_affordance.npy"
        angle_path = f"{base_path}_angles.npy"
        meta_path = f"{base_path}_meta.json"

        if os.path.exists(rgb_path):
            data['rgb'] = np.array(Image.open(rgb_path))
        if os.path.exists(depth_path):
            data['depth'] = np.load(depth_path)
        if os.path.exists(afford_path):
            data['affordance'] = np.load(afford_path)
        if os.path.exists(angle_path):
            data['angles'] = np.load(angle_path)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data['meta'] = json.load(f)

    except Exception as e:
        print(f"加载场景 {scene_id} 数据时出错: {e}")

    return data

def visualize_scene(data_dir, scene_id, save_path=None):
    """可视化单个场景"""
    data = load_scene_data(data_dir, scene_id)

    if not data:
        print(f"场景 {scene_id} 没有数据")
        return

    # 创建子图布局
    fig = plt.figure(figsize=(16, 10))

    # RGB图像
    if 'rgb' in data:
        plt.subplot(2, 4, 1)
        plt.title("RGB Image")
        plt.imshow(data['rgb'])
        plt.axis('off')

    # 深度图 - 相对深度 (数据已经是相对深度)
    if 'depth' in data:
        plt.subplot(2, 4, 2)
        plt.title("Relative Depth")
        # 数据已经是相对深度，显示实际范围
        depth_min, depth_max = data['depth'].min(), data['depth'].max()
        plt.imshow(data['depth'], cmap='plasma', vmin=depth_min, vmax=depth_max)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    # 深度图 - 增强对比度版本
    if 'depth' in data:
        plt.subplot(2, 4, 3)
        plt.title("Depth (Enhanced Contrast)")
        # 使用百分位数来增强对比度，显示主要变化范围
        depth_flat = data['depth'].flatten()
        vmin = np.percentile(depth_flat, 5)   # 5th percentile
        vmax = np.percentile(depth_flat, 95)  # 95th percentile
        plt.imshow(data['depth'], cmap='plasma', vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    # 可供性地图
    if 'affordance' in data:
        plt.subplot(2, 4, 4)
        plt.title("Affordance Map")
        plt.imshow(data['affordance'], cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    # 角度地图
    if 'angles' in data:
        plt.subplot(2, 4, 5)
        plt.title("Grasp Angles")
        # 将角度归一化到0-1范围用于HSV色彩映射
        angle_norm = data['angles'] / (2 * np.pi)
        plt.imshow(angle_norm, cmap='hsv')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    # 可供性叠加RGB
    if 'rgb' in data and 'affordance' in data:
        plt.subplot(2, 4, 6)
        plt.title("RGB + Affordance Overlay")
        rgb_copy = data['rgb'].copy()
        # 在可供性点上叠加红色标记
        afford_mask = data['affordance'] > 0.5
        rgb_copy[afford_mask] = rgb_copy[afford_mask] * 0.7 + np.array([255, 0, 0]) * 0.3
        plt.imshow(rgb_copy.astype(np.uint8))
        plt.axis('off')

    # 统计信息
    if 'meta' in data:
        plt.subplot(2, 4, 7)
        plt.title("Scene Statistics")
        plt.axis('off')

        meta = data['meta']
        stats_text = ".1f"".1f"f"""
Scene ID: {meta.get('scene_id', 'N/A')}
Image Shape: {meta.get('image_shape', 'N/A')}
Candidates: {meta.get('num_candidates', 0)}
Successful: {meta.get('num_successful', 0)}
Success Rate: {meta.get('success_rate', 0):.1%}
Depth Range: {data.get('depth', np.array([])).min():.6f} to {data.get('depth', np.array([])).max():.6f}
"""
        plt.text(0.1, 0.8, stats_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 候选点可视化
    if 'rgb' in data and 'meta' in data:
        plt.subplot(2, 4, 8)
        plt.title("Grasp Candidates")
        rgb_copy = data['rgb'].copy()

        candidates = meta.get('candidates', [])
        successful_points = []
        failed_points = []

        for cand in candidates:
            if cand.get('success', False):
                successful_points.append(cand['pixel'])
            else:
                failed_points.append(cand['pixel'])

        # 绘制失败点（红色）
        for point in failed_points[:50]:  # 限制显示数量
            if len(point) >= 2:
                plt.scatter(point[0], point[1], c='red', s=10, alpha=0.6)

        # 绘制成功点（绿色）
        for point in successful_points[:50]:  # 限制显示数量
            if len(point) >= 2:
                plt.scatter(point[0], point[1], c='green', s=15, alpha=0.8, marker='x')

        plt.imshow(rgb_copy)
        plt.axis('off')

    plt.suptitle(f"Scene {scene_id} Visualization", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='可视化可供性数据集')
    parser.add_argument('--data_dir', type=str, default='data/affordance_v5/train',
                       help='数据目录路径')
    parser.add_argument('--scene_id', type=str, help='指定场景ID，不指定则可视化所有')
    parser.add_argument('--max_scenes', type=int, default=5, help='最大可视化场景数')
    parser.add_argument('--save_dir', type=str, default='vis', help='保存目录')

    args = parser.parse_args()

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        print("可用的目录:")
        base_dir = os.path.dirname(args.data_dir)
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    scene_ids = get_scene_ids(item_path)
                    if scene_ids:
                        print(f"  {item_path}: {len(scene_ids)} 个场景")
        return

    # 获取场景ID
    scene_ids = get_scene_ids(args.data_dir)
    print(f"找到场景: {scene_ids}")

    if not scene_ids:
        print(f"在 {args.data_dir} 中没有找到场景数据")
        return

    # 选择要可视化的场景
    if args.scene_id:
        if args.scene_id not in scene_ids:
            print(f"场景 {args.scene_id} 不存在")
            return
        scenes_to_vis = [args.scene_id]
    else:
        scenes_to_vis = scene_ids[:args.max_scenes]

    print(f"将可视化 {len(scenes_to_vis)} 个场景: {scenes_to_vis}")

    # 可视化每个场景
    for scene_id in scenes_to_vis:
        save_path = os.path.join(args.save_dir, f"scene_{scene_id}_detailed_vis.png")
        visualize_scene(args.data_dir, scene_id, save_path)

    print("可视化完成!")

if __name__ == "__main__":
    main()
