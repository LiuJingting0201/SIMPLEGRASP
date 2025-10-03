import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

test_dir = "data/affordance_v5/test"

# 获取所有场景id
def get_scene_ids(test_dir):
    ids = set()
    for fname in os.listdir(test_dir):
        if fname.startswith("scene_") and fname.endswith("_rgb.png"):
            sid = fname.split("_")[1]
            ids.add(sid)
    return sorted(list(ids))

def visualize_scene(scene_id):
    rgb_path = os.path.join(test_dir, f"scene_{scene_id}_rgb.png")
    depth_path = os.path.join(test_dir, f"scene_{scene_id}_depth.npy")
    afford_path = os.path.join(test_dir, f"scene_{scene_id}_affordance.npy")
    angle_path = os.path.join(test_dir, f"scene_{scene_id}_angles.npy")

    rgb = np.array(Image.open(rgb_path))
    depth = np.load(depth_path)
    afford = np.load(afford_path)
    angle = np.load(angle_path)

    plt.figure(figsize=(15, 6))  # 2行3列布局

    plt.subplot(2, 3, 1)
    plt.title("RGB")
    plt.imshow(rgb)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Depth (abs)")
    plt.imshow(depth, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Depth (rel, -1cm to 3cm)")
    # 自动检测桌面高度
    center_region = depth[96:160, 96:160]  # 中心64x64区域
    table_depth = np.percentile(center_region, 10)  # 桌面深度
    relative_depth = depth - table_depth
    # 显示-1cm到3cm的范围，让微小差异更明显
    relative_depth_clipped = np.clip(relative_depth, -0.01, 0.03)
    plt.imshow(relative_depth_clipped, cmap='gray', vmin=-0.01, vmax=0.03)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Affordance")
    # 使用jet colormap，高可供性区域会更明显
    plt.imshow(afford, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Angle")
    plt.imshow(angle, cmap='hsv')
    plt.axis('off')

    plt.suptitle(f"Scene {scene_id}")
    plt.tight_layout()
    out_path = f"vis/scene_{scene_id}_vis.png"
    plt.savefig(out_path)
    plt.close()
    print(f"已保存: {out_path}")

if __name__ == "__main__":
    scene_ids = get_scene_ids(test_dir)
    print(f"可视化测试集场景: {scene_ids}")
    for sid in scene_ids[:5]:  # 只可视化前5个
        visualize_scene(sid)
