# -*- coding: utf-8 -*-
import os, time, numpy as np, cv2
import pybullet as p
from geom import setup_scene, set_topdown_camera, get_rgb_depth

SAVE_DIR = "../data/sim_screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)

def spawn_scene(n_objects=2):
    """使用统一的稳定配置生成场景"""
    p.resetSimulation()
    robot_id, table_id, obj_ids = setup_scene(add_objects=True, n_objects=n_objects)
    return robot_id, obj_ids

if __name__ == "__main__":
    p.connect(p.GUI)
    W, H, view, proj = set_topdown_camera()  # 顶视相机
    N = 10  # 先生成10张
    for i in range(N):
        spawn_scene(n_objects=np.random.randint(1, 4))
        rgb, depth = get_rgb_depth(W, H, view, proj)
        cv2.imwrite(os.path.join(SAVE_DIR, f"scene_{i:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[{i+1}/{N}] saved.")
    p.disconnect()
    print(f"✅ 已保存到 {SAVE_DIR}")
