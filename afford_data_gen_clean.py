# -*- coding: utf-8 -*-
"""
自监督抓取可供性数据生成器 v3 - 清理版
Self-supervised Grasp Affordance Data Generator

简单    if visualize:
        print(f"   📏 桌面深度: {table_depth:.3f}m (直方图峰值)")
        print(f"   📏 物体阈值: < {depth_threshold:.3f}m (比桌面近10mm+)")
        print(f"   🎯 深度检测: {depth_based_objects.sum()} 像素 ({100*depth_based_objects.sum()/(height*width):.1f}%)")1. 用RGB标准差找彩色物体
2. 使用这些像素的深度值
3. 生成抓取候选并测试
"""

import pybullet as p
import numpy as np
import time
import json
import os
from pathlib import Path
import cv2
import argparse

from environment_setup import setup_environment
from perception import set_topdown_camera, get_rgb_depth, get_rgb_depth_segmentation, pixel_to_world, CAMERA_PARAMS


# ==================== 配置参数 ====================

DATA_DIR = Path(__file__).parent.parent / "data" / "affordance_v4"
NUM_ANGLES = 16
ANGLE_BINS = np.linspace(0, np.pi, NUM_ANGLES, endpoint=False)

# 采样参数
FOREGROUND_STRIDE = 8
BACKGROUND_STRIDE = 64
MIN_DEPTH = 0.01
COLOR_DIFF_THRESHOLD = 30  # 颜色差异阈值：与桌子颜色的距离（越大越严格）
EDGE_MARGIN = 20  # 从图像边缘采样桌子颜色的边距（像素）

# 抓取参数
TABLE_TOP_Z = 0.625
PRE_GRASP_OFFSET = 0.25
GRASP_OFFSET = 0.05
POST_GRASP_OFFSET = 0.00
LIFT_HEIGHT = 0.30
GRIPPER_CLOSED = 0.00
FAST_STEPS = 120
SLOW_STEPS = 600


def create_data_dirs():
    """创建数据目录"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 数据目录: {DATA_DIR.absolute()}")


def sample_grasp_candidates(depth, num_angles=NUM_ANGLES, visualize=False, rgb=None, view_matrix=None, proj_matrix=None, seg_mask=None, object_ids=None):
    """基于PyBullet segmentation mask的物体分割策略
    
    使用相机的内置分割功能直接识别物体像素
    
    Args:
        depth: 深度图
        num_angles: 角度数量  
        visualize: 是否可视化
        rgb: RGB图像
        view_matrix, proj_matrix: 相机矩阵（用于调试坐标转换）
        seg_mask: PyBullet segmentation mask
        object_ids: 物体ID列表
    """
    height, width = depth.shape
    candidates = []
    
    if seg_mask is None or object_ids is None:
        print(f"   ⚠️  需要segmentation mask和object IDs")
        return candidates
    
    # Step 1: 使用PyBullet segmentation mask直接获取物体像素  
    object_mask = np.zeros((height, width), dtype=bool)
    for obj_id in object_ids:
        object_mask |= (seg_mask == obj_id)
    
    # 过滤掉深度无效的像素
    object_mask &= (depth > MIN_DEPTH)
    
    if visualize:
        print(f"   � Segmentation检测: {object_mask.sum()} 像素 ({100*object_mask.sum()/(height*width):.1f}%)")
        print(f"   � 物体IDs: {object_ids}")
        
        # 显示每个物体的像素数
        for obj_id in object_ids:
            obj_pixels = (seg_mask == obj_id).sum()
            print(f"      物体 ID={obj_id}: {obj_pixels} 像素")
        
        # 显示物体深度统计
        if object_mask.sum() > 0:
            obj_depths = depth[object_mask]
            valid_obj_depths = obj_depths[obj_depths > MIN_DEPTH]
            if len(valid_obj_depths) > 0:
                print(f"   📊 物体深度: min={valid_obj_depths.min():.3f}m, max={valid_obj_depths.max():.3f}m, mean={valid_obj_depths.mean():.3f}m")
        
        # 保存可视化
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 为每个物体分配不同的颜色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for idx, obj_id in enumerate(object_ids):
            obj_pixels = (seg_mask == obj_id)
            vis[obj_pixels] = colors[idx % len(colors)]
        
        # 标记图像中心
        center_v, center_u = height // 2, width // 2
        cv2.circle(vis, (center_u, center_v), 5, (255, 255, 255), -1)  # 白色=图像中心
        
        cv2.imwrite("/tmp/object_detection.png", vis)
        
        # 保存深度热图
        valid_depth = depth[depth > MIN_DEPTH]
        if len(valid_depth) > 0:
            depth_min, depth_max = valid_depth.min(), valid_depth.max()
            depth_vis = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-6) * 255, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_color[depth <= MIN_DEPTH] = [0, 0, 0]
            cv2.imwrite("/tmp/depth_heatmap.png", depth_color)
        
        # 保存segmentation mask可视化
        seg_vis = ((seg_mask % 10) * 25).astype(np.uint8)
        cv2.imwrite("/tmp/segmentation_mask.png", seg_vis)
        
        # 保存RGB图像
        if rgb is not None:
            cv2.imwrite("/tmp/rgb_image.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        print(f"   💾 可视化已保存:")
        print(f"      /tmp/object_detection.png (绿色=检测到的物体, 黄色=深度候选)")
        print(f"      /tmp/depth_heatmap.png (深度热图, 蓝色=近, 红色=远)")
        print(f"      /tmp/rgb_image.png (原始RGB图)")
        
        # 显示检测到的物体区域的位置
        if object_mask.sum() > 0:
            obj_coords = np.where(object_mask)
            obj_center_v = int(np.mean(obj_coords[0]))
            obj_center_u = int(np.mean(obj_coords[1]))
            print(f"   📍 检测到的物体中心像素: ({obj_center_u}, {obj_center_v})")
            print(f"   📍 图像中心像素: ({center_u}, {center_v})")
            
            # 显示物体区域的深度样本
            sample_v, sample_u = obj_coords[0][0], obj_coords[1][0]
            sample_depth = depth[sample_v, sample_u]
            sample_rgb = rgb[sample_v, sample_u]
            print(f"   🔍 物体区域样本像素 ({sample_u}, {sample_v}):")
            print(f"      深度={sample_depth:.3f}m, RGB={sample_rgb}")
            
            # 测试像素到世界坐标的转换
            if view_matrix is not None and proj_matrix is not None:
                test_world = pixel_to_world(obj_center_u, obj_center_v, depth[obj_center_v, obj_center_u], view_matrix, proj_matrix)
                print(f"   🧭 物体中心像素 → 世界坐标: [{test_world[0]:.3f}, {test_world[1]:.3f}, {test_world[2]:.3f}]")
    
    if object_mask.sum() == 0:
        print(f"   ⚠️  未检测到物体")
        return candidates
    
    print(f"   🎯 检测到物体像素: {object_mask.sum()}")
    
    # Step 3: 在物体区域采样
    # 对于小物体（< 100像素），直接在所有像素上采样，不erode
    if object_mask.sum() < 100:
        print(f"   🎯 小物体：直接采样所有像素")
        # 获取所有物体像素
        obj_coords = np.where(object_mask)
        obj_center_v = int(np.mean(obj_coords[0]))
        obj_center_u = int(np.mean(obj_coords[1]))
        
        # 对每个物体像素，按距离中心排序
        positions = []
        for i in range(len(obj_coords[0])):
            v, u = obj_coords[0][i], obj_coords[1][i]
            dist = (u - obj_center_u)**2 + (v - obj_center_v)**2
            positions.append((dist, u, v))
        
        positions.sort()
        
        # 为每个像素添加所有角度的候选
        for theta_idx in range(num_angles):
            theta = ANGLE_BINS[theta_idx]
            for dist, u, v in positions:
                candidates.append((u, v, theta_idx, theta))
    else:
        # 大物体：使用erosion找到中心区域
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=2)
        
        if eroded.sum() > 0:
            print(f"   🎯 物体中心: {eroded.sum()} 像素")
            # 中心区域
            coords = np.where(eroded > 0)
            center_v = int(np.mean(coords[0]))
            center_u = int(np.mean(coords[1]))
            
            positions = []
            for v in range(0, height, 4):
                for u in range(0, width, 4):
                    if eroded[v, u]:
                        dist = (u - center_u)**2 + (v - center_v)**2
                        positions.append((dist, u, v))
            
            positions.sort()
            
            # 添加候选（不同角度）
            for theta_idx in range(num_angles):
                theta = ANGLE_BINS[theta_idx]
                for dist, u, v in positions:
                    candidates.append((u, v, theta_idx, theta))
        
        # 边缘区域
        edge = object_mask & (~eroded.astype(bool))
        if edge.sum() > 0:
            edge_pos = []
            for v in range(0, height, FOREGROUND_STRIDE):
                for u in range(0, width, FOREGROUND_STRIDE):
                    if edge[v, u]:
                        edge_pos.append((u, v))
            
            for theta_idx in [0, 4, 8, 12]:
                if theta_idx < num_angles:
                    theta = ANGLE_BINS[theta_idx]
                    for u, v in edge_pos:
                        candidates.append((u, v, theta_idx, theta))
    
    # 背景采样（减少背景样本数量，重点测试物体）
    bg_count = 0
    bg_stride = BACKGROUND_STRIDE * 2  # 更稀疏的背景采样
    for v in range(0, height, bg_stride):
        for u in range(0, width, bg_stride):
            if not object_mask[v, u] and depth[v, u] > MIN_DEPTH:
                candidates.append((u, v, 0, 0.0))
                bg_count += 1
    
    fg_count = len(candidates) - bg_count
    print(f"   📍 采样 {len(candidates)} 个候选 (前景: {fg_count}, 背景: {bg_count})")
    
    if visualize and len(candidates) > 100:
        # 在可视化模式下，优先测试前景候选
        print(f"   🧪 可视化模式：测试前 100 个（优先前景）")
        candidates = candidates[:100]
    
    return candidates


def fast_grasp_test(robot_id, world_pos, grasp_angle, object_ids, visualize=False):
    """快速抓取测试"""
    ee_link = 11
    steps = SLOW_STEPS if visualize else FAST_STEPS
    
    # 检查Z坐标
    if world_pos[2] < TABLE_TOP_Z - 0.05 or world_pos[2] > TABLE_TOP_Z + 0.30:
        if visualize:
            print(f"         ⚠️  Z坐标不合理 ({world_pos[2]:.3f}m)")
        return False
    
    # 检查XY工作空间
    dist = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
    if dist < 0.35 or dist > 0.80:
        if visualize:
            print(f"         ⚠️  超出工作范围 (距离={dist:.3f}m)")
        return False
    
    initial_z = {}
    for obj_id in object_ids:
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        initial_z[obj_id] = pos[2]
    
    try:
        ori = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        
        # 预抓取
        if visualize:
            print(f"         ↑ 预抓取...")
        pre_pos = [world_pos[0], world_pos[1], world_pos[2] + PRE_GRASP_OFFSET]
        if not move_fast(robot_id, ee_link, pre_pos, ori, steps):
            return False
        
        # 下降
        if visualize:
            print(f"         ↓ 下降...")
        grasp_pos = [world_pos[0], world_pos[1], world_pos[2] + GRASP_OFFSET]
        if not move_fast(robot_id, ee_link, grasp_pos, ori, steps, slow=True):
            return False
        
        # 闭合夹爪
        if visualize:
            print(f"         🤏 闭合...")
        close_gripper_slow(robot_id, steps//2)
        
        # 检查夹爪
        finger_state = p.getJointState(robot_id, 9)[0]
        if finger_state < 0.001:
            if visualize:
                print(f"         ⚠️  夹爪未闭合")
            return False
        
        # 抬起
        if visualize:
            print(f"         ↑↑ 抬起...")
        lift_pos = [grasp_pos[0], grasp_pos[1], world_pos[2] + LIFT_HEIGHT]
        if not move_fast(robot_id, ee_link, lift_pos, ori, steps):
            return False
        
        # 判断成功
        for obj_id in object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] - initial_z[obj_id] > 0.08:
                if visualize:
                    print(f"         ✅ 成功！抬起 {(pos[2]-initial_z[obj_id])*100:.1f}cm")
                return True
        
        if visualize:
            print(f"         ❌ 失败")
        return False
    
    except Exception as e:
        if visualize:
            print(f"         ⚠️ 异常: {e}")
        return False


def move_fast(robot_id, ee_link, target_pos, target_ori, max_steps, slow=False):
    """移动到目标位置"""
    ll, ul, jr, rp = [], [], [], []
    for i in range(7):
        info = p.getJointInfo(robot_id, i)
        ll.append(info[8])
        ul.append(info[9])
        jr.append(info[9] - info[8])
        rp.append(p.getJointState(robot_id, i)[0])
    
    joints = p.calculateInverseKinematics(
        robot_id, ee_link, target_pos, target_ori,
        lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp
    )
    
    if not joints or len(joints) < 7:
        return False
    
    velocity = 0.3 if slow else 1.0
    force = 300 if slow else 500
    
    for i in range(7):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL,
            targetPosition=joints[i], force=force, maxVelocity=velocity
        )
    
    for _ in range(max_steps):
        p.stepSimulation()
        time.sleep(1./240.)
    
    current = p.getLinkState(robot_id, ee_link)[0]
    dist = np.linalg.norm(np.array(current) - np.array(target_pos))
    return dist < 0.10


def close_gripper_slow(robot_id, steps):
    """慢速闭合夹爪"""
    pos = GRIPPER_CLOSED / 2.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)


def open_gripper_fast(robot_id):
    """打开夹爪"""
    pos = 0.04 / 2.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=50)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=50)
    for _ in range(10):
        p.stepSimulation()


def reset_robot_home(robot_id):
    """重置机器人"""
    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(7):
        p.resetJointState(robot_id, i, home[i])
    open_gripper_fast(robot_id)


def generate_scene_data(scene_id, num_objects=3, visualize=False):
    """生成单个场景数据"""
    print(f"\n🎬 场景 {scene_id:04d}")
    
    client = p.connect(p.GUI if visualize else p.DIRECT)
    if client < 0:
        return False
    
    try:
        robot_id, object_ids = setup_environment(num_objects=num_objects)
        if not object_ids:
            return False
        
        print("   ⏳ 等待物体稳定...")
        for _ in range(120):
            p.stepSimulation()
        
        if visualize:
            print("   ⏸️  按 Enter 继续...")
            input()
        
        print("   📸 采集图像")
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        rgb, depth, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
        
        if visualize:
            for i, obj_id in enumerate(object_ids):
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                print(f"   📦 物体{i+1} (ID={obj_id}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"   📊 深度: min={depth.min():.3f}, max={depth.max():.3f}")
            print(f"   🎭 Segmentation mask: {len(np.unique(seg_mask))} 个不同ID")
        
        candidates = sample_grasp_candidates(depth, NUM_ANGLES, visualize, rgb, view_matrix, proj_matrix, seg_mask, object_ids)
        
        label = np.zeros((height, width, NUM_ANGLES + 1), dtype=np.uint8)
        
        print(f"   🧪 测试 {len(candidates)} 个候选")
        success_count = 0
        
        for idx, (u, v, theta_idx, theta) in enumerate(candidates):
            if depth[v, u] < MIN_DEPTH:
                continue
            
            if visualize:
                print(f"\n      === 候选 {idx+1}/{len(candidates)} ===")
                print(f"         像素: ({u}, {v}), 角度: {np.degrees(theta):.1f}°")
            
            world_pos = pixel_to_world(u, v, depth[v, u], view_matrix, proj_matrix)
            
            if visualize:
                print(f"         世界坐标: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")
            
            success = fast_grasp_test(robot_id, world_pos, theta, object_ids, visualize)
            
            if success:
                label[v, u, theta_idx] = 1
                success_count += 1
            
            reset_robot_home(robot_id)
            
            if not visualize and (idx + 1) % 10 == 0:
                print(f"      {idx+1}/{len(candidates)} | 成功: {success_count}")
        
        has_success = label[:, :, :-1].sum(axis=2) > 0
        label[:, :, -1] = (~has_success).astype(np.uint8)
        
        rate = success_count / len(candidates) if len(candidates) > 0 else 0
        print(f"   ✅ 成功率: {rate*100:.1f}%")
        
        save_scene_data(scene_id, rgb, depth, label, {
            "num_objects": num_objects,
            "num_samples": len(candidates),
            "success_count": success_count,
            "success_rate": rate
        })
        
        return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        p.disconnect()


def save_scene_data(scene_id, rgb, depth, label, metadata):
    """保存数据"""
    prefix = DATA_DIR / f"scene_{scene_id:04d}"
    cv2.imwrite(str(prefix) + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(str(prefix) + "_depth.npy", depth)
    np.save(str(prefix) + "_label.npy", label)
    with open(str(prefix) + "_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   💾 保存: {prefix}_*")


def generate_dataset(num_scenes, num_objects_range=(1, 3), visualize_first=False):
    """批量生成"""
    print("=" * 60)
    print("🚀 生成数据集")
    print("=" * 60)
    
    create_data_dirs()
    success_scenes = 0
    start = time.time()
    
    for scene_id in range(num_scenes):
        num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
        vis = visualize_first and (scene_id == 0)
        
        if generate_scene_data(scene_id, num_objects, vis):
            success_scenes += 1
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"✅ 完成！{success_scenes}/{num_scenes}")
    print(f"   耗时: {elapsed:.1f}s ({elapsed/num_scenes:.1f}s/场景)")
    print(f"   位置: {DATA_DIR.absolute()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=10)
    parser.add_argument("--num_objects", type=int, nargs=2, default=[1, 3])
    parser.add_argument("--visualize_first", action="store_true")
    args = parser.parse_args()
    
    generate_dataset(
        num_scenes=args.num_scenes,
        num_objects_range=tuple(args.num_objects),
        visualize_first=args.visualize_first
    )


if __name__ == "__main__":
    main()
