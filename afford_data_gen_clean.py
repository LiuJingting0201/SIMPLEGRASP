# -*- coding: utf-8 -*-
"""
自监督抓取可供性数据生成器 v3 - 清理版
Self-supervised Grasp Affordance Data Generator

简单策略：
1. 用RGB标准差找彩色物体
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
PRE_GRASP_OFFSET = 0.12  # 预抓取高度（从物体顶部）
GRASP_OFFSET = -0.01    # 抓取高度：物体顶部下方5mm（进入物体以抓取）
POST_GRASP_OFFSET = 0.00
LIFT_HEIGHT = 0.30
GRIPPER_CLOSED = 0.00
FAST_STEPS = 120
SLOW_STEPS = 600


def create_data_dirs():
    """创建数据目录"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 数据目录: {DATA_DIR.absolute()}")


def estimate_object_height(depth, object_mask, percentile=10):
    """估计物体表面高度
    
    使用检测到的物体像素的深度值估计表面高度
    使用较小百分位数来避免噪声和边缘效应
    
    Args:
        depth: 深度图
        object_mask: 物体mask
        percentile: 使用的百分位数（默认10 = 最近的10%像素）
    
    Returns:
        物体表面高度（世界坐标Z值）
    """
    obj_depths = depth[object_mask]
    valid_depths = obj_depths[obj_depths > MIN_DEPTH]
    
    if len(valid_depths) == 0:
        return None
    
    # 使用较小百分位数的深度值（最接近相机 = 最高点）
    surface_depth = np.percentile(valid_depths, percentile)
    
    # 深度到世界Z的转换（简化版，假设俯视相机）
    # 相机高度 = TABLE_TOP_Z + camera_distance
    # 物体Z = 相机高度 - 深度
    camera_height = TABLE_TOP_Z + 1.2  # CAMERA_DISTANCE = 1.2
    object_z = camera_height - surface_depth
    
    return object_z


def sample_grasp_candidates(depth, num_angles=NUM_ANGLES, visualize=False, rgb=None, view_matrix=None, proj_matrix=None, seg_mask=None, object_ids=None):
    """基于PyBullet segmentation mask的物体分割策略 - 修复版"""
    height, width = depth.shape
    candidates = []
    
    if seg_mask is None or object_ids is None:
        print(f"   ⚠️  需要segmentation mask和object IDs")
        return candidates
    
    # ✨ 关键修复：首先检查是否真的有物体存在
    if len(object_ids) == 0:
        print(f"   ⚠️  物体列表为空，无候选点")
        return candidates
    
    # Step 1: 使用PyBullet segmentation mask直接获取物体像素  
    object_mask = np.zeros((height, width), dtype=bool)
    valid_object_count = 0
    
    print(f"   🔍 分析物体像素...")
    for obj_id in object_ids:
        try:
            # ✨ 强化验证：检查物体是否真的存在且在桌面上
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            
            # 检查物体是否在合理位置（在桌面上方，在工作区域内）
            if (pos[2] < TABLE_TOP_Z or pos[2] > TABLE_TOP_Z + 0.3 or
                abs(pos[0] - 0.6) > 0.4 or abs(pos[1]) > 0.4):
                print(f"      物体 ID={obj_id}: 位置异常 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]，跳过")
                continue
            
            obj_pixels = (seg_mask == obj_id)
            pixel_count = obj_pixels.sum()
            print(f"      物体 ID={obj_id}: {pixel_count} 像素, 位置=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            if pixel_count > 5:  # 至少5个像素才认为是有效物体
                object_mask |= obj_pixels
                valid_object_count += 1
        except:
            # 物体不存在
            print(f"      物体 ID={obj_id}: 不存在或无法访问")
            continue
    
    # ✨ 关键修复：如果没有有效物体像素，明确返回空列表
    if valid_object_count == 0 or object_mask.sum() == 0:
        print(f"   ❌ 未检测到有效物体 (有效物体数: {valid_object_count}, 像素数: {object_mask.sum()})")
        return []  # 明确返回空列表，触发重新生成
    
    # 其余代码保持不变...
    # 过滤掉深度无效的像素
    valid_depth_mask = depth > MIN_DEPTH
    object_mask &= valid_depth_mask
    
    print(f"   🎯 有效物体像素: {object_mask.sum()} (深度过滤后)")
    
    # ✨ 再次检查：如果过滤后没有有效像素，返回空
    if object_mask.sum() == 0:
        print(f"   ❌ 过滤后无有效物体像素")
        return []
    
    # ... 继续原有的候选点生成逻辑
    obj_coords = np.where(object_mask)
    if len(obj_coords[0]) == 0:
        print(f"   ❌ 没有有效的物体坐标")
        return []
    
    print(f"   📍 生成候选点...")
    
    # 计算物体中心
    obj_center_v = int(np.mean(obj_coords[0]))
    obj_center_u = int(np.mean(obj_coords[1]))
    
    print(f"      物体中心: ({obj_center_u}, {obj_center_v})")
    
    # 验证中心点的深度
    center_depth = depth[obj_center_v, obj_center_u]
    print(f"      中心深度: {center_depth:.3f}m")
    
    if center_depth > MIN_DEPTH:
        # 生成中心点的多个角度候选
        for theta_idx in range(0, min(4, num_angles)):  # 最多4个角度
            theta = ANGLE_BINS[theta_idx]
            candidates.append((obj_center_u, obj_center_v, theta_idx, theta))
            print(f"      添加中心候选: ({obj_center_u}, {obj_center_v}), 角度={np.degrees(theta):.1f}°")
    
    # 添加物体区域的其他点（稀疏采样）
    step = max(1, len(obj_coords[0]) // 10)  # 最多10个额外点
    for i in range(0, len(obj_coords[0]), step):
        v, u = obj_coords[0][i], obj_coords[1][i]
        if depth[v, u] > MIN_DEPTH:
            candidates.append((u, v, 0, 0.0))  # 只用0度角
            if len(candidates) >= 15:  # 限制候选数量
                break
    
    # 只有在有物体候选的情况下才添加少量背景
    bg_count = 0
    if len(candidates) > 0:
        for v in range(0, height, BACKGROUND_STRIDE * 4):
            for u in range(0, width, BACKGROUND_STRIDE * 4):
                if not object_mask[v, u] and depth[v, u] > MIN_DEPTH:
                    candidates.append((u, v, 0, 0.0))
                    bg_count += 1
                    if bg_count >= 2:  # 最多2个背景样本
                        break
            if bg_count >= 2:
                break
    
    fg_count = len(candidates) - bg_count
    print(f"   📍 最终采样 {len(candidates)} 个候选 (前景: {fg_count}, 背景: {bg_count})")
    
    if fg_count == 0:
        print(f"   ❌ 警告：没有生成前景候选点！")
        return []  # 返回空列表，触发重新生成
    
    return candidates


# ... 其余函数保持不变 ...

def fast_grasp_test(robot_id, world_pos, grasp_angle, object_ids, visualize=False):
    """快速抓取测试 - 增强调试版本"""
    ee_link = 11
    steps = SLOW_STEPS if visualize else FAST_STEPS
    
    print(f"         🎯 开始抓取测试: 位置=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}], 角度={np.degrees(grasp_angle):.1f}°")
    
    # 检查Z坐标
    if world_pos[2] < TABLE_TOP_Z - 0.05 or world_pos[2] > TABLE_TOP_Z + 0.30:
        print(f"         ❌ Z坐标不合理 ({world_pos[2]:.3f}m), 桌面={TABLE_TOP_Z:.3f}m")
        return False
    
    # 检查XY工作空间
    dist = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
    if dist < 0.35 or dist > 0.80:
        print(f"         ❌ 超出工作范围 (距离={dist:.3f}m), 范围=[0.35, 0.80]m")
        return False
    
    print(f"         ✅ 位置检查通过: Z={world_pos[2]:.3f}m, 距离={dist:.3f}m")
    
    # 记录初始物体状态
    initial_z = {}
    initial_pos = {}
    for obj_id in object_ids:
        try:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            initial_z[obj_id] = pos[2]
            initial_pos[obj_id] = pos
            print(f"         📦 物体{obj_id}初始位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        except:
            print(f"         ❌ 无法获取物体{obj_id}位置")
            return False
    
    try:
        ori = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        print(f"         🎯 目标姿态: 俯视, 旋转={np.degrees(grasp_angle):.1f}°")
        
        # 预抓取
        print(f"         ↑ 预抓取阶段...")
        pre_pos = [world_pos[0], world_pos[1], world_pos[2] + PRE_GRASP_OFFSET]
        print(f"            目标: [{pre_pos[0]:.3f}, {pre_pos[1]:.3f}, {pre_pos[2]:.3f}]")
        
        if not move_fast(robot_id, ee_link, pre_pos, ori, steps):
            print(f"         ❌ 预抓取运动失败")
            return False
        
        # 检查实际到达位置
        actual_pos = p.getLinkState(robot_id, ee_link)[0]
        pos_error = np.linalg.norm(np.array(actual_pos) - np.array(pre_pos))
        print(f"            实际位置: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"            位置误差: {pos_error*1000:.1f}mm")
        
        # 检查物体是否被推走（预抓取阶段不应该碰到物体）
        for obj_id in object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                xy_dist = np.sqrt((pos[0]-initial_pos[obj_id][0])**2 + (pos[1]-initial_pos[obj_id][1])**2)
                z_change = pos[2] - initial_z[obj_id]
                print(f"            物体{obj_id}移动: XY={xy_dist*100:.1f}cm, Z={z_change*100:.1f}cm")
                if xy_dist > 0.05:  # 移动超过5cm
                    print(f"         ❌ 预抓取时物体ID={obj_id}被推走 {xy_dist*100:.1f}cm")
                    return False
            except:
                print(f"         ❌ 物体{obj_id}消失")
                return False
        
        # 下降
        print(f"         ↓ 下降阶段...")
        grasp_pos = [world_pos[0], world_pos[1], world_pos[2] + GRASP_OFFSET]
        print(f"            目标深度: Z={grasp_pos[2]:.3f}m (物体={world_pos[2]:.3f}m, offset={GRASP_OFFSET:+.3f}m)")
        
        if not move_fast(robot_id, ee_link, grasp_pos, ori, steps, slow=True):
            print(f"         ❌ 下降运动失败")
            return False
        
        # 检查下降后的实际位置
        actual_pos = p.getLinkState(robot_id, ee_link)[0]
        pos_error = np.linalg.norm(np.array(actual_pos) - np.array(grasp_pos))
        print(f"            实际位置: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"            位置误差: {pos_error*1000:.1f}mm")
        
        # 检查物体状态
        for obj_id in object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                xy_dist = np.sqrt((pos[0]-initial_pos[obj_id][0])**2 + (pos[1]-initial_pos[obj_id][1])**2)
                z_change = pos[2] - initial_z[obj_id]
                print(f"            物体{obj_id}移动: XY={xy_dist*100:.1f}cm, Z={z_change*100:.1f}cm")
                if xy_dist > 0.05:
                    print(f"         ❌ 下降时物体ID={obj_id}被推走 {xy_dist*100:.1f}cm")
                    return False
            except:
                print(f"         ❌ 物体{obj_id}消失")
                return False
        
        # 闭合夹爪
        print(f"         🤏 闭合夹爪...")
        close_gripper_slow(robot_id, steps//2)
        
        # 检查夹爪状态
        finger_state = p.getJointState(robot_id, 9)[0]
        finger_force = p.getJointState(robot_id, 9)[3]  # 获取力矩
        print(f"            夹爪状态: 位置={finger_state:.4f}, 力矩={finger_force:.2f}")
        print(f"            判断: {'有物体' if finger_state > 0.001 else '无物体'}")
        
        if finger_state < 0.001:
            print(f"         ❌ 夹爪未闭合（物体太小或位置不对）")
            return False
        
        # 抬起
        print(f"         ↑↑ 抬起阶段...")
        lift_pos = [grasp_pos[0], grasp_pos[1], world_pos[2] + LIFT_HEIGHT]
        print(f"            目标高度: Z={lift_pos[2]:.3f}m")
        
        if not move_fast(robot_id, ee_link, lift_pos, ori, steps):
            print(f"         ❌ 抬起运动失败")
            return False
        
        # 检查抬起后的位置
        actual_pos = p.getLinkState(robot_id, ee_link)[0]
        print(f"            实际位置: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        
        # 判断成功
        success = False
        for obj_id in object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                lift_height = pos[2] - initial_z[obj_id]
                print(f"            物体{obj_id}: 当前Z={pos[2]:.3f}m, 抬起={lift_height*100:.1f}cm")
                if lift_height > 0.08:
                    print(f"         ✅ 成功！物体{obj_id}抬起 {lift_height*100:.1f}cm")
                    success = True
            except:
                print(f"            物体{obj_id}: 可能被移除了")
        
        if not success:
            print(f"         ❌ 失败：没有物体被抬起")
        
        return success
    
    except Exception as e:
        print(f"         ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
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
    """重置机器人到初始位置"""
    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    
    # 使用位置控制而不是直接设置关节状态，更平滑
    for i in range(7):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL,
            targetPosition=home[i], 
            force=500, 
            maxVelocity=2.0
        )
    
    # 确保夹爪打开
    open_gripper_fast(robot_id)
    
    # 等待到位
    for _ in range(120):
        p.stepSimulation()
        
        # 检查是否到位
        all_in_position = True
        for i in range(7):
            current = p.getJointState(robot_id, i)[0]
            if abs(current - home[i]) > 0.05:  # 容差3度
                all_in_position = False
                break
        
        if all_in_position:
            break


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
        
        # 确保机器人在初始位置
        reset_robot_home(robot_id)
        for _ in range(60):
            p.stepSimulation()
        
        if visualize:
            print("   ⏸️  按 Enter 继续...")
            input()
        
        # 主循环：持续抓取直到没有物体
        total_samples = 0
        total_success = 0
        grasp_attempt = 0
        consecutive_failures = 0
        
        # 用于保存最终数据的变量
        final_rgb = None
        final_depth = None
        final_label = None
        
        while grasp_attempt < 50:
            grasp_attempt += 1
            
            print(f"\n   📸 更新相机图像 (尝试 {grasp_attempt})")
            
            # 确保机器人回到初始位置
            print("   🏠 确保机器人回到初始位置...")
            reset_robot_home(robot_id)
            
            # 等待机器人完全稳定
            for _ in range(120):
                p.stepSimulation()
            
            # ✨ 关键修复：先更新物体状态再拍照
            print("   🔄 更新物体状态...")
            from environment_setup import update_object_states
            old_count = len(object_ids)
            object_ids = update_object_states(object_ids)
            new_count = len(object_ids)
            
            print(f"   📦 物体状态: {old_count} → {new_count}")
            
            # ✨ 立即检查是否需要重新生成物体（不等待2次）
            if len(object_ids) == 0:
                print("   ⚠️  桌面为空，立即重新生成物体...")
                from environment_setup import reset_objects_after_grasp
                object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                consecutive_failures = 0
                
                if len(object_ids) == 0:
                    print("   ❌ 无法生成新物体，结束场景")
                    break
                else:
                    print(f"   ✅ 成功生成 {len(object_ids)} 个新物体")
                    # 等待物体稳定
                    for _ in range(120):
                        p.stepSimulation()
                    # 重新开始这个循环迭代
                    grasp_attempt -= 1  # 不计入这次尝试
                    continue
            
            # 拍摄新照片（确保有物体后才拍照）
            width, height, view_matrix, proj_matrix = set_topdown_camera()
            rgb, depth, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
            
            # 保存当前图像
            final_rgb = rgb.copy()
            final_depth = depth.copy()
            
            if visualize:
                for i, obj_id in enumerate(object_ids):
                    try:
                        pos, _ = p.getBasePositionAndOrientation(obj_id)
                        print(f"   📦 物体{i+1} (ID={obj_id}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    except:
                        print(f"   ❌ 物体{i+1} (ID={obj_id}): 已不存在")
            
            # 基于当前图像采样候选
            candidates = sample_grasp_candidates(depth, NUM_ANGLES, visualize, rgb, view_matrix, proj_matrix, seg_mask, object_ids)
            
            if len(candidates) == 0:
                print("   ⚠️  未找到有效候选点")
                consecutive_failures += 1
                
                # ✨ 更快的重新生成触发（减少到2次失败）
                if consecutive_failures >= 2:
                    print("   🔄 连续未找到候选点，重新生成物体...")
                    from environment_setup import reset_objects_after_grasp
                    object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                    consecutive_failures = 0
                    
                    if len(object_ids) == 0:
                        print("   ❌ 无法生成新物体，结束场景")
                        break
                    else:
                        print(f"   ✅ 重新生成 {len(object_ids)} 个物体")
                        # 等待物体稳定
                        for _ in range(120):
                            p.stepSimulation()
                        # 重新开始循环
                        grasp_attempt -= 1  # 不计入这次尝试
                        continue
                else:
                    continue
            
            # 重置失败计数器
            consecutive_failures = 0
            
            # 测试第一个最有希望的候选
            u, v, theta_idx, theta = candidates[0]
            
            if depth[v, u] < MIN_DEPTH:
                print("   ⚠️  候选深度无效")
                continue
            
            total_samples += 1
            
            if visualize:
                print(f"\n      === 测试候选 ===")
                print(f"         像素: ({u}, {v}), 角度: {np.degrees(theta):.1f}°")
            
            world_pos = pixel_to_world(u, v, depth[v, u], view_matrix, proj_matrix)
            
            if visualize:
                print(f"         世界坐标: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")
            
            # 执行抓取测试
            success = fast_grasp_test(robot_id, world_pos, theta, object_ids, visualize)
            
            if success:
                total_success += 1
                print(f"      ✅ 成功抓取！")
                
                # 创建/更新标签
                if final_label is None:
                    final_label = np.zeros((height, width, NUM_ANGLES + 1), dtype=np.uint8)
                
                final_label[v, u, theta_idx] = 1
                
                # ✨ 立即更新物体列表并检查是否需要重新生成
                object_ids = update_object_states(object_ids)
                print(f"      📦 剩余物体: {len(object_ids)}")
                
                # 如果成功抓取后没有物体了，立即生成新的
                if len(object_ids) == 0:
                    print("   🎉 所有物体已被抓取，生成新物体继续训练...")
                    from environment_setup import reset_objects_after_grasp
                    object_ids = reset_objects_after_grasp([], min_objects=num_objects)
                    
                    if len(object_ids) > 0:
                        print(f"   ✅ 成功生成 {len(object_ids)} 个新物体")
                        # 等待物体稳定
                        for _ in range(120):
                            p.stepSimulation()
                    else:
                        print("   ❌ 无法生成新物体，结束场景")
                        break
                
            else:
                print(f"      ❌ 抓取失败")
                consecutive_failures += 1
            
            if grasp_attempt % 10 == 0:
                print(f"   📊 进度: {grasp_attempt} 次尝试, 成功: {total_success}, 成功率: {100*total_success/total_samples if total_samples > 0 else 0:.1f}%")
        
        # 完成最终标签
        if final_label is not None:
            # 设置背景标签
            has_success = final_label[:, :, :-1].sum(axis=2) > 0
            final_label[:, :, -1] = (~has_success).astype(np.uint8)
            
            final_rate = total_success / total_samples if total_samples > 0 else 0
            print(f"\n   ✅ 总体成功率: {final_rate*100:.1f}% ({total_success}/{total_samples})")
            
            # 保存最终数据
            save_scene_data(scene_id, final_rgb, final_depth, final_label, {
                "num_objects": num_objects,
                "num_samples": total_samples,
                "success_count": int(total_success),
                "success_rate": final_rate,
                "grasp_attempts": grasp_attempt
            })
        else:
            print(f"   ⚠️  场景 {scene_id} 没有生成任何数据")
        
        return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"   🔚 断开连接...")
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


def estimate_object_height(depth, object_mask, percentile=10):
    """估计物体表面高度
    
    使用检测到的物体像素的深度值估计表面高度
    使用较小百分位数来避免噪声和边缘效应
    
    Args:
        depth: 深度图
        object_mask: 物体mask
        percentile: 使用的百分位数（默认10 = 最近的10%像素）
    
    Returns:
        物体表面高度（世界坐标Z值）
    """
    obj_depths = depth[object_mask]
    valid_depths = obj_depths[obj_depths > MIN_DEPTH]
    
    if len(valid_depths) == 0:
        return None
    
    # 使用较小百分位数的深度值（最接近相机 = 最高点）
    surface_depth = np.percentile(valid_depths, percentile)
    
    # 深度到世界Z的转换（简化版，假设俯视相机）
    # 相机高度 = TABLE_TOP_Z + camera_distance
    # 物体Z = 相机高度 - 深度
    camera_height = TABLE_TOP_Z + 1.2  # CAMERA_DISTANCE = 1.2
    object_z = camera_height - surface_depth
    
    return object_z
