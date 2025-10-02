#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化可供性训练数据收集器
Automatic Affordance Training Data Collector

基于已验证的工作系统，自动收集训练数据
"""

import numpy as np
import cv2
import os
import json
import time
import argparse
import pybullet as p
from PIL import Image
from pathlib import Path
import sys

# 导入工作的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.environment_setup import setup_environment
from src.perception import set_topdown_camera, get_rgb_depth_segmentation, pixel_to_world
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home

class AutoAffordanceCollector:
    """自动化可供性数据收集器"""
    
    def __init__(self, data_dir="data/affordance_v5", num_angles=8, train_split=0.8):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练和测试子目录
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
        self.num_angles = num_angles
        self.train_split = train_split  # 训练集比例 (0.8 = 80%)
        
        # 抓取角度设置
        self.grasp_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        print(f"🎯 自动化可供性数据收集器初始化")
        print(f"   数据目录: {self.data_dir.absolute()}")
        print(f"   训练目录: {self.train_dir}")
        print(f"   测试目录: {self.test_dir}")
        print(f"   训练集比例: {train_split:.1%}")
        print(f"   抓取角度数: {num_angles}")
    
    def collect_single_attempt_as_scene(self, scene_idx, robot_id, object_ids, target_dir):
        """收集单次抓取尝试作为独立场景，每次抓取前恢复环境到初始状态"""
        print(f"============================================================")
        
        # 1. 记录初始机器人和物体状态
        home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        initial_robot_joints = [p.getJointState(robot_id, i)[0] for i in range(7)]
        initial_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
        object_states = {}
        for obj_id in object_ids:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            object_states[obj_id] = (pos, orn)
        
        # 2. 确保机器人在初始位置
        print("   🏠 重置机器人...")
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                targetPosition=home_joints[i], 
                force=500, 
                maxVelocity=2.0
            )
        for _ in range(200):
            p.stepSimulation()
            all_in_position = True
            for i in range(7):
                current = p.getJointState(robot_id, i)[0]
                if abs(current - home_joints[i]) > 0.05:
                    all_in_position = False
                    break
            if all_in_position:
                break
        # 强制打开夹爪
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.02, force=300)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.02, force=300)
        for _ in range(40):
            p.stepSimulation()
        # 验证机器人位置
        current_pos = p.getLinkState(robot_id, 8)[0]
        print(f"   📍 机器人末端位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        # 拍摄照片
        print("   📷 拍摄场景照片...")
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
        print(f"   📷 相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
        # 采样抓取候选点
        candidates = sample_grasp_candidates(
            depth=depth_image,
            num_angles=self.num_angles,
            visualize=False,
            rgb=rgb_image,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            seg_mask=seg_mask,
            object_ids=object_ids
        )
        
        print(f"   🔍 候选点采样结果: {len(candidates)} 个候选点")
        
        if not candidates:
            print("   ❌ 没有找到有效的抓取候选点")
            return False
        
        print(f"   📍 采样了 {len(candidates)} 个候选点 - 测试前 {min(20, len(candidates))} 个")
        
        # 测试多个候选点（每个场景最多20个，提供更密集的训练标签）
        test_count = min(20, len(candidates))
        successful_grasps = []
        failed_grasps = []
        
        for i, candidate in enumerate(candidates[:test_count]):
            if len(candidate) == 4:
                u, v, theta_idx, theta = candidate
            else:
                u, v, theta_idx = candidate
                theta = self.grasp_angles[theta_idx]
            world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
            print(f"   🎯 测试 {i+1}/{test_count}: 像素({u},{v}), 角度{np.degrees(theta):.1f}°")
            # 每次抓取前恢复机器人和物体到初始状态
            if i > 0:
                print(f"      🏠 恢复机器人和物体到初始状态...")
                # 恢复机器人关节
                for j in range(7):
                    p.resetJointState(robot_id, j, home_joints[j])
                # 恢复夹爪
                p.resetJointState(robot_id, 9, 0.02)
                p.resetJointState(robot_id, 10, 0.02)
                # 恢复物体
                for obj_id in object_ids:
                    pos, orn = object_states[obj_id]
                    p.resetBasePositionAndOrientation(obj_id, pos, orn)
                # 让物理引擎稳定
                for _ in range(10):
                    p.stepSimulation()
            # 执行抓取测试
            success = fast_grasp_test(robot_id, world_pos, theta, object_ids, visualize=False)
            if success:
                print(f"      ✅ 成功!")
                successful_grasps.append({'pixel': (u, v), 'angle': theta, 'world_pos': world_pos})
            else:
                print(f"      ❌ 失败")
                failed_grasps.append({'pixel': (u, v), 'angle': theta, 'world_pos': world_pos})
        
        # 计算成功率
        success_rate = len(successful_grasps) / test_count if test_count > 0 else 0
        print(f"   📊 场景成功率: {len(successful_grasps)}/{test_count} ({success_rate:.1%})")
        
        # 生成可供性标签（标记所有测试的点）
        affordance_map = np.zeros((224, 224), dtype=np.float32)
        angle_map = np.zeros((224, 224), dtype=np.float32) 
        
        # 标记成功的抓取点
        for grasp in successful_grasps:
            u, v = grasp['pixel']
            theta = grasp['angle']
            affordance_map[v, u] = 1.0  # 成功点标记为1
            angle_map[v, u] = theta
        
        # 失败的点保持为0（已经初始化为0）
        
        # 保存数据
        data_dir = target_dir
        os.makedirs(data_dir, exist_ok=True)
        
        scene_prefix = f"scene_{scene_idx:04d}"
        
        # 保存图像
        rgb_path = os.path.join(data_dir, f"{scene_prefix}_rgb.png")
        Image.fromarray(rgb_image).save(rgb_path)
        
        # 保存深度图
        depth_path = os.path.join(data_dir, f"{scene_prefix}_depth.npy")
        np.save(depth_path, depth_image)
        
        # 保存可供性图
        affordance_path = os.path.join(data_dir, f"{scene_prefix}_affordance.npy")
        np.save(affordance_path, affordance_map)
        
        # 保存角度图
        angle_path = os.path.join(data_dir, f"{scene_prefix}_angles.npy")
        np.save(angle_path, angle_map)
        
        # 保存元数据（包含所有抓取尝试的详细信息）
        meta_data = {
            "scene_id": scene_idx,
            "total_attempts": test_count,
            "successful_grasps": len(successful_grasps),
            "failed_grasps": len(failed_grasps),
            "success_rate": success_rate,
            "image_size": [int(rgb_image.shape[1]), int(rgb_image.shape[0])],
            "num_objects": len(object_ids),
            "object_ids": [int(oid) for oid in object_ids],
            "grasp_details": {
                "successful": [
                    {
                        "pixel": [int(g['pixel'][0]), int(g['pixel'][1])],
                        "world_pos": [float(g['world_pos'][0]), float(g['world_pos'][1]), float(g['world_pos'][2])],
                        "angle_degrees": float(np.degrees(g['angle']))
                    } for g in successful_grasps
                ],
                "failed": [
                    {
                        "pixel": [int(g['pixel'][0]), int(g['pixel'][1])],
                        "world_pos": [float(g['world_pos'][0]), float(g['world_pos'][1]), float(g['world_pos'][2])],
                        "angle_degrees": float(np.degrees(g['angle']))
                    } for g in failed_grasps
                ]
            }
        }
        
        meta_path = os.path.join(data_dir, f"{scene_prefix}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"      💾 数据已保存: {scene_prefix}_*")
        
        # 返回是否有任何成功的抓取
        return len(successful_grasps) > 0
        """收集单个场景数据 - 修复为正确的场景定义
        
        正确的场景定义：
        1. 一个场景 = 一张照片 + 多次抓取尝试
        2. 机器人在每次抓取前回到初始位置
        3. 照片只拍一次（机器人在初始位置时）
        4. 多个候选点在同一场景中测试
        """
        try:
            # 1. 设置环境
            robot_id, object_ids = setup_environment(num_objects=num_objects)
            if not object_ids:
                return False
            
            # 2. 等待物体稳定
            for _ in range(120):
                p.stepSimulation()
            
                        # 3. 确保机器人在初始位置 - 使用位置控制
            print("   🏠 重置机器人...")
            
            home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            
            # 使用位置控制而不是直接设置关节状态，更平滑
            for i in range(7):
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL,
                    targetPosition=home_joints[i], 
                    force=500, 
                    maxVelocity=2.0
                )
            
            # 等待到位
            for _ in range(200):  # 增加等待时间
                p.stepSimulation()
                
                # 检查是否到位
                all_in_position = True
                for i in range(7):
                    current = p.getJointState(robot_id, i)[0]
                    if abs(current - home_joints[i]) > 0.05:  # 容差3度
                        all_in_position = False
                        break
                
                if all_in_position:
                    break
            
            # 强制打开夹爪
            p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.02, force=300)
            p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.02, force=300)
            for _ in range(40):
                p.stepSimulation()
            
            # 验证机器人是否真的到了初始位置
            ee_link = 11
            current_pos = p.getLinkState(robot_id, ee_link)[0]
            print(f"   📍 机器人末端位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            
            # 4. 拍摄照片（机器人在初始位置时）
            print("   📷 拍摄场景照片...")
            width, height, view_matrix, proj_matrix = set_topdown_camera()
            rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
            print(f"   📷 相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
            
            # 5. 采样抓取候选点
            candidates = sample_grasp_candidates(
                depth=depth_image,
                num_angles=self.num_angles,
                visualize=False,
                rgb=rgb_image,
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
                seg_mask=seg_mask,
                object_ids=object_ids
            )
            
            print(f"   🔍 候选点采样结果: {len(candidates)} 个候选点")
            
            if not candidates:
                print("   ❌ 没有候选点")
                return False
            
            # 6. 测试多个候选点 - 这是关键修复
            results = []
            success_count = 0
            test_count = min(max_attempts_per_scene, len(candidates))  # 最多测试指定数量
            
            print(f"   🎯 测试 {test_count} 个候选点...")
            
            for i, candidate in enumerate(candidates[:test_count]):
                if len(candidate) == 4:
                    u, v, theta_idx, theta = candidate
                else:
                    u, v, theta_idx = candidate
                    theta = self.grasp_angles[theta_idx]
                
                world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
                
                print(f"      🎯 测试 {i+1}/{test_count}: 像素({u},{v}), 角度{np.degrees(theta):.1f}°")
                
                # 🔑 关键：每次抓取前确保机器人在初始位置
                if i > 0:  # 第一次不需要重置，已经在初始位置
                    print(f"      🏠 重置机器人位置...")
                    end_pos_before = p.getLinkState(robot_id, 8)[0]
                    print(f"      📍 当前末端位置: [{end_pos_before[0]:.3f}, {end_pos_before[1]:.3f}, {end_pos_before[2]:.3f}]")
                    
                    # 使用位置控制重置
                    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
                    for j in range(7):
                        p.setJointMotorControl2(
                            robot_id, j, p.POSITION_CONTROL,
                            targetPosition=home[j], 
                            force=500, 
                            maxVelocity=2.0
                        )
                    
                    # 等待到位
                    for _ in range(150):
                        p.stepSimulation()
                        
                        # 检查是否到位
                        all_in_position = True
                        for j in range(7):
                            current = p.getJointState(robot_id, j)[0]
                            if abs(current - home[j]) > 0.05:
                                all_in_position = False
                                break
                        
                        if all_in_position:
                            break
                    
                    # 强制打开夹爪
                    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.02, force=300)
                    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.02, force=300)
                    
                    # 等待机器人完全稳定
                    for _ in range(150):
                        p.stepSimulation()
                        time.sleep(1./240.)
                    
                    # 验证机器人是否真的到了初始位置
                    ee_link = 11
                    current_pos = p.getLinkState(robot_id, ee_link)[0]
                    print(f"      📍 当前末端位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                    
                    # 调试: 打印物体位置
                    print(f"      🔍 重置后物体位置:")
                    for j, obj_id in enumerate(object_ids):
                        pos, orn = p.getBasePositionAndOrientation(obj_id)
                        print(f"         物体 {obj_id}: 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
                # 测试抓取
                try:
                    success = fast_grasp_test(
                        robot_id=robot_id,
                        world_pos=world_pos,
                        grasp_angle=theta,
                        object_ids=object_ids,
                        visualize=False,
                        debug_mode=False
                    )
                    
                    if success:
                        success_count += 1
                        print(f"      ✅ 成功!")
                    else:
                        print(f"      ❌ 失败")
                    
                    # 记录结果
                    results.append({
                        'pixel': [int(u), int(v)],
                        'world_pos': [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])],
                        'angle': float(theta),
                        'angle_idx': int(theta_idx),
                        'success': success
                    })
                    
                except Exception as e:
                    print(f"      ❌ 错误: {e}")
                    results.append({
                        'pixel': [int(u), int(v)],
                        'world_pos': [0, 0, 0],
                        'angle': float(theta),
                        'angle_idx': int(theta_idx),
                        'success': False
                    })
            
            # 7. 计算成功率并保存数据
            success_rate = (success_count / test_count) * 100 if test_count > 0 else 0
            print(f"   📊 成功率: {success_count}/{test_count} ({success_rate:.1f}%)")
            
            # 8. 生成并保存场景数据
            affordance_map = self.create_affordance_map(rgb_image.shape[:2], results)
            angle_map = self.create_angle_map(rgb_image.shape[:2], results)
            
            self.save_scene_data(scene_id, rgb_image, depth_image, affordance_map, angle_map, results)
            print(f"      💾 数据已保存: scene_{scene_id:04d}_*")
            print(f"   ✅ 场景 {scene_id} 完成 (成功率: {success_rate:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 场景错误: {e}")
            return False
    
    def is_position_reachable(self, world_pos):
        """检查位置是否在机器人工作空间内"""
        x, y, z = world_pos
        
        # 基本工作空间限制
        distance = np.sqrt(x**2 + y**2)
        
        # Franka Panda的工作空间约束
        if distance < 0.3 or distance > 0.85:  # 距离限制
            return False
        if abs(y) > 0.4:  # Y轴限制
            return False
        if z < 0.58 or z > 0.8:  # Z轴高度限制
            return False
        if x < 0.2 or x > 0.9:  # X轴前后限制
            return False
            
        return True
    
    def is_position_reachable(self, world_pos):
        """检查位置是否在机器人工作空间内"""
        x, y, z = world_pos
        
        # 基本工作空间限制
        distance = np.sqrt(x**2 + y**2)
        
        # Franka Panda的工作空间约束
        if distance < 0.3 or distance > 0.85:  # 距离限制
            return False
        if abs(y) > 0.4:  # Y轴限制
            return False
        if z < 0.58 or z > 0.8:  # Z轴高度限制
            return False
        if x < 0.2 or x > 0.9:  # X轴前后限制
            return False
            
        return True
    
    def generate_affordance_map(self, results, image_shape):
        """生成可供性热力图"""
        affordance_map = np.zeros(image_shape, dtype=np.float32)
        
        for result in results:
            if result['world_pos'][0] != 0:  # 有效的世界坐标
                u, v = result['pixel']
                if 0 <= v < image_shape[0] and 0 <= u < image_shape[1]:
                    # 成功为1.0，失败为0.0
                    affordance_map[v, u] = 1.0 if result['success'] else 0.0
        
        # 轻微高斯模糊来平滑热力图
        affordance_map = cv2.GaussianBlur(affordance_map, (5, 5), 1.0)
        return affordance_map
    
    def generate_angle_map(self, results, image_shape):
        """生成最佳抓取角度地图"""
        angle_map = np.zeros(image_shape, dtype=np.float32)
        
        for result in results:
            if result['success'] and result['world_pos'][0] != 0:
                u, v = result['pixel']
                if 0 <= v < image_shape[0] and 0 <= u < image_shape[1]:
                    # 归一化角度到 [0, 1]
                    normalized_angle = result['angle'] / (2 * np.pi)
                    angle_map[v, u] = normalized_angle
        
        return angle_map
    
    def create_affordance_map(self, image_shape, results):
        """创建可供性地图"""
        height, width = image_shape
        affordance_map = np.zeros((height, width), dtype=np.float32)
        
        for result in results:
            if 'pixel' in result and len(result['pixel']) == 2:
                u, v = result['pixel']
                if 0 <= v < height and 0 <= u < width:
                    affordance_map[v, u] = 1.0 if result['success'] else 0.0
        
        return affordance_map
    
    def create_angle_map(self, image_shape, results):
        """创建角度地图"""
        height, width = image_shape
        angle_map = np.zeros((height, width), dtype=np.float32)
        
        for result in results:
            if result['success'] and result['world_pos'][0] != 0:
                u, v = result['pixel']
                if 0 <= v < height and 0 <= u < width:
                    angle_map[v, u] = result['angle']
        
        return angle_map
    
    def save_scene_data(self, scene_id, rgb_image, depth_image, affordance_map, angle_map, results):
        """保存场景数据"""
        scene_prefix = f"scene_{scene_id:04d}"
        
        # 保存图像
        rgb_path = self.data_dir / f"{scene_prefix}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # 保存深度图
        depth_path = self.data_dir / f"{scene_prefix}_depth.npy"
        np.save(depth_path, depth_image)
        
        # 保存可供性地图
        affordance_path = self.data_dir / f"{scene_prefix}_affordance.npy"
        np.save(affordance_path, affordance_map)
        
        # 保存角度地图
        angle_path = self.data_dir / f"{scene_prefix}_angles.npy"
        np.save(angle_path, angle_map)
        
        # 保存元数据
        metadata = {
            'scene_id': scene_id,
            'image_shape': rgb_image.shape[:2],
            'num_candidates': len(results),
            'num_successful': sum(1 for r in results if r['success']),
            'success_rate': sum(1 for r in results if r['success']) / len(results) if results else 0,
            'candidates': results
        }
        
        meta_path = self.data_dir / f"{scene_prefix}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      💾 数据已保存: {scene_prefix}_*")
    
    def collect_dataset(self, num_scenes, num_objects_range=(2, 4), max_attempts_per_scene=25):
        """收集完整数据集"""
        print(f"\n🚀 开始收集数据集")
        print(f"   场景数量: {num_scenes}")
        print(f"   物体数量范围: {num_objects_range[0]}-{num_objects_range[1]}")
        print(f"   每场景最大尝试数: {max_attempts_per_scene}")
        print("=" * 60)
        
        successful_scenes = 0
        total_grasps = 0
        total_successful_grasps = 0
        
        for scene_id in range(num_scenes):
            # 每5个场景创建一个新的环境配置
            if scene_id % 5 == 0:
                # 清理之前的环境
                if scene_id > 0:
                    for obj_id in object_ids:
                        try:
                            p.removeBody(obj_id)
                        except:
                            pass
                
                # 随机选择物体数量并创建新环境
                num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
                
                # ✨ 确保随机性：为每个新环境设置不同的随机种子
                np.random.seed(scene_id + int(time.time()) % 1000)
                
                print(f"🏗️  创建新环境配置 #{(scene_id // 5) + 1}: {num_objects} 个物体")
                robot_id, object_ids = setup_environment(num_objects=num_objects)
                if not object_ids:
                    print(f"   ❌ 环境设置失败，跳过场景 {scene_id}")
                    continue
                
                # 调试: 打印物体位置和机器人姿态
                print(f"🔍 场景 {scene_id} 环境设置调试:")
                robot_pos = p.getLinkState(robot_id, 8)[0]
                print(f"  📍 机器人末端位置: {robot_pos}")
                print(f"  📍 物体位置:")
                for i, obj_id in enumerate(object_ids):
                    pos, orn = p.getBasePositionAndOrientation(obj_id)
                    print(f"    物体 {i}: 位置={pos}, 朝向={orn}")
            
            # 收集单次抓取尝试作为独立场景
            print(f"   🔍 场景 {scene_id} 开始 - 物体IDs: {object_ids}")
            
            # 随机决定该场景属于训练集还是测试集
            is_train = np.random.random() < self.train_split
            target_dir = self.train_dir if is_train else self.test_dir
            split_name = "训练集" if is_train else "测试集"
            print(f"   📊 场景分配: {split_name} ({target_dir.name})")
            
            success = self.collect_single_attempt_as_scene(scene_id, robot_id, object_ids, target_dir)
            
            # 读取场景统计信息
            import json
            meta_file = os.path.join(target_dir, f"scene_{scene_id:04d}_meta.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    scene_attempts = meta.get('total_attempts', 0)
                    scene_successes = meta.get('successful_grasps', 0)
                    total_grasps += scene_attempts
                    total_successful_grasps += scene_successes
            
            # 场景间短暂停顿，让物理引擎稳定
            if scene_id < num_scenes - 1:
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(1./240.)
            
            if success:
                successful_scenes += 1
                print(f"   ✅ 场景 {scene_id} 完成 (有可用数据)")
            else:
                print(f"   ❌ 场景 {scene_id} 完成 (无可用数据)")
        
        # 总结
        grasp_success_rate = (total_successful_grasps / total_grasps) if total_grasps > 0 else 0
        print("=" * 60)
        print(f"🎉 数据收集完成!")
        print(f"   有效场景: {successful_scenes}/{num_scenes}")
        print(f"   总抓取成功率: {grasp_success_rate:.1%} ({total_successful_grasps}/{total_grasps})")
        print(f"   数据保存位置:")
        print(f"     训练集: {self.train_dir} ({self.train_split:.1%})")
        print(f"     测试集: {self.test_dir} ({1-self.train_split:.1%})")
        print("✅ 数据收集成功!")
        
        return successful_scenes > 0

def main():
    parser = argparse.ArgumentParser(description='自动化可供性训练数据收集器')
    parser.add_argument('--num_scenes', type=int, default=10, help='收集的场景数量')
    parser.add_argument('--num_objects', type=int, nargs=2, default=[2, 4], help='物体数量范围 [min, max]')
    parser.add_argument('--max_attempts', type=int, default=25, help='每场景最大抓取尝试数')
    parser.add_argument('--data_dir', type=str, default='data/affordance_v5', help='数据保存目录')
    parser.add_argument('--angles', type=int, default=8, help='离散抓取角度数量')
    parser.add_argument('--train_split', type=float, default=0.8, help='训练集比例 (0.0-1.0)')
    parser.add_argument('--gui', action='store_true', help='显示GUI')
    
    args = parser.parse_args()
    
    # 连接PyBullet
    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    
    try:
        # 创建收集器
        collector = AutoAffordanceCollector(
            data_dir=args.data_dir,
            num_angles=args.angles,
            train_split=args.train_split
        )
        
        # 收集数据
        success = collector.collect_dataset(
            num_scenes=args.num_scenes,
            num_objects_range=tuple(args.num_objects),
            max_attempts_per_scene=args.max_attempts
        )
        
        if success:
            print("✅ 数据收集成功!")
        else:
            print("❌ 数据收集失败!")
    
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()