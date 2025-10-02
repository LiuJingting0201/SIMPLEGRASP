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
from pathlib import Path
import sys

# 导入工作的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.environment_setup import setup_environment
from src.perception import set_topdown_camera, get_rgb_depth_segmentation, pixel_to_world
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home

class AutoAffordanceCollector:
    """自动化可供性数据收集器"""
    
    def __init__(self, data_dir="data/affordance_v5", num_angles=8):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.num_angles = num_angles
        
        # 抓取角度设置
        self.grasp_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        print(f"🎯 自动化可供性数据收集器初始化")
        print(f"   数据目录: {self.data_dir.absolute()}")
        print(f"   抓取角度数: {num_angles}")
    
    def collect_single_scene(self, scene_id, num_objects=3, max_attempts=30):
        """收集单个场景的数据 - 每个抓取尝试就是一个完整场景"""
        print(f"\n🎬 收集场景 {scene_id:04d} (物体数: {num_objects})")
        
        try:
            # 1. 设置全新环境
            robot_id, object_ids = setup_environment(num_objects=num_objects)
            if not object_ids:
                print("   ❌ 环境设置失败")
                return False
            
            # 2. 重置机器人
            print("   🏠 重置机器人...")
            reset_robot_home(robot_id)
            
            # 等待稳定
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1./240.)
            
            # 3. 拍摄相机数据
            width, height, view_matrix, proj_matrix = set_topdown_camera()
            rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
            
            print(f"   📷 相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
            
            # 4. 采样候选点
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
            
            if not candidates:
                print("   ❌ 没有候选点")
                return False
            
            print(f"   🎯 测试 {min(3, len(candidates))} 个候选点...")
            
            # 5. 只测试少量候选点 (1-3个) - 每个场景快速完成
            results = []
            success_count = 0
            test_count = min(3, len(candidates))  # 最多测试3个
            
            for i, candidate in enumerate(candidates[:test_count]):
                if len(candidate) == 4:
                    u, v, theta_idx, theta = candidate
                else:
                    u, v, theta_idx = candidate
                    theta = self.grasp_angles[theta_idx]
                
                world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
                
                print(f"      🎯 测试 {i+1}/{test_count}: 像素({u},{v}), 角度{np.degrees(theta):.1f}°")
                
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
                        
                        # 记录结果并立即结束场景 (找到成功就够了)
                        results.append({
                            'pixel': [int(u), int(v)],
                            'world_pos': [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])],
                            'angle': float(theta),
                            'angle_idx': int(theta_idx),
                            'success': True
                        })
                        break  # 成功就结束这个场景
                    else:
                        print(f"      ❌ 失败")
                        results.append({
                            'pixel': [int(u), int(v)],
                            'world_pos': [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])],
                            'angle': float(theta),
                            'angle_idx': int(theta_idx),
                            'success': False
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
            
            success_rate = (success_count / test_count) * 100 if test_count > 0 else 0
            print(f"   📊 成功率: {success_count}/{test_count} ({success_rate:.1f}%)")
            
            # 6. 生成可供性地图
            affordance_map = self.generate_affordance_map(results, rgb_image.shape[:2])
            angle_map = self.generate_angle_map(results, rgb_image.shape[:2])
            
            # 7. 保存数据
            self.save_scene_data(scene_id, rgb_image, depth_image, affordance_map, angle_map, results)
            
            print(f"   ✅ 场景 {scene_id} 完成 (成功率: {success_rate:.1f}%)")
            return True
            
        except Exception as e:
            print(f"   ❌ 场景 {scene_id} 失败: {e}")
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
        total_success_rate = 0
        avg_success_rate = 0  # Initialize to avoid UnboundLocalError
        
        for scene_id in range(num_scenes):
            # 随机选择物体数量
            num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
            
            # 收集场景
            success = self.collect_single_scene(scene_id, num_objects, max_attempts_per_scene)
            
            # 场景间短暂停顿，让物理引擎稳定
            if scene_id < num_scenes - 1:
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(1./240.)
            
            if success:
                successful_scenes += 1
                # 读取成功率
                try:
                    meta_path = self.data_dir / f"scene_{scene_id:04d}_meta.json"
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    total_success_rate += metadata['success_rate']
                except:
                    pass
            else:
                print(f"   ⚠️  场景 {scene_id} 收集失败，跳过")
        
        # 总结
        avg_success_rate = (total_success_rate / successful_scenes) if successful_scenes > 0 else 0
        print("=" * 60)
        print(f"🎉 数据收集完成!")
        print(f"   成功场景: {successful_scenes}/{num_scenes}")
        print(f"   平均成功率: {avg_success_rate:.1f}%")
        print(f"   数据保存在: {self.data_dir.absolute()}")
        print("✅ 数据收集成功!")
        
        return successful_scenes > 0

def main():
    parser = argparse.ArgumentParser(description='自动化可供性训练数据收集器')
    parser.add_argument('--num_scenes', type=int, default=10, help='收集的场景数量')
    parser.add_argument('--num_objects', type=int, nargs=2, default=[2, 4], help='物体数量范围 [min, max]')
    parser.add_argument('--max_attempts', type=int, default=25, help='每场景最大抓取尝试数')
    parser.add_argument('--data_dir', type=str, default='data/affordance_v5', help='数据保存目录')
    parser.add_argument('--angles', type=int, default=8, help='离散抓取角度数量')
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
            num_angles=args.angles
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