# -*- coding: utf-8 -*-
"""
自监督抓取可供性（Affordance）数据收集器
按照方案A的思路：对每一帧图像在桌面上采样若干抓取候选（位置+角度），
用物理仿真"试抓"打标签（成功/失败），生成可监督数据。
"""

import numpy as np
import cv2
import os
import json
import time
from pathlib import Path
import argp                    results.append(False)
            
            # 5. 生成可供性地图
            print("   �️  生成可供性热力图...")ort pybullet as p

# 导入现有模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment_setup import setup_environment
from src.perception import set_topdown_camera, get_rgb_depth_segmentation
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home
from src.perception import set_topdown_camera, get_rgb_depth_segmentation
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home

class AffordanceDataCollector:
    """可供性数据收集器"""
    
    def __init__(self, image_width=224, image_height=224, num_angles=8):
        """
        初始化数据收集器 - 与原始工作系统保持一致
        Args:
            image_width: 相机图像宽度 (默认224，与原始系统一致)
            image_height: 相机图像高度 (默认224，与原始系统一致) 
            num_angles: 离散化的抓取角度数量（每45度一个角度）
        """
        self.image_width = image_width
        self.image_height = image_height
        self.num_angles = num_angles
        
        # 定义离散化的抓取角度 (0度到315度，每45度一个)
        self.grasp_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        # 设置相机参数 - 与原始工作系统保持一致
        self.width, self.height = image_width, image_height
        self.fov = 60.0
        self.near = 0.1
        self.far = 2.0
        
        # 数据保存路径
        self.data_dir = Path("data/affordance_dataset")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 初始化可供性数据收集器")
        print(f"   图像尺寸: {image_width}x{image_height}")
        print(f"   抓取角度数量: {num_angles} (每{360/num_angles:.0f}度一个)")
        print(f"   数据保存路径: {self.data_dir.absolute()}")
    
    def capture_camera_data(self):
        """捕获相机数据"""
        # 设置俯视相机
        width, height, view_matrix, proj_matrix = set_topdown_camera(
            target=[0.60, 0, 0.625],  # 对准物体生成区域
            distance=1.2,
            yaw=90.0,
            pitch=-89.0,
            width=self.image_width,
            height=self.image_height
        )
        
        # 获取RGB、深度和分割图像
        rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(
            width, height, view_matrix, proj_matrix
        )
        
        # 计算相机内参矩阵（简化版本）
        fov_rad = np.radians(60.0)  # 60度视野
        f = height / (2 * np.tan(fov_rad / 2))
        camera_intrinsics = np.array([
            [f, 0, width/2],
            [0, f, height/2],
            [0, 0, 1]
        ])
        
        # 创建物体掩码（非背景区域）
        mask_image = (seg_mask > 2).astype(np.uint8)  # 跳过平面(0)、桌子(1)、机器人(2)
        
        return rgb_image, depth_image, camera_intrinsics, mask_image, view_matrix, proj_matrix, seg_mask
    
    def pixel_to_world_corrected(self, u, v, depth_value):
        """从像素坐标转换到世界坐标 - 使用原始工作系统的方法"""
        # 使用与perception.py相同的相机参数
        camera_target = [0.60, 0, 0.625]
        camera_distance = 1.2
        
        # 1. PyBullet深度缓冲区 → 实际距离（沿相机光轴）
        depth_real = self.far * self.near / (self.far - (self.far - self.near) * depth_value)
        
        # 2. 计算相机在该深度处的视野范围
        fov_rad = np.radians(self.fov)
        view_width_at_depth = 2.0 * depth_real * np.tan(fov_rad / 2.0)
        view_height_at_depth = view_width_at_depth  # 正方形视野
        
        # 3. 像素坐标归一化到 [-0.5, 0.5]
        u_norm = (u / self.width) - 0.5
        v_norm = (v / self.height) - 0.5
        
        # 4. 计算XY世界坐标
        # 相机俯视，yaw=90度（相机绕Z轴旋转90°）：
        # - 图像左右(u) 对应世界 Y轴（u小=左=后=-Y，u大=右=前=+Y）
        # - 图像上下(v) 对应世界 X轴（v小=上=左=-X，v大=下=右=+X）
        y_world = camera_target[1] + u_norm * view_width_at_depth   # u → Y
        x_world = camera_target[0] + v_norm * view_height_at_depth  # v → X
        
        # 5. 计算Z坐标
        # 相机高度 = 目标高度 + 相机距离
        camera_height = camera_target[2] + camera_distance
        z_world = camera_height - depth_real
        
        return np.array([x_world, y_world, z_world])
    
    def collect_scene_data(self, scene_id, num_objects=3, num_samples=50):
        """
        收集单个场景的可供性数据
        Args:
            scene_id: 场景ID
            num_objects: 物体数量
            num_samples: 每个场景的采样点数量
        Returns:
            bool: 是否成功收集数据
        """
        print(f"\n🔬 收集场景 {scene_id} 的可供性数据...")
        print(f"   物体数量: {num_objects}")
        print(f"   采样点数量: {num_samples}")
        
        try:
            # 1. 设置环境
            robot_id, object_ids = setup_environment(num_objects=num_objects)
            
            # 确保机器人在初始位置
            print("   🏠 重置机器人到初始位置...")
            reset_robot_home(robot_id)
            
            # 等待机器人稳定
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1./240.)
            
            # 2. 拍摄相机数据
            rgb_image, depth_image, camera_intrinsics, mask_image, view_matrix, proj_matrix, seg_mask = self.capture_camera_data()
            
            if rgb_image is None:
                print("   ❌ 相机数据获取失败")
                return False
            
            print(f"   📷 获取相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
            
            # 3. 使用原始的采样函数
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
                print("   ❌ 没有有效的抓取候选点")
                return False
            
            # 4. 测试每个候选点
            print(f"   🧪 开始测试 {len(candidates)} 个抓取候选点...")
            results = []
            success_count = 0
            
            for i, candidate in enumerate(candidates):
                if i % 10 == 0:
                    print(f"      进度: {i+1}/{len(candidates)}, 当前成功: {success_count}")
                
                # 解析候选点格式 (u, v, theta_idx, theta)
                if len(candidate) == 4:
                    u, v, theta_idx, theta = candidate
                else:
                    u, v, theta_idx = candidate
                    theta = self.grasp_angles[theta_idx]
                
                # 像素到世界坐标转换
                try:
                    world_pos = self.pixel_to_world_corrected(u, v, depth_image[v, u])
                    if world_pos is None or len(world_pos) != 3:
                        results.append(False)
                        continue
                    
                    # 简单的工作空间检查
                    dist = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
                    if dist < 0.3 or dist > 0.85 or abs(world_pos[1]) > 0.5:
                        results.append(False)
                        continue
                        
                except Exception as e:
                    results.append(False)
                    continue
                
                # 使用原始的抓取测试函数
                try:
                    success = fast_grasp_test(
                        robot_id=robot_id,
                        world_pos=world_pos,
                        grasp_angle=theta,
                        object_ids=object_ids,
                        visualize=False,
                        debug_mode=False
                    )
                    results.append(success)
                    if success:
                        success_count += 1
                        print(f"      ✅ 成功: 像素({u}, {v}) -> 世界[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}], 角度{np.degrees(theta):.1f}°")
                except Exception as e:
                    print(f"      ❌ 抓取测试失败: {e}")
                    results.append(False)
                
                # 重置机器人和物体状态（如果需要）
                if len(results) > 0 and results[-1] or np.random.random() < 0.1:  # 成功抓取后或10%概率重置
                    object_ids = reset_objects_after_grasp(object_ids, min_objects=num_objects)
            
            # 5. 生成可供性地图
            print("   �️  生成可供性热力图...")
            
            # 转换候选点格式为我们的格式
            converted_candidates = []
            for i, candidate in enumerate(candidates):
                if len(candidate) == 4:
                    u, v, theta_idx, theta = candidate
                else:
                    u, v, theta_idx = candidate
                converted_candidates.append((u, v, theta_idx))
            
            affordance_map, angle_map = self.generate_affordance_maps(converted_candidates, results)
            
            # 6. 保存数据
            self.save_scene_data(
                scene_id, rgb_image, depth_image, affordance_map, 
                angle_map, converted_candidates, results, camera_intrinsics
            )
            
            # 统计信息
            success_rate = np.mean(results)
            print(f"   ✅ 场景 {scene_id} 数据收集完成")
            print(f"      总体成功率: {success_rate:.2%}")
            print(f"      可供性热力图非零像素: {np.sum(affordance_map > 0)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 场景 {scene_id} 数据收集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_affordance_maps(self, candidates, results):
        """
        从测试结果生成可供性热力图
        Args:
            candidates: [(u, v, angle_idx), ...] 候选点列表
            results: [True/False, ...] 对应的成功/失败结果
        Returns:
            tuple: (affordance_map, angle_map)
                affordance_map: (H, W) 每个像素的最佳抓取成功概率
                angle_map: (H, W) 每个像素的最佳抓取角度索引
        """
        # 初始化地图
        affordance_map = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        angle_map = np.full((self.image_height, self.image_width), -1, dtype=np.int8)
        
        # 记录每个像素位置的所有测试结果
        pixel_results = {}  # (u, v): [(angle_idx, success), ...]
        
        for candidate, success in zip(candidates, results):
            u, v, angle_idx = candidate
            
            # 确保坐标在有效范围内
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                if (u, v) not in pixel_results:
                    pixel_results[(u, v)] = []
                
                pixel_results[(u, v)].append((angle_idx, success))
        
        # 为每个像素计算最佳抓取概率和角度
        for (u, v), angle_results in pixel_results.items():
            # 计算每个角度的成功率
            angle_success_rates = {}
            for angle_idx, success in angle_results:
                if angle_idx not in angle_success_rates:
                    angle_success_rates[angle_idx] = []
                angle_success_rates[angle_idx].append(success)
            
            # 找到成功率最高的角度
            best_angle_idx = -1
            best_success_rate = 0.0
            
            for angle_idx, successes in angle_success_rates.items():
                success_rate = np.mean(successes)
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_angle_idx = angle_idx
            
            # 记录结果
            affordance_map[v, u] = best_success_rate
            angle_map[v, u] = best_angle_idx
        
        return affordance_map, angle_map
    
    def save_scene_data(self, scene_id, rgb_image, depth_image, affordance_map, 
                       angle_map, candidates, results, camera_intrinsics):
        """保存场景数据"""
        scene_name = f"scene_{scene_id:04d}"
        
        # 保存图像数据
        cv2.imwrite(str(self.data_dir / f"{scene_name}_rgb.png"), 
                   cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        np.save(str(self.data_dir / f"{scene_name}_depth.npy"), depth_image)
        
        # 保存可供性数据
        np.save(str(self.data_dir / f"{scene_name}_affordance.npy"), affordance_map)
        np.save(str(self.data_dir / f"{scene_name}_angles.npy"), angle_map)
        
        # 保存元数据
        metadata = {
            "scene_id": int(scene_id),
            "image_size": [int(self.image_width), int(self.image_height)],
            "num_angles": int(self.num_angles),
            "grasp_angles_rad": [float(x) for x in self.grasp_angles.tolist()],
            "camera_intrinsics": [[float(x) for x in row] for row in camera_intrinsics.tolist()],
            "num_candidates": int(len(candidates)),
            "success_rate": float(np.mean(results)) if results else 0.0,
            "candidates": [[int(u), int(v), int(angle_idx)] for u, v, angle_idx in candidates],  # 确保是基本类型
            "results": [bool(r) for r in results],  # 确保是基本类型
        }
        
        with open(self.data_dir / f"{scene_name}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 场景数据已保存到 {self.data_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="收集自监督抓取可供性数据")
    parser.add_argument("--num_scenes", type=int, default=100, 
                       help="要收集的场景数量")
    parser.add_argument("--num_objects", type=int, nargs=2, default=[3, 5],
                       help="每个场景的物体数量范围")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="每个场景的抓取候选点数量")
    parser.add_argument("--num_angles", type=int, default=8,
                       help="离散化抓取角度数量")
    parser.add_argument("--visualize", action="store_true",
                       help="启用可视化（仅第一个场景）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 自监督抓取可供性数据收集器")
    print("=" * 60)
    print(f"场景数量: {args.num_scenes}")
    print(f"物体数量范围: {args.num_objects[0]}-{args.num_objects[1]}")
    print(f"每场景采样数: {args.num_samples}")
    print(f"抓取角度数: {args.num_angles}")
    print("=" * 60)
    
    # 初始化PyBullet
    if args.visualize:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    
    # 创建数据收集器
    collector = AffordanceDataCollector(num_angles=args.num_angles)
    
    try:
        # 收集数据
        successful_scenes = 0
        
        for scene_id in range(args.num_scenes):
            # 随机选择物体数量
            num_objects = np.random.randint(args.num_objects[0], args.num_objects[1] + 1)
            
            # 收集场景数据
            if collector.collect_scene_data(scene_id, num_objects, args.num_samples):
                successful_scenes += 1
            
            print(f"进度: {scene_id + 1}/{args.num_scenes} 场景, "
                  f"成功: {successful_scenes}")
        
        print("\n" + "=" * 60)
        print(f"✅ 数据收集完成!")
        print(f"成功收集场景: {successful_scenes}/{args.num_scenes}")
        print(f"数据保存路径: {collector.data_dir.absolute()}")
        print("=" * 60)
        
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()