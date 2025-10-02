#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient Affordance Data Generator
Each grasp attempt = One complete scene with data saving
"""

import sys
import time
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import pybullet as p

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment_setup import setup_environment
from perception import set_topdown_camera, get_rgb_depth_segmentation, sample_grasp_candidates
from afford_data_gen import fast_grasp_test, reset_robot_home

# Data directory
DATA_DIR = Path(__file__).parent / "data" / "efficient_affordance"

def create_data_dirs():
    """Create data directories"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 数据目录: {DATA_DIR.absolute()}")

def save_scene_data(scene_id, rgb, depth, success_map, metadata):
    """保存单个场景数据"""
    prefix = DATA_DIR / f"scene_{scene_id:04d}"
    
    # Save RGB image
    cv2.imwrite(str(prefix) + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Save depth map
    np.save(str(prefix) + "_depth.npy", depth)
    
    # Save affordance/success map
    np.save(str(prefix) + "_affordance.npy", success_map)
    
    # Save metadata
    with open(str(prefix) + "_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   💾 已保存: scene_{scene_id:04d}_*")

def generate_affordance_map(candidates, success_results, image_shape):
    """Generate affordance heatmap from grasp results"""
    affordance_map = np.zeros(image_shape, dtype=np.float32)
    
    for i, (u, v, theta_idx) in enumerate(candidates):
        if i < len(success_results):
            success = success_results[i]
            # Simple approach: mark pixel as 1.0 if grasp succeeded, 0.0 if failed
            affordance_map[v, u] = 1.0 if success else 0.0
    
    return affordance_map

def generate_single_scene(scene_id, num_objects=3):
    """Generate one complete scene with ONE grasp attempt"""
    print(f"\n🎬 场景 {scene_id:04d} (物体数: {num_objects})")
    
    try:
        # 1. Setup fresh environment
        robot_id, object_ids = setup_environment(num_objects=num_objects)
        if not object_ids:
            print(f"   ❌ 场景 {scene_id} 失败: 无法创建物体")
            return False
        
        # 2. Reset robot to home position
        reset_robot_home(robot_id)
        
        # Wait for stability
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 3. Capture camera data
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        rgb_image, depth_image, seg_mask = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
        
        print(f"   📷 相机数据: RGB {rgb_image.shape}, 深度 {depth_image.shape}")
        
        # 4. Sample grasp candidates
        candidates = sample_grasp_candidates(
            depth=depth_image,
            num_angles=8,
            visualize=False,
            rgb=rgb_image,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            seg_mask=seg_mask,
            object_ids=object_ids
        )
        
        if not candidates:
            print(f"   ❌ 场景 {scene_id} 失败: 无候选点")
            return False
        
        print(f"   🎯 测试 {len(candidates)} 个候选点")
        
        # 5. Test ONE grasp attempt (or a few)
        max_attempts = min(3, len(candidates))  # Test max 3 candidates
        success_results = []
        tested_candidates = []
        
        for i in range(max_attempts):
            candidate = candidates[i]
            u, v, theta_idx = candidate[:3]
            theta = candidate[3] if len(candidate) > 3 else np.linspace(0, 2*np.pi, 8)[theta_idx]
            
            # Convert pixel to world coordinates
            from perception import pixel_to_world
            world_pos = pixel_to_world(u, v, depth_image[v, u], view_matrix, proj_matrix)
            
            print(f"      🎯 测试 {i+1}/{max_attempts}: 像素({u},{v}), 角度{np.degrees(theta):.1f}°")
            
            # Test grasp
            success = fast_grasp_test(
                robot_id=robot_id,
                world_pos=world_pos,
                grasp_angle=theta,
                object_ids=object_ids,
                visualize=False,
                debug_mode=False
            )
            
            success_results.append(success)
            tested_candidates.append(candidate)
            
            if success:
                print(f"      ✅ 成功!")
                break  # Stop after first success
            else:
                print(f"      ❌ 失败")
        
        # 6. Generate affordance map
        affordance_map = generate_affordance_map(tested_candidates, success_results, rgb_image.shape[:2])
        
        # 7. Create metadata
        success_count = sum(success_results)
        success_rate = success_count / len(success_results) if success_results else 0.0
        
        metadata = {
            'scene_id': scene_id,
            'num_objects': len(object_ids),
            'num_candidates_tested': len(success_results),
            'num_successful': success_count,
            'success_rate': success_rate,
            'image_shape': rgb_image.shape[:2],
            'candidates': [
                {
                    'pixel': [int(c[0]), int(c[1])],
                    'angle_idx': int(c[2]),
                    'success': bool(success_results[i]) if i < len(success_results) else False
                }
                for i, c in enumerate(tested_candidates)
            ]
        }
        
        # 8. Save data
        save_scene_data(scene_id, rgb_image, depth_image, affordance_map, metadata)
        
        print(f"   ✅ 场景 {scene_id} 完成 (成功率: {success_rate:.1%})")
        return True
        
    except Exception as e:
        print(f"   ❌ 场景 {scene_id} 失败: {e}")
        return False

def generate_dataset(num_scenes, num_objects_range=(2, 4)):
    """Generate complete dataset"""
    print("=" * 60)
    print("🚀 高效数据生成 - 每次抓取 = 一个场景")
    print(f"   场景数量: {num_scenes}")
    print(f"   物体数量: {num_objects_range[0]}-{num_objects_range[1]}")
    print("=" * 60)
    
    create_data_dirs()
    
    successful_scenes = 0
    total_success_rate = 0
    
    for scene_id in range(num_scenes):
        # Random number of objects
        num_objects = np.random.randint(num_objects_range[0], num_objects_range[1] + 1)
        
        # Generate scene
        success = generate_single_scene(scene_id, num_objects)
        
        if success:
            successful_scenes += 1
            
            # Read success rate from metadata
            try:
                meta_path = DATA_DIR / f"scene_{scene_id:04d}_meta.json"
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                total_success_rate += metadata['success_rate']
            except:
                pass
    
    # Summary
    avg_success_rate = (total_success_rate / successful_scenes) if successful_scenes > 0 else 0
    print("=" * 60)
    print(f"🎉 数据生成完成!")
    print(f"   成功场景: {successful_scenes}/{num_scenes}")
    print(f"   平均成功率: {avg_success_rate:.1%}")
    print(f"   数据位置: {DATA_DIR.absolute()}")
    
    return successful_scenes > 0

def main():
    parser = argparse.ArgumentParser(description='高效可供性数据生成器')
    parser.add_argument('--num_scenes', type=int, default=50, help='场景数量')
    parser.add_argument('--num_objects', type=int, nargs=2, default=[2, 4], help='物体数量范围')
    parser.add_argument('--gui', action='store_true', help='显示GUI')
    
    args = parser.parse_args()
    
    # Connect to PyBullet
    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    
    try:
        success = generate_dataset(args.num_scenes, tuple(args.num_objects))
        
        if success:
            print("\n✅ 数据生成成功!")
        else:
            print("\n❌ 数据生成失败!")
            sys.exit(1)
            
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()