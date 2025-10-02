#!/usr/bin/env python3
"""
Simple test of the working affordance system
"""

import numpy as np
import pybullet as p
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment_setup import setup_environment
from src.perception import set_topdown_camera, get_rgb_depth_segmentation
from src.afford_data_gen import sample_grasp_candidates, fast_grasp_test, reset_robot_home

def test_affordance_system():
    """Test the basic affordance system functionality"""
    print("🧪 Testing Affordance System")
    
    # Connect to PyBullet
    p.connect(p.GUI)
    
    try:
        # 1. Setup environment
        robot_id, object_ids = setup_environment(num_objects=3)
        print(f"✅ Environment setup: Robot={robot_id}, Objects={object_ids}")
        
        # 2. Reset robot to home position
        print("🏠 Resetting robot to home position...")
        reset_robot_home(robot_id)
        
        # Wait for stabilization
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 3. Capture camera data
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        rgb, depth, seg = get_rgb_depth_segmentation(width, height, view_matrix, proj_matrix)
        print(f"📷 Camera data: RGB {rgb.shape}, Depth {depth.shape}")
        
        # 4. Sample grasp candidates
        candidates = sample_grasp_candidates(
            depth=depth,
            num_angles=8,
            visualize=False,
            rgb=rgb,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            seg_mask=seg,
            object_ids=object_ids
        )
        print(f"📍 Sampled {len(candidates)} candidates")
        
        # 5. Test a few grasp candidates
        success_count = 0
        total_tests = min(10, len(candidates))
        
        for i, candidate in enumerate(candidates[:total_tests]):
            if len(candidate) == 4:
                u, v, theta_idx, theta = candidate
            else:
                u, v, theta_idx = candidate
                theta = theta_idx * np.pi / 4  # Convert to radians
            
            # Convert pixel to world coordinates using the original function
            from src.perception import pixel_to_world
            world_pos = pixel_to_world(u, v, depth[v, u], view_matrix, proj_matrix)
            
            print(f"   Test {i+1}/{total_tests}: Pixel({u}, {v}) -> World[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}], Angle={np.degrees(theta):.1f}°")
            
            # Test grasp
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
                    print(f"      ✅ SUCCESS!")
                else:
                    print(f"      ❌ Failed")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n📊 Results: {success_count}/{total_tests} successful grasps ({success_rate:.1f}% success rate)")
        
        return success_rate > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        p.disconnect()

if __name__ == "__main__":
    success = test_affordance_system()
    if success:
        print("🎉 Affordance system is working!")
    else:
        print("💔 Affordance system needs more work")