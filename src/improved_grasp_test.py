# -*- coding: utf-8 -*-
import pybullet as p
import numpy as np
import time

from environment_setup import setup_environment
from perception import set_topdown_camera, get_rgb_depth, find_best_grasp_pixel, pixel_to_world
from grasp_test_logic import execute_grasp_sequence

def main():
    """Main function to run the grasp test."""
    print("🤖 Starting modular grasp test...")
    
    # Connect to PyBullet
    p.connect(p.GUI)
    
    try:
        # 1. Setup Environment
        robot_id, object_ids = setup_environment(num_objects=3)
        if not object_ids:
            print("❌ No objects were created. Aborting test.")
            return

        # 2. Perception (simplified - use actual object positions for now)
        print("📸 Performing perception...")
        
        # For testing, directly get object position from simulation
        # Later you can replace this with camera-based perception
        target_obj_id = object_ids[0]  # Pick first object
        obj_pos, obj_ori = p.getBasePositionAndOrientation(target_obj_id)
        
        # Get object's AABB to find the top surface
        aabb = p.getAABB(target_obj_id)
        object_top_z = aabb[1][2]  # Top of the object
        
        world_grasp_pos = np.array([obj_pos[0], obj_pos[1], object_top_z])
        print(f"   🎯 Target object ID: {target_obj_id}")
        print(f"   🌍 World grasp position: {world_grasp_pos}")

        # Add a marker for visualization
        p.addUserDebugText("TARGET", [world_grasp_pos[0], world_grasp_pos[1], world_grasp_pos[2] + 0.1], 
                          textColorRGB=[1, 0, 0], textSize=1.5)
        p.addUserDebugLine([world_grasp_pos[0], world_grasp_pos[1], world_grasp_pos[2]], 
                          [world_grasp_pos[0], world_grasp_pos[1], 0], 
                          lineColorRGB=[1, 1, 0], lineWidth=2)

        # 3. Grasping
        print("\n🚀 Executing grasp sequence...")
        success = execute_grasp_sequence(robot_id, world_grasp_pos)
        
        if success:
            print("\n🎉 Grasp test completed successfully!")
        else:
            print("\n😢 Grasp test failed.")
        
        print("\n🎮 Test finished. Press Enter to exit...")
        input()
        
    except Exception as e:
        print(f"❌ An error occurred in the main test script: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    main()
