#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单次抓取测试 - 验证抓取流程和成功判断
"""
import pybullet as p
import numpy as np
import time
from geom import (setup_scene, set_topdown_camera, get_rgb_depth, 
                  pixel_to_world_on_plane, TABLE_TOP_Z)

def test_single_grasp():
    """测试单次抓取流程"""
    print("🤖 开始抓取测试...")
    
    # 连接仿真
    p.connect(p.GUI)
    
    try:
        # 设置场景
        robot_id, table_id, obj_ids = setup_scene(add_objects=True, n_objects=2)
        print(f"✅ 场景设置完成，物体数量: {len(obj_ids)}")
        
        # 获取相机图像
        W, H, view, proj = set_topdown_camera()
        rgb, depth = get_rgb_depth(W, H, view, proj)
        print(f"📷 相机图像获取完成: {rgb.shape}")
        
        # 记录物体初始位置
        print("\n📍 物体初始位置:")
        initial_positions = {}
        for i, obj_id in enumerate(obj_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            initial_positions[obj_id] = pos
            print(f"   物体{i+1}: {pos}")
        
        # 选择图像中心点进行抓取测试
        target_u, target_v = W//2, H//2
        print(f"\n🎯 选择抓取目标: 像素({target_u}, {target_v})")
        
        # 转换到世界坐标
        world_pos = pixel_to_world_on_plane(target_u, target_v, W, H, view, proj)
        print(f"   世界坐标: {world_pos}")
        
        if world_pos is not None:
            # 执行抓取
            grasp_angle = 0.0  # 简单测试，使用0度角
            print(f"\n🦾 开始抓取...")
            print(f"   位置: {world_pos}")
            print(f"   角度: {np.degrees(grasp_angle):.1f}°")
            
            success = attempt_grasp_test(robot_id, world_pos, grasp_angle)
            
            # 检查结果
            print(f"\n📊 抓取结果: {'✅ 成功' if success else '❌ 失败'}")
            
            # 显示物体最终位置
            print("\n📍 物体最终位置:")
            for i, obj_id in enumerate(obj_ids):
                try:
                    final_pos, _ = p.getBasePositionAndOrientation(obj_id)
                    initial_pos = initial_positions[obj_id]
                    height_change = final_pos[2] - initial_pos[2]
                    
                    print(f"   物体{i+1}: {final_pos}")
                    print(f"            高度变化: {height_change:+.3f}m")
                    
                    if final_pos[2] > TABLE_TOP_Z + 0.05:
                        print(f"            ✅ 被抬起")
                    else:
                        print(f"            ❌ 仍在桌面")
                except:
                    print(f"   物体{i+1}: 已被移除或错误")
        
        print(f"\n🎮 测试完成！请查看PyBullet GUI中的结果")
        print("按 Enter 键退出...")
        input()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        p.disconnect()

def attempt_grasp_test(robot_id, world_pos, grasp_angle):
    """改进的抓取测试函数 - 添加调试信息"""
    print("   🔍 开始抓取调试...")
    
    # 检查机械臂初始状态
    print("   📊 机械臂初始状态:")
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_state = p.getJointState(robot_id, i)
        if joint_info[2] != p.JOINT_FIXED:  # 只显示可动关节
            print(f"      关节{i}: {joint_state[0]:.3f} rad ({np.degrees(joint_state[0]):.1f}°)")
    
    # 获取末端执行器当前位置
    ee_link = 11
    ee_state = p.getLinkState(robot_id, ee_link)
    current_ee_pos = ee_state[0]
    current_ee_orn = ee_state[1]
    print(f"   📍 当前末端位置: {current_ee_pos}")
    print(f"   🔄 当前末端姿态: {current_ee_orn}")
    
    # 抓取高度设置
    pre_grasp_height = TABLE_TOP_Z + 0.15
    grasp_height = TABLE_TOP_Z + 0.05
    lift_height = TABLE_TOP_Z + 0.25
    
    # 根据角度计算姿态
    euler = [0, np.pi, grasp_angle]
    grasp_ori = p.getQuaternionFromEuler(euler)
    
    print(f"   🎯 目标位置: {world_pos}")
    print(f"   🔄 目标姿态: {grasp_ori}")
    
    try:
        # 1. 预抓取位置
        print("   ⬆️  计算预抓取逆运动学...")
        pre_grasp_pos = world_pos.copy()
        pre_grasp_pos[2] = pre_grasp_height
        
        # 计算逆运动学
        joints_pre = p.calculateInverseKinematics(
            robot_id, ee_link, pre_grasp_pos, grasp_ori,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        if joints_pre is None:
            print("   ❌ 逆运动学求解失败")
            return False
        
        print(f"   📐 预抓取关节角度: {[f'{np.degrees(j):.1f}°' for j in joints_pre[:7]]}")
        
        # 检查关节限制
        valid_joints = True
        for i in range(7):
            joint_info = p.getJointInfo(robot_id, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit < upper_limit:  # 有限制
                if joints_pre[i] < lower_limit or joints_pre[i] > upper_limit:
                    print(f"   ⚠️  关节{i}超出限制: {np.degrees(joints_pre[i]):.1f}° (限制: {np.degrees(lower_limit):.1f}° - {np.degrees(upper_limit):.1f}°)")
                    valid_joints = False
        
        if not valid_joints:
            print("   ❌ 关节角度超出限制")
            return False
        
        # 先打开夹爪
        print("   🤲 打开夹爪...")
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.04, force=50)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.04, force=50)
        
        # 移动到预抓取位置
        print("   ⬆️  移动到预抓取位置...")
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL, 
                joints_pre[i], 
                force=500,  # 增加力度
                maxVelocity=1.0  # 限制速度
            )
        
        # 等待移动完成并监控进度
        max_wait = 300  # 最大等待步数
        for step in range(max_wait):
            p.stepSimulation()
            time.sleep(1./240.)
            
            # 每50步检查一次位置
            if step % 50 == 0:
                current_ee_state = p.getLinkState(robot_id, ee_link)
                current_pos = current_ee_state[0]
                distance = np.linalg.norm(np.array(current_pos) - np.array(pre_grasp_pos))
                print(f"      步骤{step}: 距离目标 {distance:.3f}m")
                
                if distance < 0.05:  # 5cm内认为到达
                    print(f"      ✅ 到达预抓取位置 (步骤 {step})")
                    break
        
        # 检查是否真的到达了
        final_ee_state = p.getLinkState(robot_id, ee_link)
        final_pos = final_ee_state[0]
        final_distance = np.linalg.norm(np.array(final_pos) - np.array(pre_grasp_pos))
        print(f"   📏 最终距离目标: {final_distance:.3f}m")
        
        if final_distance > 0.1:  # 10cm
            print("   ⚠️  未能到达预抓取位置，继续尝试...")
        
        # 2. 下降到抓取位置
        print("   ⬇️  下降到抓取位置...")
        grasp_pos = world_pos.copy()
        grasp_pos[2] = grasp_height
        
        joints_grasp = p.calculateInverseKinematics(
            robot_id, ee_link, grasp_pos, grasp_ori,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        if joints_grasp is None:
            print("   ❌ 抓取位置逆运动学求解失败")
            return False
        
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL, 
                joints_grasp[i], 
                force=500,
                maxVelocity=0.5  # 更慢的下降速度
            )
        
        for step in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
            
            if step % 50 == 0:
                current_ee_state = p.getLinkState(robot_id, ee_link)
                current_pos = current_ee_state[0]
                distance = np.linalg.norm(np.array(current_pos) - np.array(grasp_pos))
                print(f"      下降步骤{step}: 距离目标 {distance:.3f}m")
        
        print("   🤏 闭合夹爪...")
        # 3. 闭合夹爪
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.0, force=100)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.0, force=100)
        
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 检查夹爪状态
        gripper1_state = p.getJointState(robot_id, 9)
        gripper2_state = p.getJointState(robot_id, 10)
        print(f"   🤏 夹爪状态: {gripper1_state[0]:.3f}, {gripper2_state[0]:.3f}")
        
        print("   ⬆️  抬起物体...")
        # 4. 抬起
        lift_pos = world_pos.copy()
        lift_pos[2] = lift_height
        
        joints_lift = p.calculateInverseKinematics(
            robot_id, ee_link, lift_pos, grasp_ori,
            maxNumIterations=100
        )
        
        if joints_lift is None:
            print("   ❌ 抬起位置逆运动学求解失败")
            return False
        
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL, 
                joints_lift[i], 
                force=500,
                maxVelocity=1.0
            )
        
        for step in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
            
            if step % 50 == 0:
                current_ee_state = p.getLinkState(robot_id, ee_link)
                current_pos = current_ee_state[0]
                print(f"      抬起步骤{step}: 当前高度 {current_pos[2]:.3f}m")
        
        # 5. 判断成功
        print("   🔍 检查抓取结果...")
        success = False
        for body_id in range(p.getNumBodies()):
            if body_id == 0 or body_id == robot_id:  # 跳过地面和机器人
                continue
            try:
                obj_pos, _ = p.getBasePositionAndOrientation(body_id)
                if obj_pos[2] > TABLE_TOP_Z + 0.08:  # 降低成功阈值
                    success = True
                    print(f"   ✅ 检测到物体被抬起到 {obj_pos[2]:.3f}m")
                    break
                else:
                    print(f"   📍 物体{body_id}高度: {obj_pos[2]:.3f}m")
            except:
                continue
        
        if not success:
            print(f"   ❌ 没有物体被成功抬起")
        
        return success
        
    except Exception as e:
        print(f"   ❌ 抓取执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_single_grasp()
