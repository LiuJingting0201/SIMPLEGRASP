#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能抓取测试 - 自动找到物体位置进行抓取
"""
import pybullet as p
import numpy as np
import cv2
import time
from geom import (setup_scene, set_topdown_camera, get_rgb_depth, 
                  pixel_to_world_on_plane, TABLE_TOP_Z)

def find_object_pixels(rgb, depth):
    """在图像中找到物体像素点"""
    # 转换为灰度图
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # 简单的物体检测：寻找非桌面颜色的区域
    # 桌面通常是棕色/灰色，物体是白色
    
    # 阈值化找到明亮区域（假设物体是白色的）
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_points = []
    for contour in contours:
        # 过滤太小的轮廓
        if cv2.contourArea(contour) > 100:  # 最小面积阈值
            # 获取轮廓中心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                object_points.append((cx, cy))
    
    return object_points

def smart_grasp_test():
    """智能抓取测试 - 自动找物体"""
    print("🧠 开始智能抓取测试...")
    
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
        
        # 保存原始图像用于调试
        cv2.imwrite("../data/debug_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("🖼️ 调试图像已保存到 ../data/debug_rgb.png")
        
        # 记录物体真实位置
        print("\n📍 物体真实位置:")
        for i, obj_id in enumerate(obj_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            print(f"   物体{i+1}: {pos}")
        
        # 智能检测物体像素
        object_pixels = find_object_pixels(rgb, depth)
        print(f"\n🔍 检测到 {len(object_pixels)} 个物体像素点:")
        
        if not object_pixels:
            print("❌ 没有检测到物体，尝试手动选择...")
            # 回退到图像中心
            object_pixels = [(W//2, H//2)]
        
        # 测试第一个检测到的物体
        target_u, target_v = object_pixels[0]
        print(f"🎯 选择目标: 像素({target_u}, {target_v})")
        
        # 转换到世界坐标
        world_pos = pixel_to_world_on_plane(target_u, target_v, W, H, view, proj)
        print(f"   世界坐标: {world_pos}")
        
        if world_pos is not None:
            # 执行抓取
            print(f"\n🦾 开始抓取...")
            success = attempt_grasp_with_feedback(robot_id, world_pos, 0.0)
            print(f"\n📊 抓取结果: {'✅ 成功' if success else '❌ 失败'}")
        
        print(f"\n🎮 测试完成！")
        print("按 Enter 键退出...")
        input()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        p.disconnect()

def attempt_grasp_with_feedback(robot_id, world_pos, grasp_angle):
    """带反馈的抓取函数"""
    print(f"   📍 目标位置: {world_pos}")
    print(f"   🔄 抓取角度: {np.degrees(grasp_angle):.1f}°")
    
    # 检查目标位置是否合理
    if world_pos[2] < TABLE_TOP_Z - 0.1 or world_pos[2] > TABLE_TOP_Z + 0.1:
        print(f"   ⚠️  目标高度异常: {world_pos[2]:.3f} (桌面: {TABLE_TOP_Z})")
    
    # 抓取参数
    pre_grasp_height = TABLE_TOP_Z + 0.15
    grasp_height = TABLE_TOP_Z + 0.02  # 更接近桌面
    lift_height = TABLE_TOP_Z + 0.25
    
    euler = [0, np.pi, grasp_angle]
    grasp_ori = p.getQuaternionFromEuler(euler)
    ee_link = 11
    
    try:
        # 记录周围物体的初始位置
        nearby_objects = []
        for body_id in range(p.getNumBodies()):
            if body_id != 0 and body_id != robot_id:  # 排除地面和机器人
                try:
                    pos, _ = p.getBasePositionAndOrientation(body_id)
                    dist = np.linalg.norm(np.array(pos[:2]) - np.array(world_pos[:2]))
                    if dist < 0.1:  # 10cm范围内
                        nearby_objects.append((body_id, pos))
                        print(f"   🎯 附近物体: ID{body_id}, 距离{dist:.3f}m")
                except:
                    pass
        
        # 执行抓取动作（简化版）
        print("   ⬆️  移动到预抓取位置...")
        
        # 1. 预抓取
        pre_grasp_pos = world_pos.copy()
        pre_grasp_pos[2] = pre_grasp_height
        joints_pre = p.calculateInverseKinematics(robot_id, ee_link, pre_grasp_pos, grasp_ori)
        
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joints_pre[i])
        
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.04)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.04)
        
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("   ⬇️  下降并抓取...")
        
        # 2. 下降和抓取
        grasp_pos = world_pos.copy()
        grasp_pos[2] = grasp_height
        joints_grasp = p.calculateInverseKinematics(robot_id, ee_link, grasp_pos, grasp_ori)
        
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joints_grasp[i])
        
        for _ in range(80):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 3. 闭合夹爪
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.0)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.0)
        
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("   ⬆️  抬起...")
        
        # 4. 抬起
        lift_pos = world_pos.copy()
        lift_pos[2] = lift_height
        joints_lift = p.calculateInverseKinematics(robot_id, ee_link, lift_pos, grasp_ori)
        
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joints_lift[i])
        
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 5. 检查结果
        success = False
        for body_id, initial_pos in nearby_objects:
            try:
                final_pos, _ = p.getBasePositionAndOrientation(body_id)
                height_change = final_pos[2] - initial_pos[2]
                
                if final_pos[2] > TABLE_TOP_Z + 0.08:
                    success = True
                    print(f"   ✅ 物体ID{body_id}被抬起 {height_change:+.3f}m 到 {final_pos[2]:.3f}m")
                    break
                else:
                    print(f"   📍 物体ID{body_id}高度变化 {height_change:+.3f}m")
            except:
                print(f"   ❓ 物体ID{body_id}状态未知")
        
        return success
        
    except Exception as e:
        print(f"   ❌ 抓取失败: {e}")
        return False

if __name__ == "__main__":
    smart_grasp_test()
