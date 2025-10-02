import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import cv2
from geom import (set_topdown_camera, get_rgb_depth, pixel_to_world_on_plane, 
                  TABLE_TOP_Z, setup_scene)

# 保存路径
IMG_SAVE_DIR = "../data/afford_images"
LABEL_SAVE_DIR = "../data/afford_labels"
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(LABEL_SAVE_DIR, exist_ok=True)

# 相机参数 - 使用与geom.py一致的设置
img_width = 224
img_height = 224
fov = 60.0

# 抓取角度离散化 - K个角度类别
NUM_ANGLES = 8  # 每45度一个角度类别 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
ANGLES = np.linspace(0, 2*np.pi, NUM_ANGLES, endpoint=False)  # [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]

def reset_scene():
    """重置场景，使用稳定的几何配置"""
    p.resetSimulation()
    # 使用geom.py中的统一场景设置
    robot_id, table_id, obj_ids = setup_scene(add_objects=True, n_objects=np.random.randint(2, 5))
    return robot_id, obj_ids

def get_camera_image():
    """使用geom.py的相机设置获取图像"""
    W, H, view, proj = set_topdown_camera(width=img_width, height=img_height)
    rgb, depth = get_rgb_depth(W, H, view, proj)
    return rgb, depth, view, proj

def attempt_grasp(panda_id, world_pos, grasp_angle):
    """尝试在指定位置和角度进行抓取"""
    # 抓取高度设置
    pre_grasp_height = TABLE_TOP_Z + 0.15  # 预抓取高度
    grasp_height = TABLE_TOP_Z + 0.05      # 抓取高度
    lift_height = TABLE_TOP_Z + 0.25       # 抬起高度
    
    # 根据角度计算末端执行器姿态
    # 工具Z轴朝下，绕Z轴旋转grasp_angle
    euler = [0, np.pi, grasp_angle]
    grasp_ori = p.getQuaternionFromEuler(euler)
    
    # 获取末端执行器链接ID（Panda通常是11）
    ee_link = 11
    
    try:
        # 1. 移动到预抓取位置
        pre_grasp_pos = world_pos.copy()
        pre_grasp_pos[2] = pre_grasp_height
        joints_pre = p.calculateInverseKinematics(panda_id, ee_link, pre_grasp_pos, grasp_ori)
        
        # 移动到预抓取位置
        for i in range(7):
            p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joints_pre[i])
        
        # 打开夹爪
        p.setJointMotorControl2(panda_id, 9, p.POSITION_CONTROL, 0.04)
        p.setJointMotorControl2(panda_id, 10, p.POSITION_CONTROL, 0.04)
        
        for _ in range(50):
            p.stepSimulation()
        
        # 2. 下降到抓取位置
        grasp_pos = world_pos.copy()
        grasp_pos[2] = grasp_height
        joints_grasp = p.calculateInverseKinematics(panda_id, ee_link, grasp_pos, grasp_ori)
        
        for i in range(7):
            p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joints_grasp[i])
        for _ in range(50):
            p.stepSimulation()
        
        # 3. 闭合夹爪
        p.setJointMotorControl2(panda_id, 9, p.POSITION_CONTROL, 0.0)
        p.setJointMotorControl2(panda_id, 10, p.POSITION_CONTROL, 0.0)
        for _ in range(30):
            p.stepSimulation()
        
        # 4. 抬起
        lift_pos = world_pos.copy()
        lift_pos[2] = lift_height
        joints_lift = p.calculateInverseKinematics(panda_id, ee_link, lift_pos, grasp_ori)
        
        for i in range(7):
            p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joints_lift[i])
        for _ in range(50):
            p.stepSimulation()
        
        # 5. 判断抓取成功：检查是否有物体被抬到足够高度
        success = False
        for body_id in range(p.getNumBodies()):
            if body_id == 0 or body_id == panda_id:  # 跳过地面和机器人
                continue
            try:
                obj_pos, _ = p.getBasePositionAndOrientation(body_id)
                if obj_pos[2] > TABLE_TOP_Z + 0.1:  # 物体被抬起
                    success = True
                    break
            except:
                continue
        
        return success
        
    except Exception as e:
        print(f"抓取失败: {e}")
        return False

def main():
    """生成自监督抓取可供性数据"""
    p.connect(p.GUI)
    
    # 数据收集参数
    n_scenes = 20        # 场景数量
    n_samples_per_scene = 30  # 每个场景采样的抓取候选数量
    
    all_labels = []
    
    for scene_id in range(n_scenes):
        print(f"\n=== 场景 {scene_id+1}/{n_scenes} ===")
        
        # 重置场景
        panda_id, obj_ids = reset_scene()
        
        # 让物体稳定
        for _ in range(200):
            p.stepSimulation()
        
        # 获取图像
        rgb, depth, view, proj = get_camera_image()
        
        # 保存图像
        img_path = os.path.join(IMG_SAVE_DIR, f"scene_{scene_id:04d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 在这个场景中采样多个抓取候选
        scene_labels = []
        
        for sample_id in range(n_samples_per_scene):
            # 随机采样像素位置（避免边缘）
            u = np.random.randint(30, img_width - 30)
            v = np.random.randint(30, img_height - 30)
            
            # 随机采样抓取角度
            angle_idx = np.random.randint(0, NUM_ANGLES)
            grasp_angle = ANGLES[angle_idx]
            
            # 像素到世界坐标转换
            world_pos = pixel_to_world_on_plane(u, v, img_width, img_height, view, proj)
            
            if world_pos is not None:
                # 尝试抓取
                success = attempt_grasp(panda_id, world_pos, grasp_angle)
                
                # 记录标签：[u, v, angle_idx, success]
                scene_labels.append([u, v, angle_idx, int(success)])
                
                print(f"  样本 {sample_id+1:2d}: ({u:3d},{v:3d}) 角度{angle_idx} -> {'成功' if success else '失败'}")
                
                # 重置场景（因为抓取可能改变了物体位置）
                panda_id, obj_ids = reset_scene()
                for _ in range(200):
                    p.stepSimulation()
        
        # 保存该场景的标签
        scene_labels = np.array(scene_labels)
        label_path = os.path.join(LABEL_SAVE_DIR, f"scene_{scene_id:04d}.npy")
        np.save(label_path, scene_labels)
        
        all_labels.append(scene_labels)
        
        print(f"  场景 {scene_id+1} 完成，成功率: {scene_labels[:, 3].mean():.2f}")
    
    # 保存汇总数据
    all_labels = np.vstack(all_labels)
    summary_path = os.path.join(LABEL_SAVE_DIR, "all_labels.npy")
    np.save(summary_path, all_labels)
    
    # 统计信息
    total_samples = len(all_labels)
    success_rate = all_labels[:, 3].mean()
    
    print(f"\n=== 数据生成完成 ===")
    print(f"总样本数: {total_samples}")
    print(f"总体成功率: {success_rate:.3f}")
    print(f"各角度成功率:")
    for angle_idx in range(NUM_ANGLES):
        angle_mask = all_labels[:, 2] == angle_idx
        if angle_mask.sum() > 0:
            angle_success_rate = all_labels[angle_mask, 3].mean()
            print(f"  角度 {angle_idx} ({np.degrees(ANGLES[angle_idx]):.0f}°): {angle_success_rate:.3f}")
    
    print(f"\n数据保存在:")
    print(f"  图像: {IMG_SAVE_DIR}")
    print(f"  标签: {LABEL_SAVE_DIR}")
    
    p.disconnect()

if __name__ == "__main__":
    main()
