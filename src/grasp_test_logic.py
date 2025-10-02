# -*- coding: utf-8 -*-
import numpy as np
import pybullet as p
import time

# Constants from the original script
GRIPPER_OPEN_VALUE = 0.04
GRIPPER_CLOSED_VALUE = 0.00
TABLE_TOP_Z = 0.65

def move_to_position(robot_id, ee_link, target_pos, target_ori, action_name, velocity=1.5, max_steps=600):
    """移动机器人末端执行器到目标位置和姿态（关节空间规划）
    
    使用PyBullet的IK求解器，这是标准做法，符合工业机器人SDK的行为。
    对于更高级的任务空间轨迹规划，应使用MoveIt或OMPL等专业库。
    """
    print(f"      🔄 执行 {action_name}...")
    
    # 记录初始状态
    initial_ee_state = p.getLinkState(robot_id, ee_link)
    initial_pos = initial_ee_state[0]
    initial_ori = initial_ee_state[1]
    print(f"         初始位置: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    print(f"         目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # 可视化目标末端执行器姿态
    rot_matrix = p.getMatrixFromQuaternion(target_ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    axis_len = 0.08
    p.addUserDebugLine(target_pos, target_pos + rot_matrix[:, 0] * axis_len, [1, 0.5, 0.5], 2, 5)
    p.addUserDebugLine(target_pos, target_pos + rot_matrix[:, 1] * axis_len, [0.5, 1, 0.5], 2, 5)
    p.addUserDebugLine(target_pos, target_pos + rot_matrix[:, 2] * axis_len, [0.5, 0.5, 1], 2, 5)
    p.addUserDebugText(action_name, target_pos + np.array([0, 0, 0.05]), [1, 1, 0], 1, 5)
    
    # 从URDF读取真实的关节限制
    ll, ul, jr, rp = [], [], [], []
    for i in range(7):
        joint_info = p.getJointInfo(robot_id, i)
        ll.append(joint_info[8])
        ul.append(joint_info[9])
        jr.append(joint_info[9] - joint_info[8])
        # 使用当前关节角度作为rest pose，这样IK会更倾向于保持当前配置
        current_joint_state = p.getJointState(robot_id, i)
        rp.append(current_joint_state[0])
    
    # IK求解（关节空间）- 使用更严格的参数以保证姿态准确性
    joints = p.calculateInverseKinematics(
        robot_id, ee_link, target_pos, target_ori,
        lowerLimits=ll, 
        upperLimits=ul, 
        jointRanges=jr, 
        restPoses=rp,
        maxNumIterations=200,      # 增加迭代次数
        residualThreshold=1e-5     # 更严格的误差阈值
    )
    
    if joints is None:
        print(f"      ❌ {action_name} IK求解失败")
        return False
    
    # 打印目标关节角度
    joint_angles_deg = [np.degrees(joints[i]) for i in range(7)]
    print(f"         目标关节角度（度）: {[f'{a:.1f}' for a in joint_angles_deg]}")
    
    # 设置关节控制（使用位置控制+速度限制+阻尼）
    for i in range(7):
        p.setJointMotorControl2(
            robot_id, i, 
            p.POSITION_CONTROL,
            targetPosition=joints[i], 
            force=300,                    # 力限制
            maxVelocity=velocity,         # 速度限制
            positionGain=0.3,            # P增益（降低可使运动更平滑）
            velocityGain=1.0             # D增益
        )
    
    # 可视化初始TCP坐标系（用虚线显示，持续10秒）
    initial_rot = p.getMatrixFromQuaternion(initial_ori)
    initial_rot = np.array(initial_rot).reshape(3, 3)
    tcp_axis_len = 0.06
    p.addUserDebugLine(initial_pos, initial_pos + initial_rot[:, 0] * tcp_axis_len, [1, 0, 0], 1, 10)  # 初始X-红
    p.addUserDebugLine(initial_pos, initial_pos + initial_rot[:, 1] * tcp_axis_len, [0, 1, 0], 1, 10)  # 初始Y-绿
    p.addUserDebugLine(initial_pos, initial_pos + initial_rot[:, 2] * tcp_axis_len, [0, 0, 1], 1, 10)  # 初始Z-蓝
    
    # 执行运动（等待关节到达目标位置）
    for step in range(max_steps):
        p.stepSimulation()
        time.sleep(1./240.)
        
        # 每50步显示一次进度
        if step % 50 == 0 and step > 0:
            current_ee_state = p.getLinkState(robot_id, ee_link)
            current_pos = current_ee_state[0]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            print(f"         步骤 {step}: 距离={distance:.3f}m, 当前Z={current_pos[2]:.3f}m")
        
        # 检查是否到达目标位置
        current_ee_state = p.getLinkState(robot_id, ee_link)
        current_pos = current_ee_state[0]
        current_ori = current_ee_state[1]
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if distance < 0.03:  # 3cm容差
            # 可视化成功到达时的TCP坐标系
            current_rot = p.getMatrixFromQuaternion(current_ori)
            current_rot = np.array(current_rot).reshape(3, 3)
            tcp_axis_len = 0.1
            p.addUserDebugLine(current_pos, current_pos + current_rot[:, 0] * tcp_axis_len, [1, 0.2, 0.2], 4, 15)
            p.addUserDebugLine(current_pos, current_pos + current_rot[:, 1] * tcp_axis_len, [0.2, 1, 0.2], 4, 15)
            p.addUserDebugLine(current_pos, current_pos + current_rot[:, 2] * tcp_axis_len, [0.2, 0.2, 1], 4, 15)
            p.addUserDebugText("实际TCP", current_pos + np.array([0, 0, 0.08]), [0, 1, 0], 1.5, 15)
            
            # 验证姿态（检查Z轴是否朝下 - Franka Panda的TCP Z轴应该朝下）
            z_axis_world = current_rot[:, 2]  # TCP Z轴在世界坐标系中的方向
            if z_axis_world[2] < -0.7:  # Z轴应该大致朝下（-Z方向）
                print(f"         ✅ {action_name} 完成 (位置误差: {distance:.3f}m, TCP姿态正确)")
            else:
                print(f"         ⚠️  {action_name} 完成但姿态可能不对 (TCP Z轴世界方向: [{z_axis_world[0]:.2f}, {z_axis_world[1]:.2f}, {z_axis_world[2]:.2f}])")
            
            return True
    
    # 超时：显示最终位置和姿态
    final_ee_state = p.getLinkState(robot_id, ee_link)
    final_pos = final_ee_state[0]
    final_ori = final_ee_state[1]
    final_distance = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
    
    # 可视化最终TCP坐标系（粗线，持续15秒）
    final_rot = p.getMatrixFromQuaternion(final_ori)
    final_rot = np.array(final_rot).reshape(3, 3)
    tcp_axis_len = 0.1
    p.addUserDebugLine(final_pos, final_pos + final_rot[:, 0] * tcp_axis_len, [1, 0.2, 0.2], 4, 15)  # 最终X-深红
    p.addUserDebugLine(final_pos, final_pos + final_rot[:, 1] * tcp_axis_len, [0.2, 1, 0.2], 4, 15)  # 最终Y-深绿
    p.addUserDebugLine(final_pos, final_pos + final_rot[:, 2] * tcp_axis_len, [0.2, 0.2, 1], 4, 15)  # 最终Z-深蓝
    p.addUserDebugText("实际TCP", final_pos + np.array([0, 0, 0.08]), [1, 0, 1], 1.5, 15)
    
    print(f"         ⚠️  {action_name} 超时")
    print(f"         最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"         位置误差: {final_distance:.3f}m")
    
    # 打印姿态误差（比较旋转矩阵）
    final_rot_flat = final_rot.flatten()
    target_rot_flat = rot_matrix.flatten()
    ori_error = np.linalg.norm(final_rot_flat - target_rot_flat)
    print(f"         姿态误差（矩阵差）: {ori_error:.3f}")
    
    return True  # 继续执行但给出警告

def enhanced_gripper_control(robot_id, target_width, force=50, max_velocity=0.2, wait_steps=120):
    """Controls the gripper to a target width."""
    finger_pos = target_width / 2.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=finger_pos, force=force, maxVelocity=max_velocity)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=finger_pos, force=force, maxVelocity=max_velocity)
    for _ in range(wait_steps):
        p.stepSimulation()
        time.sleep(1./240.)

def execute_grasp_sequence(robot_id, world_grasp_pos, grasp_angle=0):
    """Executes a grasp sequence at a given world position."""
    print(f"   🎯 开始抓取序列，目标位置: {world_grasp_pos}")

    pre_grasp_offset = 0.15  # 预抓取高度偏移（物体上方15cm）
    grasp_offset = -0.02     # 抓取时下降2cm到物体内部
    lift_height = world_grasp_pos[2] + 0.25  # 抬起高度
    ee_link = 11
    
    # 先移动到Home位置（快速降低高度）
    print("   🏠 步骤0: 移动到Home位置")
    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Franka Panda的标准Home姿态
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                               targetPosition=home_joints[i], force=300, maxVelocity=2.0)
    # 等待到达Home位置
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1./240.)
    
    ee_state = p.getLinkState(robot_id, ee_link)
    print(f"      ✅ Home位置: [{ee_state[0][0]:.3f}, {ee_state[0][1]:.3f}, {ee_state[0][2]:.3f}]")
    
    # 正确的夹爪朝下姿态
    # 对于Franka Panda TCP坐标系：Z轴是接近方向（应该朝下）
    # 需要让TCP的Z轴指向世界坐标系的-Z方向（下方）
    # PyBullet的欧拉角顺序是XYZ (roll, pitch, yaw)
    # 要让Z轴朝下，需要绕X轴旋转180度（翻转）
    # 然后绕Z轴旋转grasp_angle来调整抓取角度（手指方向）
    grasp_ori = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
    
    print(f"   🧭 夹爪姿态（欧拉角XYZ）: [π, 0, {grasp_angle:.2f}] = [180°, 0°, {np.degrees(grasp_angle):.0f}°]")
    
    # 可视化抓取点的目标TCP坐标系
    rot_matrix = p.getMatrixFromQuaternion(grasp_ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    # 绘制目标TCP坐标系（X=红色，Y=绿色，Z=蓝色）
    axis_length = 0.1
    x_end = world_grasp_pos + rot_matrix[:, 0] * axis_length
    y_end = world_grasp_pos + rot_matrix[:, 1] * axis_length
    z_end = world_grasp_pos + rot_matrix[:, 2] * axis_length
    
    p.addUserDebugLine(world_grasp_pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=0)  # X轴-红色
    p.addUserDebugLine(world_grasp_pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0)  # Y轴-绿色
    p.addUserDebugLine(world_grasp_pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=3, lifeTime=0)  # Z轴-蓝色
    p.addUserDebugText("目标TCP", world_grasp_pos + np.array([0, 0, 0.12]), [1, 1, 0], 1.2, 0)
    
    print(f"   🎨 已绘制目标TCP坐标系：")
    print(f"      X轴（红色）: [{rot_matrix[0, 0]:.3f}, {rot_matrix[1, 0]:.3f}, {rot_matrix[2, 0]:.3f}]")
    print(f"      Y轴（绿色）: [{rot_matrix[0, 1]:.3f}, {rot_matrix[1, 1]:.3f}, {rot_matrix[2, 1]:.3f}]")
    print(f"      Z轴（蓝色）: [{rot_matrix[0, 2]:.3f}, {rot_matrix[1, 2]:.3f}, {rot_matrix[2, 2]:.3f}] ← 接近方向（应该朝下）")
    
    # 验证Z轴确实朝下
    if rot_matrix[2, 2] < -0.9:  # Z轴的世界Z分量应该是负的
        print(f"      ✅ TCP Z轴正确朝下")
    else:
        print(f"      ⚠️  警告：TCP Z轴方向可能不对！")

    try:
        # 1. 打开夹爪
        print("   🤲 步骤1: 打开夹爪")
        enhanced_gripper_control(robot_id, GRIPPER_OPEN_VALUE, force=30)
        
        # 2. 移动到预抓取位置（物体上方）
        print(f"   ⬆️  步骤2: 移动到预抓取位置（物体上方{pre_grasp_offset*100:.0f}cm）")
        pre_grasp_pos = [world_grasp_pos[0], world_grasp_pos[1], world_grasp_pos[2] + pre_grasp_offset]
        print(f"      目标位置: {pre_grasp_pos}")
        if not move_to_position(robot_id, ee_link, pre_grasp_pos, grasp_ori, "预抓取"):
            return False
        
        # 3. 下降到抓取位置
        print(f"   ⬇️  步骤3: 下降到抓取位置")
        grasp_pos = [world_grasp_pos[0], world_grasp_pos[1], world_grasp_pos[2] + grasp_offset]
        print(f"      目标位置: {grasp_pos}")
        if not move_to_position(robot_id, ee_link, grasp_pos, grasp_ori, "抓取", velocity=0.2):
            return False
            
        # 4. 闭合夹爪
        print("   🤏 步骤4: 闭合夹爪")
        enhanced_gripper_control(robot_id, GRIPPER_CLOSED_VALUE, force=100, max_velocity=0.1)
        
        # 5. 抬起物体
        print(f"   📏 步骤5: 抬起物体到 {lift_height:.2f}m")
        lift_pos = [grasp_pos[0], grasp_pos[1], lift_height]
        if not move_to_position(robot_id, ee_link, lift_pos, grasp_ori, "抬起", velocity=0.2):
            return False

        # 检查是否成功抓取
        print("   ✅ 抓取序列执行完毕")
        return True

    except Exception as e:
        print(f"   ❌ An error occurred during the grasp sequence: {e}")
        return False
