import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import pybullet as p
import pybullet_data
import time
import cv2
from geom import setup_scene, TABLE_TOP_Z
from perception import set_topdown_camera, get_rgb_depth, pixel_to_world, CAMERA_PARAMS

# 复制UNet模型类（与训练时一致）
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=37):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.final = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.upconv3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

class AffordanceGraspPipeline:
    def __init__(self, model_path='./models/affordance_model.pth'):
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"模型已加载到 {self.device}")

        # 角度类数
        self.num_angle_classes = 36

        # 相机参数
        self.camera_params = CAMERA_PARAMS

        # PyBullet初始化
        self.physics_client = None
        self.robot_id = None

    def initialize_simulation(self):
        """初始化PyBullet仿真环境"""
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # 设置场景
        self.robot_id, table_id, obj_ids = setup_scene(add_objects=True, n_objects=3)

        # 设置相机
        width, height, view_matrix, proj_matrix = set_topdown_camera()
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix

        return width, height, view_matrix, proj_matrix

    def capture_scene(self):
        """捕获当前场景的RGB和深度图像"""
        width, height = self.camera_params['width'], self.camera_params['height']
        rgb, depth = get_rgb_depth(width, height, self.view_matrix, self.proj_matrix)
        return rgb, depth

    def preprocess_input(self, rgb, depth):
        """预处理输入数据为模型格式"""
        # RGB转换为tensor
        transform = transforms.ToTensor()
        rgb_tensor = transform(Image.fromarray(rgb))

        # 深度转换为tensor
        depth_tensor = torch.tensor(depth).unsqueeze(0).float()

        # 拼接输入
        x = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(self.device)
        return x

    def infer_affordance(self, rgb, depth):
        """推理可供性和角度热力图"""
        x = self.preprocess_input(rgb, depth)

        with torch.no_grad():
            pred = self.model(x)  # (1, 37, H, W)

        # 分离可供性和角度
        affordance_logits = pred[0, 0]  # (H, W)
        angle_logits = pred[0, 1:]  # (36, H, W)

        # 可供性概率
        affordance_prob = torch.sigmoid(affordance_logits).cpu().numpy()

        # 角度预测
        angle_pred = torch.argmax(angle_logits, dim=0).cpu().numpy()  # (H, W)
        angle_degrees = angle_pred * (360.0 / self.num_angle_classes)

        return affordance_prob, angle_degrees

    def find_best_grasp_point(self, affordance_prob, angle_degrees, depth):
        """找到最佳抓取点（在有效工作空间内的最大可供性）"""
        height, width = affordance_prob.shape

        # 在有效工作空间内搜索最佳点
        best_affordance = -1
        best_u, best_v = -1, -1

        # 采样多个候选点，检查它们是否在有效工作空间内
        for _ in range(50):  # 采样50个候选点
            # 随机选择点，但偏向高可供性区域
            if np.random.random() < 0.7:  # 70%概率选择高可供性区域
                # 在可供性概率加权的区域采样
                flat_affordance = affordance_prob.flatten()
                flat_indices = np.arange(len(flat_affordance))
                # 使用可供性值作为权重进行采样
                weights = flat_affordance / flat_affordance.sum() if flat_affordance.sum() > 0 else None
                flat_idx = np.random.choice(flat_indices, p=weights)
                v, u = np.unravel_index(flat_idx, affordance_prob.shape)
            else:
                # 随机采样
                u = np.random.randint(10, width-10)
                v = np.random.randint(10, height-10)

            # 检查这个点是否在有效工作空间内
            world_pos = self.pixel_to_world(u, v, depth)
            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2)

            # 工作空间检查（与主pipeline相同）
            if (dist >= 0.25 and dist <= 0.85 and abs(y) <= 0.5 and
                z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15):
                # 这个点在有效工作空间内
                affordance_val = affordance_prob[v, u]
                if affordance_val > best_affordance:
                    best_affordance = affordance_val
                    best_u, best_v = u, v

        if best_u == -1:
            # 如果没有找到有效点，回退到图像中心（如果在工作空间内）
            u, v = width // 2, height // 2
            world_pos = self.pixel_to_world(u, v, depth)
            x, y, z = world_pos
            dist = np.sqrt(x**2 + y**2)
            if dist >= 0.25:  # 至少不离机器人太近
                best_u, best_v = u, v
            else:
                # 找到最远的有效点
                max_dist = 0
                for i in range(10, width-10, 20):
                    for j in range(10, height-10, 20):
                        world_pos = self.pixel_to_world(i, j, depth)
                        x, y, z = world_pos
                        dist = np.sqrt(x**2 + y**2)
                        if dist > max_dist and dist <= 0.85:
                            max_dist = dist
                            best_u, best_v = i, j

        u, v = best_u, best_v

        # 获取对应角度
        if v < angle_degrees.shape[0] and u < angle_degrees.shape[1]:
            angle = angle_degrees[v, u]
        else:
            angle = 0.0

        # 可供性值
        affordance_value = affordance_prob[v, u]

        return u, v, angle, affordance_value

    def pixel_to_world(self, u, v, depth):
        """像素坐标反投影到世界坐标"""
        return pixel_to_world(u, v, depth[v, u], self.view_matrix, self.proj_matrix)

    def generate_grasp_pose(self, grasp_point_world, angle_degrees, depth):
        """生成抓取姿态（预抓取和抓取位置）"""
        x, y, z = grasp_point_world

        # pixel_to_world已经给出了物体表面的世界坐标
        # 直接在物体表面上方一点进行抓取
        grasp_z = z + 0.01  # 稍微高于物体表面

        # 角度转换为四元数（绕Z轴旋转）- 参考src/afford_data_gen.py
        angle_rad = np.radians(angle_degrees)
        orn = p.getQuaternionFromEuler([np.pi, 0, angle_rad])  # 末端执行器朝下

        # 预抓取位置（稍微抬高，但不要太高）
        pre_grasp_pos = [x, y, min(z + 0.08, TABLE_TOP_Z + 0.25)]

        # 抓取位置
        grasp_pos = [x, y, grasp_z]

        return pre_grasp_pos, grasp_pos, orn

    def reset_robot_home(self):
        """重置机器人到初始位置 - 参考src/afford_data_gen.py"""
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

        # 确保在移动前夹爪是完全打开的
        print("确保夹爪完全打开...")

        # 多次尝试确保夹爪打开
        for attempt in range(3):
            self.open_gripper()
            finger_state = p.getJointState(self.robot_id, 9)[0]
            print(f"  尝试 {attempt+1}: 夹爪状态 = {finger_state:.4f}")

            if finger_state > 0.015:
                print("  ✅ 夹爪已确认打开")
                break
            else:
                print("  ⚠️  夹爪未完全打开，重试...")

        # 使用位置控制而不是直接设置关节状态，更平滑
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=home[i],
                force=500,
                maxVelocity=2.0
            )

        # 等待到位
        for _ in range(120):
            p.stepSimulation()

            # 检查是否到位
            all_in_position = True
            for i in range(7):
                current = p.getJointState(self.robot_id, i)[0]
                if abs(current - home[i]) > 0.05:  # 容差3度
                    all_in_position = False
                    break

            if all_in_position:
                break

        # 最后再次强制确保夹爪打开
        print("最终确保夹爪打开...")
        self.open_gripper()

        final_finger_state = p.getJointState(self.robot_id, 9)[0]
        print(f"机器人已回到初始位置，夹爪状态: {final_finger_state:.4f}")

    def move_to_position(self, target_pos, target_ori, slow=False, debug_mode=False):
        """正确的运动控制函数，参考src/afford_data_gen.py"""
        ee_link = 11  # 末端执行器link索引

        print(f"移动到位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        # 获取关节限制和rest poses
        home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # 使用home位置作为rest poses
        ll, ul, jr = [], [], []
        for i in range(7):
            info = p.getJointInfo(self.robot_id, i)
            ll.append(info[8])
            ul.append(info[9])
            jr.append(info[9] - info[8])
            print(f"关节{i}限制: [{ll[i]:.3f}, {ul[i]:.3f}] 范围: {jr[i]*180/np.pi:.1f}°")

        # IK求解 - 使用home位置作为rest poses
        joints = p.calculateInverseKinematics(
            self.robot_id, ee_link, target_pos, target_ori,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=home_joints,
            maxNumIterations=100,
            residualThreshold=1e-4
        )

        if not joints or len(joints) < 7:
            print("IK求解失败")
            return False

        # 检查关节限制 - 添加小容差
        joint_tolerance = 0.05  # 约3度容差
        for i in range(7):
            if joints[i] < ll[i] - joint_tolerance or joints[i] > ul[i] + joint_tolerance:
                print(f"关节{i}超出限制: {joints[i]:.3f} 不在 [{ll[i]-joint_tolerance:.3f}, {ul[i]+joint_tolerance:.3f}]")
                return False

        # 根据距离动态调整运动参数
        current_pos = p.getLinkState(self.robot_id, ee_link)[0]
        move_distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        print(f"需要移动距离: {move_distance*100:.1f}cm")

        if move_distance > 0.3:  # 如果距离超过30cm
            velocity = 2.0 if slow else 3.0    # 增加速度
            force = 1500 if slow else 2500     # 大幅增加力度  
            actual_steps = 300 if slow else 300  # 大幅增加步数确保到达
            print(f"远距离移动模式: 速度={velocity}, 力度={force}, 步数={actual_steps}")
        else:
            velocity = 1.0 if slow else 2.0
            force = 600 if slow else 1000
            actual_steps = 80 if slow else 60
            print(f"近距离移动模式: 速度={velocity}, 力度={force}, 步数={actual_steps}")

        # 控制每个关节
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=joints[i], force=force, maxVelocity=velocity
            )

        # 执行运动
        progress_interval = actual_steps // 8 if debug_mode else actual_steps // 4

        for step in range(actual_steps):
            p.stepSimulation()
            time.sleep(1./240.)

            # 更频繁的进度检查
            if step % progress_interval == 0:
                current = p.getLinkState(self.robot_id, ee_link)[0]
                dist = np.linalg.norm(np.array(current) - np.array(target_pos))
                progress = max(0, (move_distance - dist) / move_distance * 100)
                print(f"步骤 {step}/{actual_steps}: 距离目标 {dist*100:.1f}cm, 进度 {progress:.1f}%")

                # 早期成功检测
                if dist < 0.05:  # 如果已经很接近目标
                    print("提前到达目标")
                    break

        # 最终位置验证
        final_pos = p.getLinkState(self.robot_id, ee_link)[0]
        final_dist = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
        print(f"最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"最终误差: {final_dist*100:.1f}cm")

        # 根据移动距离动态调整容差
        if move_distance > 0.4:
            success_threshold = 0.15  # 15cm for very long moves
        elif move_distance > 0.2:
            success_threshold = 0.10  # 10cm for medium moves  
        else:
            success_threshold = 0.05  # 5cm for short moves

        success = final_dist < success_threshold

        if success:
            print(f"移动成功 (误差 {final_dist*100:.1f}cm < {success_threshold*100:.0f}cm)")
        else:
            print(f"移动失败 (误差 {final_dist*100:.1f}cm >= {success_threshold*100:.0f}cm)")

        return success

    def open_gripper(self):
        """打开夹爪"""
        pos = 0.04 / 2.0  # 完全打开
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=300, maxVelocity=3.0)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=300, maxVelocity=3.0)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1./240.)

    def close_gripper(self):
        """关闭夹爪"""
        pos = 0.0  # 完全关闭
        for step in range(40):
            p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
            p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=50, maxVelocity=0.05)
            p.stepSimulation()
            time.sleep(1./240.)

    def execute_grasp(self, pre_grasp_pos, grasp_pos, orn):
        """执行抓取动作"""
        print("张开夹爪")
        self.open_gripper()

        print(f"移动到预抓取位置: {pre_grasp_pos}")
        if not self.move_to_position(pre_grasp_pos, orn):
            print("预抓取移动失败")
            return False

        print(f"移动到抓取位置: {grasp_pos}")
        if not self.move_to_position(grasp_pos, orn, slow=True):
            print("抓取移动失败")
            return False

        print("关闭夹爪")
        self.close_gripper()

        print("抬起物体")
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1]
        if not self.move_to_position(lift_pos, orn):
            print("抬起移动失败")
            return False

        return True

    def evaluate_grasp_success(self, obj_ids, initial_heights):
        """评估抓取是否成功"""
        success = False
        for i, obj_id in enumerate(obj_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            current_height = pos[2]
            initial_height = initial_heights[i]

            # 检查物体是否被明显抬起（至少5cm）
            if current_height - initial_height > 0.05:
                success = True
                break

        return success

    def run_pipeline(self):
        """运行完整pipeline"""
        print("=== 初始化仿真环境 ===")
        width, height, view_matrix, proj_matrix = self.initialize_simulation()

        print("=== 捕获场景 ===")
        rgb, depth = self.capture_scene()

        # 记录初始物体高度
        obj_ids = []
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() not in ['plane', 'table', 'panda']:
                obj_ids.append(i)
        initial_heights = [p.getBasePositionAndOrientation(obj_id)[0][2] for obj_id in obj_ids]

        print("=== 推理可供性 ===")
        affordance_prob, angle_degrees = self.infer_affordance(rgb, depth)

        print("=== 选择最佳抓取点 ===")
        u, v, angle, affordance_value = self.find_best_grasp_point(affordance_prob, angle_degrees, depth)
        print(f"最佳抓取点: 像素({u}, {v}), 角度: {angle:.1f}°, 可供性: {affordance_value:.3f}")

        print("=== 反投影到世界坐标 ===")
        grasp_point_world = self.pixel_to_world(u, v, depth)
        print(f"世界坐标: {grasp_point_world}")

        # 提取坐标
        x, y, z = grasp_point_world

        # 额外的机器人基座距离检查 - 避免模型预测靠近机器人基座的错误点
        robot_base_dist = np.sqrt(x**2 + y**2)
        if robot_base_dist < 0.25:  # 如果距离机器人基座太近，强制跳过
            print(f"⚠️  模型预测的抓取点太靠近机器人基座 (距离={robot_base_dist:.3f}m)，跳过")
            return False

        # 检查工作空间 - 避免太靠近机器人导致关节角度过大
        dist = np.sqrt(x**2 + y**2)
        workspace_valid = (
            dist >= 0.25 and dist <= 0.85 and  # 增加最小距离到25cm，避免极端关节角度
            abs(y) <= 0.5 and
            z >= TABLE_TOP_Z - 0.05 and z <= TABLE_TOP_Z + 0.15
        )
        print(f"工作空间检查: 距离={dist:.3f}m, Y={y:.3f}m, Z={z:.3f}m, 桌面={TABLE_TOP_Z:.3f}m, 有效={workspace_valid}")

        if not workspace_valid:
            print("抓取点超出工作空间，跳过")
            return False

        print("=== 生成抓取姿态 ===")
        pre_grasp_pos, grasp_pos, orn = self.generate_grasp_pose(grasp_point_world, angle, depth[v, u])

        print("=== 执行抓取 ===")
        # 每次抓取前重置机器人到home位置
        self.reset_robot_home()
        success = self.execute_grasp(pre_grasp_pos, grasp_pos, orn)

        print("=== 评估成功 ===")
        success = self.evaluate_grasp_success(obj_ids, initial_heights)
        print(f"抓取成功: {success}")

        return success

def main():
    pipeline = AffordanceGraspPipeline()
    success = pipeline.run_pipeline()

    # 保持仿真运行
    while True:
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == '__main__':
    main()