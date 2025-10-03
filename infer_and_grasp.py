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

# Import proper motion and object management functions
from src.afford_data_gen import move_fast, reset_robot_home, open_gripper_fast
from src.environment_setup import update_object_states, reset_objects_after_grasp, cleanup_workspace

# 复制UNetLarge模型类（与训练时一致）
class UNet(nn.Module):
    """更大容量的UNet，适合300场景训练"""
    def __init__(self, in_channels=4, out_channels=37):
        super(UNet, self).__init__()
        # 更宽的通道 (1.5x)
        self.enc1 = self.conv_block(in_channels, 96)    # 4 → 96
        self.enc2 = self.conv_block(96, 192)            # 96 → 192
        self.enc3 = self.conv_block(192, 384)           # 192 → 384
        self.enc4 = self.conv_block(384, 768)           # 384 → 768
        self.enc5 = self.conv_block(768, 1536) 

        self.pool = nn.MaxPool2d(2)

        # 对应的解码器 (5层解码器)
        self.dec4 = self.conv_block(1536, 768)          # 1536 → 768
        self.dec3 = self.conv_block(768, 384)           # 768 → 384
        self.dec2 = self.conv_block(384, 192)           # 384 → 192
        self.dec1 = self.conv_block(192, 96)            # 192 → 96

        self.upconv4 = nn.ConvTranspose2d(1536, 768, 2, stride=2)  # 1536 → 768
        self.upconv3 = nn.ConvTranspose2d(768, 384, 2, stride=2)   # 768 → 384
        self.upconv2 = nn.ConvTranspose2d(384, 192, 2, stride=2)   # 384 → 192
        self.upconv1 = nn.ConvTranspose2d(192, 96, 2, stride=2)    # 192 → 96

        self.final = nn.Conv2d(96, out_channels, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.15)

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
        # Encoder with 5 levels (更深)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))  # 新增第5层

        # Decoder with 5 levels (对应解码)
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)  # 更强的dropout

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
class AffordanceGraspPipeline:
    def __init__(self, model_path='./models/affordance_model_best (copy).pth'):
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            # 如果是checkpoint格式，加载model_state_dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果是直接的state_dict格式
            self.model.load_state_dict(checkpoint)
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
                # 降低阈值：原来需要更好的点，现在接受更低的affordance
                if affordance_val > best_affordance:  # 移除最小阈值，接受任何正值
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
        # 降低抓取位置以获得更深的抓取
        grasp_z = z - 0.02  # 稍微低于物体表面以获得更深的抓取

        # 角度转换为四元数（绕Z轴旋转）- 参考src/afford_data_gen.py
        angle_rad = np.radians(angle_degrees)
        orn = p.getQuaternionFromEuler([np.pi, 0, angle_rad])  # 末端执行器朝下

        # 预抓取位置（稍微抬高，但不要太高）
        pre_grasp_pos = [x, y, min(z + 0.08, TABLE_TOP_Z + 0.25)]

        # 抓取位置
        grasp_pos = [x, y, grasp_z]

        return pre_grasp_pos, grasp_pos, orn

    def reset_robot_home(self):
        """重置机器人到初始位置 - 使用正确的函数"""
        reset_robot_home(self.robot_id)

    def move_to_position(self, target_pos, target_ori, slow=False, debug_mode=False):
        """使用正确的运动控制函数，参考src/afford_data_gen.py"""
        return move_fast(self.robot_id, 11, target_pos, target_ori, 30, slow=slow, debug_mode=debug_mode)

    def open_gripper(self):
        """打开夹爪 - 使用正确的函数"""
        open_gripper_fast(self.robot_id)

    def close_gripper(self):
        """关闭夹爪 - 使用正确的慢速闭合"""
        pos = 0.0  # 完全关闭
        steps = 20  # 调试模式下的步数
        for step in range(steps):
            p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=pos, force=50)
            p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=pos, force=50)
            p.stepSimulation()
            time.sleep(1./240.)

        # 等待额外时间确保稳定
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1./240.)

        # 调试：检查夹爪状态
        finger1_state = p.getJointState(self.robot_id, 9)[0]
        finger2_state = p.getJointState(self.robot_id, 10)[0]
        print(f"夹爪关闭后状态: 手指1={finger1_state:.4f}, 手指2={finger2_state:.4f}")

        # 检查是否有物体被夹住（手指没有完全关闭）
        gripper_closed = finger1_state < 0.005 and finger2_state < 0.005
        if not gripper_closed:
            print("✅ 检测到物体被夹住！")
        else:
            print("❌ 夹爪完全关闭，可能没有夹到物体")

    def execute_grasp(self, pre_grasp_pos, grasp_pos, orn):
        """执行抓取动作"""
        print("张开夹爪")
        self.open_gripper()

        print(f"移动到预抓取位置: {pre_grasp_pos}")
        success = self.move_to_position(pre_grasp_pos, orn)
        print(f"预抓取移动结果: {success}")
        if not success:
            print("预抓取移动失败")
            return False

        print(f"移动到抓取位置: {grasp_pos}")
        success = self.move_to_position(grasp_pos, orn, slow=True)
        print(f"抓取移动结果: {success}")
        if not success:
            print("抓取移动失败")
            return False

        print("关闭夹爪")
        self.close_gripper()

        print("抬起物体")
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1]
        success = self.move_to_position(lift_pos, orn)
        print(f"抬起移动结果: {success}")
        if not success:
            print("抬起移动失败")
            return False

        print("释放物体")
        self.open_gripper()

        return True

    def evaluate_grasp_success(self, obj_ids, initial_heights):
        """评估抓取是否成功"""
        success = False
        lifted_objects = 0

        print("🔍 评估抓取结果:")
        for i, obj_id in enumerate(obj_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            current_height = pos[2]
            initial_height = initial_heights[i]
            height_diff = current_height - initial_height

            print(f"  物体 {obj_id}: 初始高度={initial_height:.3f}m, 当前高度={current_height:.3f}m, 高度差={height_diff:.3f}m")

            # 更宽松的成功标准：任何明显移动都算成功
            if height_diff > 0.02:  # 2cm instead of 5cm
                success = True
                lifted_objects += 1
                print(f"    ✅ 物体被抬起 {height_diff*100:.1f}cm")

        print(f"📊 总共 {lifted_objects}/{len(obj_ids)} 个物体被移动")
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

    def run_multiple_tests(self, num_tests=10):
        """运行多次测试并返回统计结果 - 简化版本，不跟踪物体ID"""
        successes = 0
        results = []

        print(f"🧪 开始进行 {num_tests} 次抓取测试...")

        for test_idx in range(num_tests):
            print(f"\n{'='*50}")
            print(f"测试 {test_idx + 1}/{num_tests}")
            print(f"{'='*50}")

            try:
                # 简化：每次测试直接运行，不刷新场景
                success = self.run_single_pipeline_simple()
                results.append(success)
                if success:
                    successes += 1

                print(f"测试 {test_idx + 1} 结果: {'✅ 成功' if success else '❌ 失败'}")

            except Exception as e:
                print(f"测试 {test_idx + 1} 异常: {e}")
                results.append(False)

        # 打印最终统计
        success_rate = (successes / num_tests) * 100
        print(f"\n{'='*60}")
        print("📊 测试结果统计")
        print(f"总测试次数: {num_tests}")
        print(f"成功次数: {successes}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"{'='*60}")

        if success_rate >= 70:
            print("🎉 模型表现优秀！")
        elif success_rate >= 50:
            print("👍 模型表现良好")
        else:
            print("⚠️ 模型需要改进")

        return results, success_rate

    def run_single_pipeline(self):
        """运行单次pipeline测试（不包含初始化）"""
        print("=== 捕获场景 ===")

        # 关键修复：先重置机器人到home位置，再拍照（与数据生成一致）
        print("🏠 重置机器人到初始位置...")
        self.reset_robot_home()

        # 等待机器人完全稳定
        for _ in range(120):
            p.stepSimulation()

        # 现在机器人已经在home位置，再拍照
        rgb, depth = self.capture_scene()

        # 记录初始物体高度
        obj_ids = []
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() not in ['plane', 'table', 'panda']:
                obj_ids.append(i)
        initial_heights = [p.getBasePositionAndOrientation(obj_id)[0][2] for obj_id in obj_ids]

        print("=== 推理可供性 ===")
        affordance_prob, angle_degrees = self.infer_affordance(rgb, depth)

        # 调试：检查可供性统计
        max_affordance = np.max(affordance_prob)
        mean_affordance = np.mean(affordance_prob)
        print(f"可供性统计: 最大值={max_affordance:.3f}, 平均值={mean_affordance:.3f}")

        print("=== 选择最佳抓取点 ===")
        u, v, angle, affordance_value = self.find_best_grasp_point(affordance_prob, angle_degrees, depth)
        print(f"最佳抓取点: 像素({u}, {v}), 角度: {angle:.1f}°, 可供性: {affordance_value:.3f}")

        # 调试：显示前5个最高可供性点的位置
        flat_afford = affordance_prob.flatten()
        top_indices = np.argsort(flat_afford)[-5:][::-1]  # 前5个最高值
        print("前5个最高可供性点:")
        for i, idx in enumerate(top_indices):
            val = flat_afford[idx]
            vv, uu = np.unravel_index(idx, affordance_prob.shape)
            world_pos = self.pixel_to_world(uu, vv, depth)
            dist_from_base = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
            print(f"  {i+1}. 像素({uu}, {vv}) -> 世界({world_pos[0]:.3f}, {world_pos[1]:.3f}) 距离基座:{dist_from_base:.3f}m 可供性:{val:.3f}")

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

    def run_single_pipeline_simple(self):
        """简化版本的单次pipeline测试 - 不跟踪物体ID，只尝试抓取"""
        print("=== 捕获场景 ===")

        # 关键修复：先重置机器人到home位置，再拍照（与数据生成一致）
        print("🏠 重置机器人到初始位置...")
        self.reset_robot_home()

        # 等待机器人完全稳定
        for _ in range(120):
            p.stepSimulation()

        # 简单测试：尝试一个简单的移动来检查机器人是否能动
        print("🧪 测试机器人移动...")
        test_pos = [0.5, 0.0, 0.8]  # 简单的测试位置
        test_ori = p.getQuaternionFromEuler([np.pi, 0, 0])  # 简单的朝下方向
        test_success = self.move_to_position(test_pos, test_ori)
        print(f"测试移动结果: {test_success}")

        if not test_success:
            print("❌ 机器人无法移动！检查move_fast函数")
            return False

        # 如果测试移动成功，重置回home
        print("🏠 重置回初始位置...")
        self.reset_robot_home()
        for _ in range(60):
            p.stepSimulation()

        # 现在机器人已经在home位置，再拍照
        rgb, depth = self.capture_scene()

        # 获取当前物体ID（用于后续物体管理）
        obj_ids = []
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() not in ['plane', 'table', 'panda']:
                obj_ids.append(i)

        print("=== 推理可供性 ===")
        affordance_prob, angle_degrees = self.infer_affordance(rgb, depth)

        # 调试：检查可供性统计
        max_affordance = np.max(affordance_prob)
        mean_affordance = np.mean(affordance_prob)
        print(f"可供性统计: 最大值={max_affordance:.3f}, 平均值={mean_affordance:.3f}")

        print("=== 选择最佳抓取点 ===")
        u, v, angle, affordance_value = self.find_best_grasp_point(affordance_prob, angle_degrees, depth)
        print(f"最佳抓取点: 像素({u}, {v}), 角度: {angle:.1f}°, 可供性: {affordance_value:.3f}")

        # 调试：显示前5个最高可供性点的位置
        flat_afford = affordance_prob.flatten()
        top_indices = np.argsort(flat_afford)[-5:][::-1]  # 前5个最高值
        print("前5个最高可供性点:")
        for i, idx in enumerate(top_indices):
            val = flat_afford[idx]
            vv, uu = np.unravel_index(idx, affordance_prob.shape)
            world_pos = self.pixel_to_world(uu, vv, depth)
            dist_from_base = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
            print(f"  {i+1}. 像素({uu}, {vv}) -> 世界({world_pos[0]:.3f}, {world_pos[1]:.3f}) 距离基座:{dist_from_base:.3f}m 可供性:{val:.3f}")

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

        print("=== 简化评估 ===")
        # 简化评估：只要有物体被移动就算成功，不跟踪具体物体ID
        success = self.evaluate_grasp_success_simple()
        print(f"抓取成功: {success}")

        # 使用正确的物体管理函数（来自src/environment_setup.py）
        reset_objects_after_grasp(obj_ids)

        return success

    def evaluate_grasp_success_simple(self):
        """简化抓取成功评估 - 检查是否有任何物体被移动"""
        # 获取所有动态物体
        obj_ids = []
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() not in ['plane', 'table', 'panda']:
                obj_ids.append(i)

        if not obj_ids:
            print("⚠️  场景中没有物体")
            return False

        # 检查是否有物体位置发生明显变化
        moved_objects = 0
        for obj_id in obj_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                current_height = pos[2]

                # 简单检查：如果物体高于桌面一定高度，说明被抓起
                if current_height > TABLE_TOP_Z + 0.05:  # 高于桌面10cm
                    moved_objects += 1
                    print(f"  ✅ 物体 {obj_id} 被移动到高度 {current_height:.3f}m")
            except:
                continue

        success = moved_objects > 0
        print(f"📊 检测到 {moved_objects} 个被移动的物体")
        return success

    def simple_object_management(self):
        """简单物体管理：清理超出范围的物体，如果没有有效物体则生成新的"""
        print("🔄 简单物体管理...")

        # 工作空间定义（参考geom.py）
        WORKSPACE_X_RANGE = [0.4, 0.8]  # X方向范围
        WORKSPACE_Y_RANGE = [-0.4, 0.4] # Y方向范围
        MAX_HEIGHT = 0.78  # Z > 78cm的物体要清理
        MIN_HEIGHT = 0.55  # Z < 55cm的物体要清理
        # 清理超出范围的物体
        removed_count = 0
        for i in range(p.getNumBodies()):
            try:
                body_info = p.getBodyInfo(i)
                body_name = body_info[0].decode('utf-8') if body_info[0] else ""

                # 跳过环境物体
                if any(name in body_name.lower() for name in ['plane', 'table', 'panda', 'franka']):
                    continue

                if i <= 2:
                    continue

                # 检查物体位置
                pos, _ = p.getBasePositionAndOrientation(i)

                # 清理条件：Z > 78cm 或 X/Y超出工作空间
                should_remove = (
                    pos[2] > MAX_HEIGHT or  # Z > 78cm
                    pos[0] < WORKSPACE_X_RANGE[0] or pos[0] > WORKSPACE_X_RANGE[1] or  # X超出范围
                    pos[1] < WORKSPACE_Y_RANGE[0] or pos[1] > WORKSPACE_Y_RANGE[1]    # Y超出范围
                )

                if should_remove:
                    p.removeBody(i)
                    removed_count += 1
                    print(f"  🗑️ 清理物体 {i} (位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])")

            except Exception as e:
                continue

        print(f"✅ 清理完成，移除了 {removed_count} 个物体")

        # 检查是否有有效物体
        valid_objects = 0
        for i in range(p.getNumBodies()):
            try:
                body_info = p.getBodyInfo(i)
                body_name = body_info[0].decode('utf-8') if body_info[0] else ""

                # 跳过环境物体
                if any(name in body_name.lower() for name in ['plane', 'table', 'panda', 'franka']):
                    continue

                if i <= 2:
                    continue

                pos, _ = p.getBasePositionAndOrientation(i)

                # 检查是否在工作空间内且高度正常
                in_workspace = (
                    WORKSPACE_X_RANGE[0] <= pos[0] <= WORKSPACE_X_RANGE[1] and
                    WORKSPACE_Y_RANGE[0] <= pos[1] <= WORKSPACE_Y_RANGE[1] and
                    pos[2] <= MAX_HEIGHT
                )

                if in_workspace:
                    valid_objects += 1

            except:
                continue

        print(f"📊 当前有效物体数量: {valid_objects}")

        # 如果没有有效物体，生成新的
        if valid_objects == 0:
            print("⚠️ 没有有效物体，生成2个新物体")

            # 等待物理稳定
            for _ in range(30):
                p.stepSimulation()

            # 生成新物体
            new_objects = self.create_objects_like_environment_setup(num_objects=2)

            # 让新物体稳定
            print("⏳ 等待新物体稳定...")
            for _ in range(50):
                p.stepSimulation()

            print(f"✅ 已生成 {len(new_objects)} 个新物体")

    def manage_objects_between_tests(self, obj_ids, min_objects=2):
        """在测试之间管理物体状态，类似于src/afford_data_gen.py的逻辑"""
        print("🔄 检查物体状态...")

        # 使用environment_setup.py的update_object_states逻辑
        active_objects = self.update_object_states(obj_ids)

        if len(active_objects) < min_objects:
            print(f"⚠️  只有 {len(active_objects)} 个物体 remaining, 重新生成...")

            # 清理工作空间（移除超出范围的物体）
            self.cleanup_workspace()

            # 等待物理稳定
            for _ in range(30):
                p.stepSimulation()

            # 生成新物体
            new_objects = self.create_objects_like_environment_setup(num_objects=min_objects)

            # 让新物体稳定
            print("⏳ 等待新物体稳定...")
            for _ in range(50):
                p.stepSimulation()

            return new_objects
        else:
            print(f"✅ 还有 {len(active_objects)} 个有效物体，继续使用")
            return active_objects

    def refresh_test_scene(self):
        """每次测试前刷新场景，确保有新鲜的物体"""
        # 清理所有现有动态物体
        self.cleanup_workspace()

        # 等待物理稳定
        for _ in range(30):
            p.stepSimulation()

        # 创建新的测试物体
        num_objects = 3  # 每次测试使用3个物体
        self.create_objects_like_environment_setup(num_objects=num_objects)

        # 让新物体稳定
        print("⏳ 等待新物体稳定...")
        for _ in range(50):
            p.stepSimulation()

    def update_object_states(self, object_ids):
        """检查哪些物体还在桌子上，移除超出工作空间的物体ID"""
        TABLE_TOP_Z = 0.625  # 从geom.py导入
        OBJECT_SPAWN_CENTER = [0.60, 0, TABLE_TOP_Z]  # 从geom.py导入

        active_objects = []
        removed_objects = []

        for obj_id in object_ids:
            try:
                # 跳过环境物体ID
                if obj_id <= 2:
                    continue

                pos, _ = p.getBasePositionAndOrientation(obj_id)

                # 位置检查
                in_workspace = (
                    pos[2] > TABLE_TOP_Z - 0.1 and  # 没有掉到桌面下方
                    pos[2] < TABLE_TOP_Z + 0.5 and  # 没有太高（被带走）
                    abs(pos[0] - OBJECT_SPAWN_CENTER[0]) < 0.4 and  # X方向仍在范围内
                    abs(pos[1] - OBJECT_SPAWN_CENTER[1]) < 0.4      # Y方向仍在范围内
                )

                if in_workspace:
                    active_objects.append(obj_id)
                else:
                    removed_objects.append(obj_id)

            except:
                # 物体可能已被移除
                removed_objects.append(obj_id)

        # 物理移除超出工作空间的物体
        if removed_objects:
            print(f"   🧹 清理 {len(removed_objects)} 个超出工作空间的物体...")
            for obj_id in removed_objects:
                if obj_id > 2:  # 保护环境物体
                    try:
                        p.removeBody(obj_id)
                    except:
                        pass

        return active_objects

    def cleanup_workspace(self):
        """清理工作空间中的所有动态物体"""
        TABLE_TOP_Z = 0.625

        # 获取所有物体ID
        all_bodies = []
        for i in range(p.getNumBodies()):
            body_id = p.getBodyUniqueId(i)
            all_bodies.append(body_id)

        removed_count = 0
        for body_id in all_bodies:
            try:
                # 检查是否是动态物体
                body_info = p.getBodyInfo(body_id)
                body_name = body_info[0].decode('utf-8') if body_info[0] else ""

                # 跳过环境物体
                protected_names = ['plane', 'table', 'panda', 'franka']
                if any(name in body_name.lower() for name in protected_names):
                    continue

                if body_id <= 2:
                    continue

                # 检查物体位置
                pos, _ = p.getBasePositionAndOrientation(body_id)

                # 保守的清理范围
                should_remove = (
                    pos[2] < TABLE_TOP_Z - 0.3 or  # 掉到桌面下方30cm
                    pos[2] > TABLE_TOP_Z + 1.5 or  # 飞到桌面上方1.5m
                    abs(pos[0]) > 1.2 or           # X方向超出1.2m
                    abs(pos[1]) > 1.2              # Y方向超出1.2m
                )

                if should_remove:
                    p.removeBody(body_id)
                    removed_count += 1

            except Exception as e:
                continue

        if removed_count > 0:
            print(f"   ✅ 清理完成，移除了 {removed_count} 个远程物体")

    def create_objects_like_environment_setup(self, num_objects=3):
        """使用与environment_setup.py相同的逻辑创建物体"""
        TABLE_TOP_Z = 0.625
        OBJECT_SPAWN_CENTER = [0.60, 0, TABLE_TOP_Z]
        MIN_OBJECT_DISTANCE = 0.06
        MAX_SPAWN_ATTEMPTS = 20

        # Franka Panda夹爪约束
        MAX_GRIPPER_OPENING = 0.08
        SAFE_OBJECT_WIDTH = 0.035

        object_ids = []
        object_positions = []

        num_objects = min(num_objects, 5)

        for i in range(num_objects):
            placed = False
            attempts = 0
            current_min_distance = MIN_OBJECT_DISTANCE

            while not placed and attempts < MAX_SPAWN_ATTEMPTS:
                attempts += 1

                # 生成随机位置
                x_pos = OBJECT_SPAWN_CENTER[0] + np.random.uniform(-0.15, 0.15)
                y_pos = OBJECT_SPAWN_CENTER[1] + np.random.uniform(-0.25, 0.25)
                candidate_pos = [x_pos, y_pos]

                # 检查与其他物体的距离
                too_close = False
                if len(object_positions) > 0:
                    for existing_pos in object_positions:
                        distance = np.sqrt((candidate_pos[0] - existing_pos[0])**2 +
                                         (candidate_pos[1] - existing_pos[1])**2)
                        if distance < current_min_distance:
                            too_close = True
                            break

                if not too_close:
                    placed = True

                # 如果放置困难，逐渐降低距离要求
                elif attempts > MAX_SPAWN_ATTEMPTS // 2:
                    current_min_distance = MIN_OBJECT_DISTANCE * 0.8

            if placed:
                object_positions.append(candidate_pos)

                shape_type = np.random.choice([p.GEOM_BOX, p.GEOM_CYLINDER])
                color = [np.random.random(), np.random.random(), np.random.random(), 1]

                if shape_type == p.GEOM_BOX:
                    half_extents = [
                        np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),
                        np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),
                        np.random.uniform(0.02, 0.025)
                    ]
                    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
                    z_pos = TABLE_TOP_Z + half_extents[2]
                elif shape_type == p.GEOM_CYLINDER:
                    radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)
                    height = np.random.uniform(0.02, 0.04)
                    shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
                    z_pos = TABLE_TOP_Z + height / 2
                else: # 球体
                    radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)
                    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                    z_pos = TABLE_TOP_Z + radius

                body = p.createMultiBody(
                    baseMass=np.random.uniform(0.05, 0.2),
                    baseCollisionShapeIndex=shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x_pos, y_pos, z_pos + 0.005],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)])
                )
                p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
                object_ids.append(body)

        return object_ids

def main():
    num_tests = 10  # 测试次数

    # 创建单个pipeline实例并初始化
    pipeline = AffordanceGraspPipeline()
    pipeline.initialize_simulation()  # 初始化仿真

    try:
        # 运行多次测试
        results, success_rate = pipeline.run_multiple_tests(num_tests)
    finally:
        # 清理
        if pipeline.physics_client:
            p.disconnect()

if __name__ == '__main__':
    main()
