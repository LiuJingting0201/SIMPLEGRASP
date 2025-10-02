# -*- coding: utf-8 -*-
import numpy as np
import pybullet as p

# ===== 统一的场景几何配置 (修复版本) =====
TABLE_TOP_Z = 0.625  # PyBullet内置table桌面高度近似值
ROBOT_BASE_POS = [0, 0, TABLE_TOP_Z - 0.05]  # 机械臂基座位置 (补偿机械臂自身的0.05偏移)
TABLE_POS = [0.6, 0, 0]         # 桌子位置 (稳定配置)
OBJECT_SPAWN_HEIGHT = TABLE_TOP_Z + 0.02  # 物体生成高度 (降低一点)

# 工作区域定义 (桌面上的安全抓取区域)
WORKSPACE_X_RANGE = [0.45, 0.75]  # X方向范围 (调整到桌子周围)
WORKSPACE_Y_RANGE = [-0.15, 0.15] # Y方向范围

# 相机配置
CAMERA_TARGET = (TABLE_POS[0], TABLE_POS[1], TABLE_TOP_Z)  # 相机目标点 - 与桌子位置对齐
CAMERA_DISTANCE = 0.65  # 相机距离
CAMERA_PARAMS = {
    'width': 224,
    'height': 224, 
    'fov': 60.0,
    'near': 0.1,
    'far': 2.0
}

def set_topdown_camera(target=CAMERA_TARGET, distance=CAMERA_DISTANCE, 
                       yaw=0.0, pitch=-89.0, **camera_params):
    """设置近似顶视相机，返回 (W,H, view, proj)。"""
    params = {**CAMERA_PARAMS, **camera_params}  # 合并默认参数和自定义参数
    cx, cy, cz = target
    eye = [cx, cy, cz + distance]    # 相机在桌面正上方
    up = [0, 1, 0]                   # 顶视时 up 方向取 Y 更稳定
    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(params['fov'], params['width']/float(params['height']), 
                                        params['near'], params['far'])
    return params['width'], params['height'], view, proj

def get_rgb_depth(width, height, view, proj):
    """获取 RGB & 深度（float32），RGB形状(H,W,3)。"""
    img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(img[2], dtype=np.uint8)[..., :3]
    depth = np.asarray(img[3], dtype=np.float32)
    return rgb, depth

def _mat_from_list(m): return np.array(m, dtype=np.float32).reshape(4,4)
def _invert_mat4(m):   return np.linalg.inv(m)

def pixel_to_world_on_plane(u, v, width, height, view, proj, plane_z=TABLE_TOP_Z):
    """像素(u,v)反投影到世界坐标，并与 z=plane_z 平面求交，返回 xyz。"""
    x_ndc = (2.0 * (u + 0.5) / width) - 1.0
    y_ndc = 1.0 - (2.0 * (v + 0.5) / height)
    pts_clip = np.array([[x_ndc, y_ndc, -1.0, 1.0],
                         [x_ndc, y_ndc,  1.0, 1.0]], dtype=np.float32)
    inv_vp = _invert_mat4(_mat_from_list(proj) @ _mat_from_list(view))
    pts_world = []
    for pc in pts_clip:
        pw = inv_vp @ pc
        pw = pw / pw[3]
        pts_world.append(pw[:3])
    p0, p1 = np.array(pts_world[0]), np.array(pts_world[1])
    dirv = p1 - p0
    if abs(dirv[2]) < 1e-6: return None
    t = (plane_z - p0[2]) / dirv[2]
    return p0 + t * dirv

def move_ee_via_ik(robot_id, ee_link, pos, orn=None, steps=240):
    """用IK移动末端到 pos, orn。"""
    if orn is None:
        orn = p.getQuaternionFromEuler([0, np.pi, 0])  # 工具Z朝下
    joints = p.calculateInverseKinematics(robot_id, ee_link, pos, orn, maxNumIterations=200)
    idxs = list(range(p.getNumJoints(robot_id)))
    p.setJointMotorControlArray(robot_id, idxs, p.POSITION_CONTROL, targetPositions=joints)
    for _ in range(steps): p.stepSimulation()

def control_gripper(robot_id, open_width=0.08, steps=120):
    """Franka手爪开合，关节9/10为手指关节。"""
    # 根据URDF，夹爪范围是0.000-0.040米，open_width需要调整
    max_width = 0.04  # 最大开口宽度
    target_width = min(open_width, max_width)  # 限制在有效范围内
    
    # 每个手指的位置是总宽度的一半
    finger_pos = target_width / 2.0
    
    print(f"设置夹爪宽度: {target_width:.3f}m (每个手指: {finger_pos:.3f}m)")
    
    p.setJointMotorControl2(robot_id, 9,  p.POSITION_CONTROL, 
                           targetPosition=finger_pos, force=20, maxVelocity=0.1)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 
                           targetPosition=finger_pos, force=20, maxVelocity=0.1)
    
    for _ in range(steps): 
        p.stepSimulation()
    
    # 返回实际手指位置用于验证
    finger1_pos = p.getJointState(robot_id, 9)[0]
    finger2_pos = p.getJointState(robot_id, 10)[0]
    actual_width = finger1_pos + finger2_pos
    print(f"实际夹爪宽度: {actual_width:.3f}m (手指1: {finger1_pos:.3f}m, 手指2: {finger2_pos:.3f}m)")
    
    return actual_width

def setup_scene(add_objects=True, n_objects=2, set_gravity=True):
    """统一的场景设置函数，确保所有位置一致。"""
    import pybullet_data
    
    if set_gravity:
        p.setGravity(0, 0, -9.8)
    
    # 设置PyBullet数据路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    plane_id = p.loadURDF("plane.urdf")
    
    # 加载桌子 (使用统一位置)
    table_id = p.loadURDF("table/table.urdf", TABLE_POS, useFixedBase=True)
    
    # 加载机械臂 (使用统一位置)
    robot_id = p.loadURDF("franka_panda/panda.urdf", ROBOT_BASE_POS, useFixedBase=True)
    
    obj_ids = []
    if add_objects:
        # 在工作区域内生成物体
        for _ in range(n_objects):
            x = TABLE_POS[0] + np.random.uniform(-0.15, 0.15)  # 基于桌子位置
            y = TABLE_POS[1] + np.random.uniform(-0.15, 0.15)  # 基于桌子位置
            z = OBJECT_SPAWN_HEIGHT
            
            # 随机选择物体类型 - 使用小尺寸物体
            if np.random.rand() > 0.5:
                obj_id = p.loadURDF("cube_small.urdf", [x, y, z])
            else:
                # 使用小球体而不是大球体
                obj_id = p.loadURDF("sphere_small.urdf", [x, y, z])
            
            # 设置物理属性 - 更稳定的设置
            p.changeDynamics(obj_id, -1, 
                           lateralFriction=2.0,    # 增加摩擦力
                           restitution=0.1,        # 降低弹性
                           linearDamping=0.8,      # 增加线性阻尼
                           angularDamping=0.8,     # 增加角阻尼
                           mass=0.1)               # 设置质量
            obj_ids.append(obj_id)
    
    # 让物体稳定下来 - 增加仿真时间
    for _ in range(1000):  # 增加到1000步
        p.stepSimulation()
    
    return robot_id, table_id, obj_ids

def is_position_in_workspace(x, y):
    """检查位置是否在工作区域内。"""
    return (WORKSPACE_X_RANGE[0] <= x <= WORKSPACE_X_RANGE[1] and 
            WORKSPACE_Y_RANGE[0] <= y <= WORKSPACE_Y_RANGE[1])
