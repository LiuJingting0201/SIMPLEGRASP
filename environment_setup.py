# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np

TABLE_TOP_Z = 0.625  # Table surface height (实际桌面高度)
TABLE_POS = [0.5, 0, 0]  # Table position (table center)
ROBOT_BASE_POS = [0, 0, TABLE_TOP_Z]  # Robot base mounted on the table surface

# 物体生成区域：在桌面上，机械臂前方
# 相机俯视目标也应该对准这个区域
# Franka Panda俯视抓取的理想范围：X在0.4-0.8米（距离基座更远，避免奇异点）
OBJECT_SPAWN_CENTER = [0.60, 0, TABLE_TOP_Z]  # 往前移动到0.6米！

def setup_environment(num_objects=3):
    """Sets up the simulation environment with a robot, table, and objects."""
    print("🏗️  Setting up the environment...")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)
    
    # Load plane and table
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=TABLE_POS, useFixedBase=True)
    
    # Load robot
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=ROBOT_BASE_POS, useFixedBase=True)
    
    # Create objects
    object_ids = create_better_objects(num_objects)
    
    print("⏳ Waiting for objects to settle...")
    for _ in range(100):
        p.stepSimulation()
        
    print(f"✅ Environment setup complete. Robot ID: {robot_id}, Object IDs: {object_ids}")
    return robot_id, object_ids

def update_object_states(object_ids):
    """Check which objects are still on the table and remove IDs of fallen/moved objects."""
    active_objects = []
    for obj_id in object_ids:
        try:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            # Check if object is still on table (not fallen or moved too far)
            if (pos[2] > TABLE_TOP_Z - 0.1 and  # Not fallen below table
                abs(pos[0] - OBJECT_SPAWN_CENTER[0]) < 0.5 and  # Still in X range
                abs(pos[1] - OBJECT_SPAWN_CENTER[1]) < 0.5):    # Still in Y range
                active_objects.append(obj_id)
            else:
                print(f"   🗑️  Object {obj_id} removed from active list (pos: {pos})")
        except:
            # Object might have been removed from simulation
            print(f"   ❌ Object {obj_id} no longer exists in simulation")
            continue
    
    return active_objects

def reset_objects_after_grasp(object_ids, min_objects=2):
    """Reset/respawn objects if too few remain for continued training."""
    active_objects = update_object_states(object_ids)
    
    if len(active_objects) < min_objects:
        print(f"⚠️  Only {len(active_objects)} objects remaining, respawning new ones...")
        
        # Remove remaining objects
        for obj_id in active_objects:
            p.removeBody(obj_id)
        
        # Create new batch of objects
        new_objects = create_better_objects(num_objects=5)
        
        # Let them settle
        print("⏳ Waiting for new objects to settle...")
        for _ in range(50):
            p.stepSimulation()
            
        return new_objects
    
    return active_objects

def create_better_objects(num_objects=5):
    """Creates objects with more stable physical properties.
    
    Object size is constrained to fit within the Franka Panda gripper opening.
    Max gripper opening: ~0.08m (8cm), so objects should be < 0.06m (6cm) to be graspable.
    """
    # Franka Panda gripper constraints
    MAX_GRIPPER_OPENING = 0.08  # 8cm maximum
    SAFE_OBJECT_WIDTH = 0.035   # 3.5cm - safe size for reliable grasping
    
    object_ids = []
    for i in range(num_objects):
        # 在机器人前方生成物体，确保在工作空间内
        # X: 0.45-0.75米（距离基座更远，俯视抓取更容易）
        # Y: -0.25到0.25米（左右范围）
        x_pos = OBJECT_SPAWN_CENTER[0] + np.random.uniform(-0.15, 0.15)  # 0.45-0.75m
        y_pos = OBJECT_SPAWN_CENTER[1] + np.random.uniform(-0.25, 0.25)  # -0.25~0.25m
        
        shape_type = np.random.choice([p.GEOM_BOX, p.GEOM_CYLINDER])
        color = [np.random.random(), np.random.random(), np.random.random(), 1]
        
        if shape_type == p.GEOM_BOX:
            # 限制盒子的尺寸，确保可以被夹爪抓取
            # 至少有一个维度要小于SAFE_OBJECT_WIDTH
            half_extents = [
                np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),  # 1.5-3.5cm
                np.random.uniform(0.02, SAFE_OBJECT_WIDTH/2),  # 1.5-3.5cm
                np.random.uniform(0.02, 0.03)                   # 高度: 1.5-3cm
            ]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            z_pos = TABLE_TOP_Z + half_extents[2]
        elif shape_type == p.GEOM_CYLINDER:
            # 圆柱体直径要小于夹爪开口
            radius = np.random.uniform(0.01, SAFE_OBJECT_WIDTH/2)  # 1-3.5cm
            height = np.random.uniform(0.03, 0.06)                  # 高度: 3-6cm
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            z_pos = TABLE_TOP_Z + height / 2
        else: # p.GEOM_SPHERE
            # 球体直径要小于夹爪开口
            radius = np.random.uniform(0.01, SAFE_OBJECT_WIDTH/2)  # 1-3.5cm
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            z_pos = TABLE_TOP_Z + radius

        body = p.createMultiBody(
            baseMass=np.random.uniform(0.05, 0.3),  # 较轻的物体更容易抓取
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_pos, y_pos, z_pos + 0.01],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)])
        )
        p.changeDynamics(body, -1, lateralFriction=2, restitution=0.1)
        object_ids.append(body)
        
    print(f"   📦 Created {num_objects} objects (size < {SAFE_OBJECT_WIDTH*100:.1f}cm for gripper compatibility)")
    return object_ids
