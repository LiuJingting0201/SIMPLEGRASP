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

# 物体间距参数 - 减小最小距离要求
MIN_OBJECT_DISTANCE = 0.06  # Reduced from 0.10 to 0.06 (6cm minimum distance)
MAX_SPAWN_ATTEMPTS = 20     # Reduced from 50 to 20 for faster placement

def setup_environment(num_objects=3):
    """Sets up the simulation environment with a robot, table, and objects."""
    print("🏗️  Setting up the environment...")
    
    # ✨ 修复：重置模拟以确保干净的状态
    p.resetSimulation()
    
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
    removed_objects = []
    
    for obj_id in object_ids:
        try:
            # ✨ 额外检查：确保物体ID仍然存在
            if obj_id <= 2:  # 跳过环境物体ID
                print(f"   ⚠️  跳过环境物体ID {obj_id}")
                continue
                
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            
            # ✨ 更严格的位置检查
            in_workspace = (
                pos[2] > TABLE_TOP_Z - 0.1 and  # Not fallen below table
                pos[2] < TABLE_TOP_Z + 0.5 and  # Not too high (carried away)
                abs(pos[0] - OBJECT_SPAWN_CENTER[0]) < 0.4 and  # Still in X range
                abs(pos[1] - OBJECT_SPAWN_CENTER[1]) < 0.4      # Still in Y range
            )
            
            if in_workspace:
                active_objects.append(obj_id)
            else:
                print(f"   🗑️  Object {obj_id} outside workspace (pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]) - will be removed")
                removed_objects.append(obj_id)
                
        except:
            # Object might have been removed from simulation
            print(f"   ❌ Object {obj_id} no longer exists in simulation")
            removed_objects.append(obj_id)
            continue
    
    # ✨ 修复：物理移除超出工作空间的物体（但保护环境物体）
    if removed_objects:
        print(f"   🧹 清理 {len(removed_objects)} 个超出工作空间的物体...")
        for obj_id in removed_objects:
            if obj_id <= 2:  # 保护环境物体
                print(f"      🛡️  保护环境物体 {obj_id}，不移除")
                continue
                
            try:
                p.removeBody(obj_id)
                print(f"      ✅ 移除物体 {obj_id}")
            except:
                print(f"      ⚠️  无法移除物体 {obj_id} (可能已被移除)")
    
    return active_objects

def cleanup_workspace():
    """清理工作空间中的所有动态物体"""
    print("   🧹 清理工作空间...")
    
    all_bodies = []
    for i in range(p.getNumBodies()):
        body_id = p.getBodyUniqueId(i)
        all_bodies.append(body_id)
    
    removed_count = 0
    for body_id in all_bodies:
        try:
            # 保护环境物体
            if body_id <= 2:
                continue
            
            # 检查物体位置
            pos, _ = p.getBasePositionAndOrientation(body_id)
            
            # ✨ 更激进的清理：移除工作空间外的物体
            dist_from_base = np.sqrt(pos[0]**2 + pos[1]**2)
            
            should_remove = (
                pos[2] < TABLE_TOP_Z - 0.2 or pos[2] > TABLE_TOP_Z + 1.0 or  # 高度范围
                dist_from_base < 0.3 or dist_from_base > 0.9 or              # 距离范围（比工作空间略大）
                abs(pos[1]) > 0.5                                            # Y轴范围
            )
            
            if should_remove:
                p.removeBody(body_id)
                removed_count += 1
                print(f"      🗑️  移除物体 {body_id} at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] 距离={dist_from_base:.3f}m")
                
        except:
            continue
    
    print(f"   ✅ 清理了 {removed_count} 个物体")

def reset_objects_after_grasp(object_ids, min_objects=2):
    """简化的物体重置"""
    active_objects = update_object_states(object_ids)
    
    if len(active_objects) < min_objects:
        print(f"⚠️  重新生成物体...")
        
        # 简化清理
        cleanup_workspace()
        
        # 等待稳定
        for _ in range(20):
            p.stepSimulation()
        
        # 创建新物体
        new_objects = create_better_objects(num_objects=min_objects)
        
        # 等待稳定
        for _ in range(30):
            p.stepSimulation()
            
        return new_objects
    
    return active_objects

def create_better_objects(num_objects=5):
    """Creates objects with more stable physical properties and ensures minimum spacing.
    
    Object size is constrained to fit within the Franka Panda gripper opening.
    Max gripper opening: ~0.08m (8cm), so objects should be < 0.06m (6cm) to be graspable.
    Objects are placed with minimum distance to avoid interference during grasping.
    """
    # Franka Panda gripper constraints
    MAX_GRIPPER_OPENING = 0.08  # 8cm maximum
    SAFE_OBJECT_WIDTH = 0.035   # 3.5cm - safe size for reliable grasping
    
    object_ids = []
    object_positions = []  # Track positions to maintain distance
    
    # ✨ 修复：更合理的工作空间计算
    workspace_area = 0.30 * 0.50  # 30cm x 50cm workspace
    object_area = np.pi * (MIN_OBJECT_DISTANCE/2)**2  # Exclusion zone per object
    max_objects = max(1, int(workspace_area / object_area * 0.7))  # 提高到70%装填效率
    
    # ✨ 不要过度限制物体数量，允许稍微紧密的放置
    if num_objects <= 5:  # 对于合理的物体数量，不进行限制
        effective_num_objects = num_objects
        print(f"   🎯 Creating {num_objects} objects (requested, max theoretical: {max_objects})")
    else:
        effective_num_objects = min(num_objects, max_objects)
        print(f"   🎯 Creating {effective_num_objects} objects (limited from {num_objects}, max feasible: {max_objects})")
    
    for i in range(effective_num_objects):
        placed = False
        attempts = 0
        current_min_distance = MIN_OBJECT_DISTANCE
        
        while not placed and attempts < MAX_SPAWN_ATTEMPTS:
            attempts += 1
            
            # Generate random position in workspace
            x_pos = OBJECT_SPAWN_CENTER[0] + np.random.uniform(-0.15, 0.15)  # 0.45-0.75m
            y_pos = OBJECT_SPAWN_CENTER[1] + np.random.uniform(-0.25, 0.25)  # -0.25~0.25m
            candidate_pos = [x_pos, y_pos]
            
            # Check distance to existing objects
            too_close = False
            if len(object_positions) > 0:  # Only check if there are existing objects
                for existing_pos in object_positions:
                    distance = np.sqrt((candidate_pos[0] - existing_pos[0])**2 + 
                                     (candidate_pos[1] - existing_pos[1])**2)
                    if distance < current_min_distance:
                        too_close = True
                        break
            
            if not too_close:
                # Position is valid, create object here
                placed = True
                
            # ✨ 更激进的距离减少策略
            elif attempts > MAX_SPAWN_ATTEMPTS // 3:  # 早一点开始减少距离
                current_min_distance = MIN_OBJECT_DISTANCE * 0.7  # 减少到70%
                if attempts > MAX_SPAWN_ATTEMPTS * 0.6:
                    current_min_distance = MIN_OBJECT_DISTANCE * 0.5  # 进一步减少到50%
                if attempts > MAX_SPAWN_ATTEMPTS * 0.8:
                    current_min_distance = MIN_OBJECT_DISTANCE * 0.3  # 最后减少到30%
        
        if placed:
            object_positions.append(candidate_pos)
            
            shape_type = np.random.choice([p.GEOM_BOX, p.GEOM_CYLINDER])
            color = [np.random.random(), np.random.random(), np.random.random(), 1]
            
            if shape_type == p.GEOM_BOX:
                half_extents = [
                    np.random.uniform(0.015, SAFE_OBJECT_WIDTH/2),  # 更小的物体：1.2-1.75cm
                    np.random.uniform(0.015, SAFE_OBJECT_WIDTH/2),  
                    np.random.uniform(0.010, 0.020)                 # 更低的高度: 1.0-2.0cm
                ]
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
                z_pos = TABLE_TOP_Z + half_extents[2]
            elif shape_type == p.GEOM_CYLINDER:
                radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)  
                height = np.random.uniform(0.015, 0.030)                  # 更低的高度: 1.5-3.0cm
                shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
                z_pos = TABLE_TOP_Z + height / 2
            else: # p.GEOM_SPHERE
                radius = np.random.uniform(0.008, SAFE_OBJECT_WIDTH/2)  
                shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
                z_pos = TABLE_TOP_Z + radius

            body = p.createMultiBody(
                baseMass=np.random.uniform(0.05, 0.15),  # 更轻的物体
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x_pos, y_pos, z_pos + 0.002],  # 更贴近桌面
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)])
            )
            p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
            object_ids.append(body)
            
            actual_distance = current_min_distance if len(object_positions) > 1 else 0
            print(f"   📦 Object {i+1} placed at [{x_pos:.3f}, {y_pos:.3f}] (attempts: {attempts}, min_dist: {actual_distance*100:.1f}cm)")
        else:
            print(f"   ⚠️  Could not place object {i+1} after {MAX_SPAWN_ATTEMPTS} attempts")
            # ✨ 继续尝试放置，但减少距离要求
            print(f"       尝试使用更小的间距重新放置...")
            
            # 最后的尝试：使用非常小的间距
            for final_attempt in range(10):
                x_pos = OBJECT_SPAWN_CENTER[0] + np.random.uniform(-0.15, 0.15)
                y_pos = OBJECT_SPAWN_CENTER[1] + np.random.uniform(-0.25, 0.25)
                candidate_pos = [x_pos, y_pos]
                
                # 检查是否与现有物体重叠（使用很小的距离）
                min_distance_to_existing = 0.02  # 只要2cm间距
                too_close = False
                
                for existing_pos in object_positions:
                    distance = np.sqrt((candidate_pos[0] - existing_pos[0])**2 + 
                                     (candidate_pos[1] - existing_pos[1])**2)
                    if distance < min_distance_to_existing:
                        too_close = True
                        break
                
                if not too_close:
                    # 创建一个小物体
                    object_positions.append(candidate_pos)
                    
                    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.015])
                    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.015], 
                                                     rgbaColor=[np.random.random(), np.random.random(), np.random.random(), 1])
                    body = p.createMultiBody(
                        baseMass=0.05,
                        baseCollisionShapeIndex=shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=[x_pos, y_pos, TABLE_TOP_Z + 0.02]
                    )
                    p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
                    object_ids.append(body)
                    
                    print(f"   📦 Object {i+1} placed with reduced spacing at [{x_pos:.3f}, {y_pos:.3f}]")
                    placed = True
                    break
            
            if not placed:
                print(f"   ❌ 完全无法放置物体 {i+1}")
    
    # Print final spacing statistics
    if len(object_positions) > 1:
        distances = []
        for i in range(len(object_positions)):
            for j in range(i+1, len(object_positions)):
                dist = np.sqrt((object_positions[i][0] - object_positions[j][0])**2 + 
                              (object_positions[i][1] - object_positions[j][1])**2)
                distances.append(dist)
        
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        print(f"   📏 Object spacing - Min: {min_distance*100:.1f}cm, Avg: {avg_distance*100:.1f}cm")
    
    # ✨ 修改落后策略：总是至少创建一个物体
    if len(object_ids) == 0:
        print("   ❌ No objects could be placed! Creating a single object without distance constraints...")
        # Fallback: create at least one object in the center
        x_pos = OBJECT_SPAWN_CENTER[0]
        y_pos = OBJECT_SPAWN_CENTER[1]
        
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], 
                                         rgbaColor=[1, 0, 0, 1])  # Red fallback object
        body = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_pos, y_pos, TABLE_TOP_Z + 0.025]
        )
        p.changeDynamics(body, -1, lateralFriction=1.5, restitution=0.1)
        object_ids.append(body)
        print(f"   🆘 Created fallback object at center")
        
    print(f"   ✅ Successfully created {len(object_ids)} objects")
    return object_ids