# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import time
from geom import setup_scene

# 连接仿真（GUI）
p.connect(p.GUI)

# 使用统一的场景设置
robot_id, table_id, obj_ids = setup_scene(add_objects=True, n_objects=2)

print("场景加载完成 ✅")
print(f"机械臂 ID: {robot_id}")
print(f"桌子 ID: {table_id}")
print(f"物体 ID: {obj_ids}")

# 无限循环，保持 GUI 打开
while True:
    p.stepSimulation()
    time.sleep(1./240.)
