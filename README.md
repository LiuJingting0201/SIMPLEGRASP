# 自监督抓取可供性（Affordance）数据收集系统

这是一个完整的自监督抓取可供性数据收集系统，用于生成训练深度学习模型所需的自标注数据集。

## 🎯 系统概述

本系统实现了方案A的核心思路：
- 在PyBullet仿真环境中自动生成自标注数据
- 对每一帧RGB-D图像采样若干抓取候选点 (u,v,θ)
- 用物理仿真"试抓"打标签（成功/失败）
- 生成像素级可供性热力图和抓取角度图

## 📁 文件结构

```
affordance_workspace/
├── src/                          # 核心模块
│   ├── environment_setup.py      # 环境和物体设置
│   ├── perception.py             # 相机感知和图像处理
│   ├── geom.py                   # 几何变换和抓取测试
│   └── afford_data_gen.py        # 原始数据生成器
├── sim_afford_data.py            # 🔥 可供性数据收集器 (主要脚本)
├── visualize_affordance.py       # 🎨 数据可视化工具
├── test_affordance_pipeline.py   # 🧪 管道测试脚本
└── data/
    └── affordance_dataset/       # 生成的数据集
        ├── scene_0000_rgb.png    # RGB图像
        ├── scene_0000_depth.npy  # 深度图像
        ├── scene_0000_affordance.npy  # 可供性热力图
        ├── scene_0000_angles.npy # 最佳抓取角度图
        └── scene_0000_meta.json  # 元数据
```

## 🚀 快速开始

### 1. 测试系统管道
```bash
python3 test_affordance_pipeline.py
```

### 2. 小规模数据收集（用于验证）
```bash
python3 sim_afford_data.py --num_scenes 10 --num_samples 30 --visualize
```

### 3. 大规模数据收集（用于训练）
```bash
python3 sim_afford_data.py --num_scenes 1000 --num_samples 50
```

### 4. 数据可视化
```bash
# 查看单个场景
python3 visualize_affordance.py --scene_id 0

# 分析整个数据集
python3 visualize_affordance.py --analyze
```

## 📊 数据格式说明

### 输入数据
- **RGB图像**: (480, 640, 3) uint8 彩色图像
- **深度图像**: (480, 640) float32 深度值（米）

### 输出标签
- **可供性热力图**: (480, 640) float32，每个像素的最佳抓取成功概率 [0-1]
- **角度图**: (480, 640) int8，每个像素的最佳抓取角度索引 [-1 to 7]
  - -1: 无有效抓取
  - 0-7: 对应0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

### 元数据
```json
{
  "scene_id": 0,
  "image_size": [640, 480],
  "num_angles": 8,
  "grasp_angles_rad": [0.0, 0.785, 1.571, ...],
  "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "num_candidates": 50,
  "success_rate": 0.34,
  "candidates": [[u, v, angle_idx], ...],
  "results": [true, false, ...]
}
```

## ⚙️ 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_scenes` | 100 | 要收集的场景数量 |
| `--num_objects` | [3, 5] | 每个场景的物体数量范围 |
| `--num_samples` | 50 | 每个场景的抓取候选点数量 |
| `--num_angles` | 8 | 离散化抓取角度数量 |
| `--visualize` | False | 启用可视化（仅第一个场景） |

## 🔬 技术细节

### 抓取候选点采样
- 在有深度值且在物体上的像素中随机采样
- 每个候选点包含像素坐标 (u,v) 和随机抓取角度
- 通过相机内参将像素坐标反投影到3D世界坐标

### 抓取仿真测试
- 使用Franka Panda机械臂进行抓取仿真
- 每个测试包含：接近→抓取→抬起→成功判断
- 成功标准：抓取后物体高度提升超过阈值

### 可供性地图生成
- 将所有测试结果映射回像素坐标
- 对每个像素，计算各角度的成功率
- 选择成功率最高的角度作为该像素的最佳抓取方向

## 🎯 推荐的数据收集策略

### 阶段1：快速验证 (5-10个场景)
```bash
python3 sim_afford_data.py --num_scenes 10 --num_samples 20 --visualize
```

### 阶段2：原型开发 (100-200个场景)
```bash
python3 sim_afford_data.py --num_scenes 200 --num_samples 50
```

### 阶段3：模型训练 (1000-2000个场景)
```bash
python3 sim_afford_data.py --num_scenes 2000 --num_samples 100
```

## 📈 性能考虑

- **单场景收集时间**: ~30-60秒（取决于采样点数量）
- **推荐并行策略**: 可以在多台机器上并行运行不同scene_id范围
- **存储需求**: 每个场景约1-2MB（RGB图像占主要空间）

## 🔧 故障排除

### 常见问题

1. **物体生成过少**
   - 检查 `--num_objects` 参数
   - 确保工作空间足够大

2. **抓取成功率过低**
   - 检查物体大小是否适合机械臂抓手
   - 调整抓取测试的判断标准

3. **可视化异常**
   - 确保安装了matplotlib和cv2
   - 检查数据文件是否完整生成

### 调试模式
```bash
python3 sim_afford_data.py --num_scenes 1 --num_samples 10 --visualize
```

## 🎨 可视化功能

系统提供丰富的可视化功能：
- 原始RGB图像
- 可供性热力图
- 最佳抓取角度图
- 热力图叠加效果
- 采样点分布（成功/失败）
- 抓取方向箭头
- 数据集统计分析

## 🔮 下一步：U-Net训练

收集到足够数据后，可以：
1. 创建PyTorch数据加载器读取这些数据
2. 设计轻量U-Net架构
3. 训练像素级分类/回归模型
4. 实现在线推理和机械臂控制

## 📚 引用和参考

这个系统为实现"自监督抓取可供性热力图"提供了完整的数据收集管道，符合现代机器人学习的最佳实践。