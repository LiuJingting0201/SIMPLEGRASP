#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试可供性数据收集管道
"""

import subprocess
import sys
from pathlib import Path

def test_affordance_data_collection():
    """测试可供性数据收集"""
    print("🧪 测试可供性数据收集管道...")
    
    try:
        # 运行一个小规模测试
        cmd = [
            sys.executable, 
            "sim_afford_data.py",
            "--num_scenes", "2",
            "--num_objects", "3", "4", 
            "--num_samples", "20",  # 较少的采样点用于快速测试
            "--num_angles", "4",    # 较少的角度用于快速测试
            "--visualize"          # 启用可视化
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("📊 测试结果:")
        print(f"返回码: {result.returncode}")
        print(f"标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"错误输出:\n{result.stderr}")
        
        # 检查数据文件是否生成
        data_dir = Path("data/affordance_dataset")
        if data_dir.exists():
            files = list(data_dir.glob("scene_*"))
            print(f"\n📁 生成的数据文件数量: {len(files)}")
            
            for file in sorted(files)[:10]:  # 只显示前10个
                print(f"   {file.name}")
        
        if result.returncode == 0:
            print("✅ 可供性数据收集测试通过！")
            
            # 运行可视化测试
            print("\n🎨 测试数据可视化...")
            viz_cmd = [sys.executable, "visualize_affordance.py", "--scene_id", "0"]
            viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=60)
            
            if viz_result.returncode == 0:
                print("✅ 可视化测试也通过！")
            else:
                print("⚠️ 可视化测试失败，但数据收集成功")
            
            return True
        else:
            print("❌ 可供性数据收集测试失败！")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def show_usage():
    """显示使用说明"""
    print("\n" + "=" * 60)
    print("🎯 可供性数据收集系统使用指南")
    print("=" * 60)
    print("""
1. 快速测试 (2个场景，用于验证管道):
   python3 sim_afford_data.py --num_scenes 2 --num_samples 20 --visualize

2. 小规模数据收集 (50个场景):
   python3 sim_afford_data.py --num_scenes 50 --num_samples 50

3. 大规模数据收集 (1000个场景，用于训练):
   python3 sim_afford_data.py --num_scenes 1000 --num_samples 100

4. 可视化收集的数据:
   python3 visualize_affordance.py --scene_id 0
   python3 visualize_affordance.py --analyze

5. 数据集分析:
   python3 visualize_affordance.py --analyze

参数说明:
   --num_scenes: 要收集的场景数量
   --num_objects: 每个场景的物体数量范围 [min, max]
   --num_samples: 每个场景的抓取候选点数量
   --num_angles: 离散化抓取角度数量 (推荐8个，即每45度一个)
   --visualize: 启用可视化 (仅对第一个场景)

数据格式:
   - scene_XXXX_rgb.png: RGB图像
   - scene_XXXX_depth.npy: 深度图像
   - scene_XXXX_affordance.npy: 可供性热力图 (每个像素的最佳抓取成功概率)
   - scene_XXXX_angles.npy: 角度图 (每个像素的最佳抓取角度索引)
   - scene_XXXX_meta.json: 元数据 (相机参数、采样点、结果等)

推荐的数据收集策略:
   1. 先运行小规模测试确保管道正常工作
   2. 收集100-200个场景用于原型开发
   3. 收集1000-2000个场景用于训练U-Net模型
   4. 每个场景50-100个采样点可以得到良好的热力图质量
    """)
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        success = test_affordance_data_collection()
        if success:
            show_usage()
        sys.exit(0 if success else 1)