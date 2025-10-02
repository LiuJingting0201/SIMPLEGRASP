#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的数据生成器
"""

import subprocess
import sys

def test_data_generation():
    """测试数据生成器是否正常工作"""
    print("🧪 测试修复后的数据生成器...")
    
    try:
        # 运行一个简单的测试
        result = subprocess.run([
            sys.executable, 
            "src/afford_data_gen.py",
            "--num_scenes", "2",
            "--num_objects", "3", "5",
            "--visualize_first"
        ], capture_output=True, text=True, timeout=60)
        
        print("📊 测试结果:")
        print(f"返回码: {result.returncode}")
        print(f"标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"错误输出:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✅ 测试通过！数据生成器工作正常")
            return True
        else:
            print("❌ 测试失败！仍有问题需要修复")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时，可能卡住了")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    success = test_data_generation()
    sys.exit(0 if success else 1)
