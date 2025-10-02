#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Data Collection Wrapper for afford_data_gen.py
Just runs the existing working script that achieves 7.5cm lifts!
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_data_collection(num_scenes=10, num_objects_range=(2, 4)):
    """
    Run the existing afford_data_gen.py script - it works perfectly!
    """
    
    print(f"🚀 Running the WORKING afford_data_gen.py script...")
    print(f"   Scenes: {num_scenes}")
    print(f"   Objects per scene: {num_objects_range[0]}-{num_objects_range[1]}")
    print("=" * 50)
    
    try:
        # Run the existing working script with correct parameters
        cmd = [
            sys.executable, 
            "src/afford_data_gen.py",
            "--num_scenes", str(num_scenes),
            "--num_objects", str(num_objects_range[0]), str(num_objects_range[1])
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        
        # Execute the script (let it run directly so you can see the output)
        result = subprocess.run(cmd, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"\n✅ Data collection completed successfully!")
            return True
        else:
            print(f"\n❌ Data collection failed with code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Data collection interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running data collection: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple wrapper for the working afford_data_gen.py')
    parser.add_argument('--scenes', type=int, default=10, help='Number of scenes to collect')
    parser.add_argument('--objects', type=int, nargs=2, default=[2, 4], help='Object count range [min max]')
    
    args = parser.parse_args()
    
    print("🎯 Simple Data Collector")
    print("   Just runs the existing WORKING afford_data_gen.py script")
    print("   (The one that achieves 7.5cm successful lifts!)")
    
    success = run_data_collection(args.scenes, tuple(args.objects))
    
    if success:
        print("\n🎉 All done! Check the data/ directory for your datasets.")
    else:
        print("\n💔 Something went wrong.")
        sys.exit(1)

if __name__ == "__main__":
    main()