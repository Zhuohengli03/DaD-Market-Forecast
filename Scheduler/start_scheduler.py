#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时任务启动脚本
提供简单的命令行界面来管理定时任务
"""

import os
import sys
import argparse
from smart_scheduler import SmartScheduler

def main():
    parser = argparse.ArgumentParser(description='定时任务调度器')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有任务')
    parser.add_argument('--status', '-s', action='store_true', help='显示状态')
    parser.add_argument('--run', '-r', type=str, help='立即运行指定脚本')
    parser.add_argument('--start', action='store_true', help='启动调度器')
    
    args = parser.parse_args()
    
    scheduler = SmartScheduler()
    
    if args.list:
        scheduler.load_tasks_from_config()
        scheduler.list_tasks()
        
    elif args.status:
        scheduler.load_tasks_from_config()
        status = scheduler.get_status()
        print(f"📊 调度器状态:")
        print(f"  总任务数: {status['total_tasks']}")
        print(f"  运行中任务: {status['running_tasks']}")
        print(f"  下次运行: {status['next_run'] or '无'}")
        
    elif args.run:
        script_path = args.run
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(script_path):
            script_path = os.path.join(os.getcwd(), script_path)
        
        if not os.path.exists(script_path):
            print(f"❌ 脚本文件不存在: {script_path}")
            return
        
        script_name = os.path.basename(script_path)
        print(f"▶️ 立即运行脚本: {script_name}")
        success = scheduler.run_script(script_path, script_name)
        if success:
            print("✅ 脚本运行完成")
        else:
            print("❌ 脚本运行失败")
            
    elif args.start:
        scheduler.load_tasks_from_config()
        scheduler.start_scheduler()
        
    else:
        # 默认启动调度器
        scheduler.load_tasks_from_config()
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()
