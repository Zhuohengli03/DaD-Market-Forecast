#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时任务调度器
支持定时运行指定的Python文件
"""

import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

class TaskScheduler:
    def __init__(self):
        self.running_tasks = {}
        
    def run_script(self, script_path, script_name="Unknown"):
        """运行指定的Python脚本"""
        try:
            logging.info(f"🚀 开始运行脚本: {script_name} ({script_path})")
            
            # 检查文件是否存在
            if not os.path.exists(script_path):
                logging.error(f"❌ 脚本文件不存在: {script_path}")
                return False
            
            # 运行脚本
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(script_path))
            )
            
            if result.returncode == 0:
                logging.info(f"✅ 脚本运行成功: {script_name}")
                if result.stdout:
                    logging.info(f"输出: {result.stdout}")
            else:
                logging.error(f"❌ 脚本运行失败: {script_name}")
                logging.error(f"错误: {result.stderr}")
                
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"❌ 运行脚本时出错: {script_name} - {str(e)}")
            return False
    
    def add_daily_task(self, script_path, time_str, script_name=None):
        """添加每日定时任务"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        schedule.every().day.at(time_str).do(
            self.run_script, 
            script_path=script_path, 
            script_name=script_name
        )
        logging.info(f"📅 已添加每日任务: {script_name} 在 {time_str}")
    
    def add_hourly_task(self, script_path, script_name=None):
        """添加每小时定时任务"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        schedule.every().hour.do(
            self.run_script, 
            script_path=script_path, 
            script_name=script_name
        )
        logging.info(f"⏰ 已添加每小时任务: {script_name}")
    
    def add_interval_task(self, script_path, minutes, script_name=None):
        """添加间隔定时任务（分钟）"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        schedule.every(minutes).minutes.do(
            self.run_script, 
            script_path=script_path, 
            script_name=script_name
        )
        logging.info(f"🔄 已添加间隔任务: {script_name} 每 {minutes} 分钟")
    
    def add_weekly_task(self, script_path, day, time_str, script_name=None):
        """添加每周定时任务"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        getattr(schedule.every(), day.lower()).at(time_str).do(
            self.run_script, 
            script_path=script_path, 
            script_name=script_name
        )
        logging.info(f"📆 已添加每周任务: {script_name} 在 {day} {time_str}")
    
    def run_once(self, script_path, script_name=None):
        """立即运行一次脚本"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        logging.info(f"▶️ 立即运行脚本: {script_name}")
        return self.run_script(script_path, script_name)
    
    def start_scheduler(self):
        """启动调度器"""
        logging.info("🕐 定时任务调度器启动")
        logging.info("按 Ctrl+C 停止调度器")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 调度器已停止")
    
    def list_tasks(self):
        """列出所有任务"""
        jobs = schedule.get_jobs()
        if not jobs:
            logging.info("📝 当前没有定时任务")
            return
        
        logging.info("📝 当前定时任务:")
        for i, job in enumerate(jobs, 1):
            logging.info(f"  {i}. {job.job_func.__name__} - {job.next_run}")

def main():
    """主函数 - 示例用法"""
    scheduler = TaskScheduler()
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 示例：添加各种定时任务
    # 注意：请根据实际文件路径修改
    
    # 1. 每日运行 Iron Ore API (早上8点)
    iron_ore_script = os.path.join(current_dir, "Ore", "Iron_Ore_API.py")
    if os.path.exists(iron_ore_script):
        scheduler.add_daily_task(iron_ore_script, "08:00", "Iron Ore API")
    
    # 2. 每日运行 Gold Ore API (晚上8点)
    gold_ore_script = os.path.join(current_dir, "Ore", "Gold_Ore_API.py")
    if os.path.exists(gold_ore_script):
        scheduler.add_daily_task(gold_ore_script, "20:00", "Gold Ore API")
    
    # 3. 每小时运行一次 (如果需要)
    # scheduler.add_hourly_task(iron_ore_script, "Iron Ore API")
    
    # 4. 每30分钟运行一次 (如果需要)
    # scheduler.add_interval_task(iron_ore_script, 30, "Iron Ore API")
    
    # 5. 每周运行一次 (周一早上9点)
    # scheduler.add_weekly_task(iron_ore_script, "monday", "09:00", "Iron Ore API")
    
    # 显示当前任务
    scheduler.list_tasks()
    
    # 启动调度器
    scheduler.start_scheduler()

if __name__ == "__main__":
    main()
