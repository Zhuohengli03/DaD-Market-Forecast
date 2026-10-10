#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能定时任务调度器
基于配置文件管理定时任务
"""

import schedule
import time
import subprocess
import sys
import os
import threading
from datetime import datetime
import logging
from task_config import TASKS, SCHEDULER_CONFIG

# collectors share one API key's rate limit, so they queue instead of running together
_RUN_SLOTS = threading.BoundedSemaphore(SCHEDULER_CONFIG.get("max_concurrent_tasks", 1))

class SmartScheduler:
    def __init__(self):
        self.running_tasks = {}
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        log_file = SCHEDULER_CONFIG.get("log_file", "scheduler.log")
        log_level = getattr(logging, SCHEDULER_CONFIG.get("log_level", "INFO"))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def run_script(self, script_path, script_name="Unknown", args=None):
        """运行指定的Python脚本"""
        try:
            logging.info(f"🚀 开始运行脚本: {script_name} ({script_path})")
            
            # 检查文件是否存在
            if not os.path.exists(script_path):
                logging.error(f"❌ 脚本文件不存在: {script_path}")
                return False
            
            # 记录任务开始时间
            start_time = datetime.now()
            self.running_tasks[script_name] = start_time
            
            # 设置工作目录为项目根目录，这样模块导入才能正常工作，CSV文件也会生成到根目录
            # 从 src/scheduler/ 向上两级到达项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))  # src/scheduler/
            project_root = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
            
            # 运行脚本
            result = subprocess.run(
                [sys.executable, script_path, *(args or [])],
                capture_output=True,
                text=True,
                cwd=project_root  # 使用项目根目录作为工作目录
            )
            
            # 计算运行时间
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                logging.info(f"✅ 脚本运行成功: {script_name} (耗时: {duration:.2f}秒)")
                logging.info(f"✅ Script execution successful: {script_name} (duration: {duration:.2f}s)")
                
                # 解析并显示关键输出信息
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines[-5:]:  # 显示最后5行输出
                        if line.strip() and any(keyword in line for keyword in ["成功", "完成", "总计", "错误", "失败", "Success", "Complete", "Total", "Error", "Failed"]):
                            logging.info(f"📊 输出摘要: {line.strip()}")
            else:
                logging.error(f"❌ 脚本运行失败: {script_name} (耗时: {duration:.2f}秒, 返回码: {result.returncode})")
                logging.error(f"❌ Script execution failed: {script_name} (duration: {duration:.2f}s, return code: {result.returncode})")
                
                # 显示错误信息
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-3:]:  # 显示最后3行错误
                        if line.strip():
                            logging.error(f"🚨 错误详情: {line.strip()}")
                
                # 如果有stdout但返回码非0，也显示输出（可能包含有用信息）
                if result.stdout:
                    logging.warning(f"⚠️ 程序输出: {result.stdout.strip()[-200:]}")  # 显示最后200字符
            
            # 移除运行中的任务
            if script_name in self.running_tasks:
                del self.running_tasks[script_name]
                
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"❌ 运行脚本时出错: {script_name} - {str(e)}")
            if script_name in self.running_tasks:
                del self.running_tasks[script_name]
            return False
    
    def run_script_async(self, script_path, script_name="Unknown", args=None):
        """异步运行脚本（避免阻塞）"""
        def run():
          with _RUN_SLOTS:
            logging.info(f"🔄 异步任务开始: {script_name}")
            logging.info(f"🔄 Async task started: {script_name}")
            success = self.run_script(script_path, script_name, args)
            if success:
                logging.info(f"🎉 异步任务完成: {script_name}")
                logging.info(f"🎉 Async task completed: {script_name}")
            else:
                logging.error(f"💥 异步任务失败: {script_name}")
                logging.error(f"💥 Async task failed: {script_name}")
        
        thread = threading.Thread(target=run, name=f"AsyncTask-{script_name}", daemon=True)
        thread.start()
        logging.info(f"📋 任务已提交到后台执行: {script_name}")
        logging.info(f"📋 Task submitted to background: {script_name}")
        return thread
    
    def add_task_from_config(self, task_config):
        """从配置添加任务"""
        if not task_config.get("enabled", True):
            logging.info(f"⏸️ 任务已禁用: {task_config['name']}")
            return
        
        script_path = task_config["script_path"]
        script_name = task_config["name"]
        args = task_config.get("args", [])
        schedule_type = task_config["schedule_type"]
        schedule_value = task_config["schedule_value"]
        
        # 检查脚本文件是否存在
        if not os.path.exists(script_path):
            logging.warning(f"⚠️ 脚本文件不存在，跳过任务: {script_name} ({script_path})")
            return
        
        try:
            if schedule_type == "daily":
                schedule.every().day.at(schedule_value).do(
                    self.run_script_async, 
                    script_path=script_path, 
                    script_name=script_name,
                    args=args
                )
                logging.info(f"📅 已添加每日任务: {script_name} 在 {schedule_value}")
                
            elif schedule_type == "hourly":
                schedule.every().hour.do(
                    self.run_script_async, 
                    script_path=script_path, 
                    script_name=script_name,
                    args=args
                )
                logging.info(f"⏰ 已添加每小时任务: {script_name}")
                
            elif schedule_type == "interval":
                schedule.every(schedule_value).minutes.do(
                    self.run_script_async, 
                    script_path=script_path, 
                    script_name=script_name,
                    args=args
                )
                logging.info(f"🔄 已添加间隔任务: {script_name} 每 {schedule_value} 分钟")
                
            elif schedule_type == "weekly":
                day, time_str = schedule_value
                getattr(schedule.every(), day.lower()).at(time_str).do(
                    self.run_script_async, 
                    script_path=script_path, 
                    script_name=script_name,
                    args=args
                )
                logging.info(f"📆 已添加每周任务: {script_name} 在 {day} {time_str}")
                
            else:
                logging.error(f"❌ 未知的调度类型: {schedule_type}")
                
        except Exception as e:
            logging.error(f"❌ 添加任务失败: {script_name} - {str(e)}")
    
    def load_tasks_from_config(self):
        """从配置文件加载所有任务"""
        logging.info("📋 从配置文件加载任务...")
        
        for task_config in TASKS:
            self.add_task_from_config(task_config)
    
    def run_once(self, script_path, script_name=None):
        """立即运行一次脚本"""
        if script_name is None:
            script_name = os.path.basename(script_path)
        
        logging.info(f"▶️ 立即运行脚本: {script_name}")
        return self.run_script(script_path, script_name)
    
    def start_scheduler(self):
        """启动调度器"""
        logging.info("🕐 智能定时任务调度器启动")
        logging.info("按 Ctrl+C 停止调度器")
        
        # 显示当前任务
        self.list_tasks()
        
        # 显示运行中的任务
        if self.running_tasks:
            logging.info("🔄 当前运行中的任务:")
            for name, start_time in self.running_tasks.items():
                duration = (datetime.now() - start_time).total_seconds()
                logging.info(f"  - {name} (运行了 {duration:.0f} 秒)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(SCHEDULER_CONFIG.get("check_interval", 1))
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
            next_run = job.next_run.strftime("%Y-%m-%d %H:%M:%S") if job.next_run else "未安排"
            logging.info(f"  {i}. {job.job_func.__name__} - 下次运行: {next_run}")
    
    def get_status(self):
        """获取调度器状态"""
        jobs = schedule.get_jobs()
        running_count = len(self.running_tasks)
        
        status = {
            "total_tasks": len(jobs),
            "running_tasks": running_count,
            "next_run": None
        }
        
        if jobs:
            next_job = min(jobs, key=lambda x: x.next_run if x.next_run else datetime.max)
            status["next_run"] = next_job.next_run.strftime("%Y-%m-%d %H:%M:%S") if next_job.next_run else None
        
        return status

def main():
    """主函数"""
    scheduler = SmartScheduler()
    
    # 从配置文件加载任务
    scheduler.load_tasks_from_config()
    
    # 启动调度器
    scheduler.start_scheduler()

if __name__ == "__main__":
    main()
