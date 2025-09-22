#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时任务配置文件
在这里配置需要定时运行的任务
"""

import os

# 获取项目根目录（Scheduler的上级目录）
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 任务配置
TASKS = [
    {
        "name": "Iron Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Iron_Ore_API.py"),
        "schedule_type": "daily",  # daily, hourly, interval, weekly
        "schedule_value": "23:00",  # 时间或间隔
        "enabled": True,
        "description": "每日晚上23点运行Iron Ore数据收集"
    },
    {
        "name": "Gold Ore API", 
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Gold_Ore_API.py"),
        "schedule_type": "daily",
        "schedule_value": "23:00",
        "enabled": True,
        "description": "每日晚上23点运行Gold Ore数据收集"
    },
    {
        "name": "Cobalt Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Cobalt_Ore_API.py"),
        "schedule_type": "daily",
        "schedule_value": "23:00",
        "enabled": True,  # 已启用
        "description": "每日晚上23点运行Cobalt Ore数据收集"
    },
    # 可以添加更多任务...
    # {
    #     "name": "Machine Learning Analysis",
    #     "script_path": os.path.join(CURRENT_DIR, "Machine_learning_analysis.py"),
    #     "schedule_type": "weekly",
    #     "schedule_value": ("monday", "09:00"),
    #     "enabled": False,
    #     "description": "每周一早上9点运行机器学习分析"
    # },
    # {
    #     "name": "Data Backup",
    #     "script_path": os.path.join(CURRENT_DIR, "backup_data.py"),
    #     "schedule_type": "interval",
    #     "schedule_value": 60,  # 每60分钟
    #     "enabled": False,
    #     "description": "每60分钟备份一次数据"
    # }
]

# 调度器配置
SCHEDULER_CONFIG = {
    "log_file": "scheduler.log",
    "log_level": "INFO",
    "check_interval": 1,  # 检查间隔（秒）
    "max_concurrent_tasks": 3,  # 最大并发任务数
}
