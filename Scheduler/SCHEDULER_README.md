# 定时任务调度器使用说明

## 文件说明

- `scheduler.py` - 基础定时任务调度器
- `smart_scheduler.py` - 智能定时任务调度器（推荐使用）
- `task_config.py` - 任务配置文件
- `start_scheduler.py` - 启动脚本
- `scheduler.log` - 运行日志文件

## 快速开始

### 1. 安装依赖

```bash
pip install schedule
```

### 2. 配置任务

编辑 `task_config.py` 文件，添加或修改需要定时运行的任务：

```python
TASKS = [
    {
        "name": "Iron Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Iron_Ore_API.py"),
        "schedule_type": "daily",  # 每日
        "schedule_value": "08:00",  # 早上8点
        "enabled": True,
        "description": "每日早上8点运行Iron Ore数据收集"
    },
    # 添加更多任务...
]
```

### 3. 启动调度器

```bash
# 启动调度器
python start_scheduler.py --start

# 或者直接运行
python smart_scheduler.py
```

## 使用方法

### 命令行选项

```bash
# 列出所有任务
python start_scheduler.py --list

# 显示状态
python start_scheduler.py --status

# 立即运行指定脚本
python start_scheduler.py --run "Ore/Iron_Ore_API.py"

# 启动调度器
python start_scheduler.py --start
```

### 任务配置

在 `task_config.py` 中配置任务：

#### 每日任务
```python
{
    "name": "任务名称",
    "script_path": "脚本路径",
    "schedule_type": "daily",
    "schedule_value": "08:00",  # 时间格式 HH:MM
    "enabled": True
}
```

#### 每小时任务
```python
{
    "name": "任务名称",
    "script_path": "脚本路径",
    "schedule_type": "hourly",
    "schedule_value": None,  # 不需要值
    "enabled": True
}
```

#### 间隔任务
```python
{
    "name": "任务名称",
    "script_path": "脚本路径",
    "schedule_type": "interval",
    "schedule_value": 30,  # 每30分钟
    "enabled": True
}
```

#### 每周任务
```python
{
    "name": "任务名称",
    "script_path": "脚本路径",
    "schedule_type": "weekly",
    "schedule_value": ("monday", "09:00"),  # 周一早上9点
    "enabled": True
}
```

## 功能特性

- ✅ 支持多种调度类型（每日、每小时、间隔、每周）
- ✅ 配置文件管理任务
- ✅ 异步执行，避免任务阻塞
- ✅ 详细的运行日志
- ✅ 任务状态监控
- ✅ 命令行界面
- ✅ 错误处理和恢复

## 日志查看

所有运行日志保存在 `scheduler.log` 文件中：

```bash
# 查看最新日志
tail -f scheduler.log

# 查看所有日志
cat scheduler.log
```

## 注意事项

1. 确保要运行的脚本文件存在且可执行
2. 脚本路径使用绝对路径或相对于项目根目录的路径
3. 调度器会持续运行，按 Ctrl+C 停止
4. 任务运行时会记录详细的日志信息
5. 支持并发执行多个任务

## 示例配置

```python
TASKS = [
    # Iron Ore - 每日早上8点
    {
        "name": "Iron Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Iron_Ore_API.py"),
        "schedule_type": "daily",
        "schedule_value": "08:00",
        "enabled": True,
        "description": "每日早上8点运行Iron Ore数据收集"
    },
    
    # Gold Ore - 每日晚上8点
    {
        "name": "Gold Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Gold_Ore_API.py"),
        "schedule_type": "daily",
        "schedule_value": "20:00",
        "enabled": True,
        "description": "每日晚上8点运行Gold Ore数据收集"
    },
    
    # 机器学习分析 - 每周一早上9点
    {
        "name": "ML Analysis",
        "script_path": os.path.join(CURRENT_DIR, "Machine_learning_analysis.py"),
        "schedule_type": "weekly",
        "schedule_value": ("monday", "09:00"),
        "enabled": False,
        "description": "每周一早上9点运行机器学习分析"
    }
]
```
