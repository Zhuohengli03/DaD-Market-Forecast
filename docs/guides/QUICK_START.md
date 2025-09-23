# Quick Start Guide / 快速开始指南

## 🚀 5-Minute Setup / 5分钟设置

### Step 1: Install Dependencies / 步骤1：安装依赖
```bash
pip install -r requirements.txt
```

### Step 2: Configure Database / 步骤2：配置数据库
Edit `Database_connect.py` and update your PostgreSQL connection settings:
编辑`Database_connect.py`并更新你的PostgreSQL连接设置：

```python
# Update these settings / 更新这些设置
DB_CONFIG = {
    'host': 'localhost',
    'database': 'darker_market',
    'user': 'your_username',
    'password': 'your_password',
    'port': 5432
}
```

### Step 3: Test Data Collection / 步骤3：测试数据收集
```bash
# Test Iron Ore collection / 测试铁矿石收集
python Ore/Iron_Ore_API.py
```

### Step 4: Start Task Scheduler / 步骤4：启动任务调度器
```bash
cd Scheduler
python start_scheduler.py --start
```

### Step 5: Run Analysis / 步骤5：运行分析
```bash
# Time series analysis / 时间序列分析
python ARIMA_analysis.py

# Machine learning analysis / 机器学习分析
python Analysis/Machine_learning_analysis.py
```
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
———————————————————————————————————————— 📋 Common Commands / 常用命令————————————————————————————————————————————-————

### Data Collection / 数据收集
```bash
# Manual collection / 手动收集
python Ore/Iron_Ore_API.py
python Ore/Gold_Ore_API.py
python Ore/Cobalt_Ore_API.py
```

### Task Management / 任务管理
```bash
cd Scheduler

# Check status / 查看状态
python start_scheduler.py --status

# List tasks / 列出任务
python start_scheduler.py --list

# Run specific task / 运行特定任务
python start_scheduler.py --run "../Ore/Iron_Ore_API.py"

# Start scheduler / 启动调度器
python start_scheduler.py --start
```

### Data Analysis / 数据分析
```bash
# ARIMA analysis / ARIMA分析
python ARIMA_analysis.py

# Machine learning / 机器学习
python Analysis/Machine_learning_analysis.py
```

## ⚙️ Configuration / 配置

### Schedule Tasks / 调度任务
Edit `Scheduler/task_config.py`:
编辑`Scheduler/task_config.py`：

```python
TASKS = [
    {
        "name": "Iron Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Iron_Ore_API.py"),
        "schedule_type": "daily",
        "schedule_value": "23:00",
        "enabled": True,
        "description": "Daily Iron Ore data collection at 23:00"
    }
]
```

### Schedule Types / 调度类型
- `"daily"` - Daily at specific time / 每日特定时间
- `"hourly"` - Every hour / 每小时
- `"interval"` - Every N minutes / 每N分钟
- `"weekly"` - Weekly on specific day / 每周特定日期

## 🔍 Troubleshooting / 故障排除

### Database Issues / 数据库问题
```bash
# Check PostgreSQL / 检查PostgreSQL
sudo service postgresql status

# Test connection / 测试连接
python -c "from Database_connect import DarkerMarketDB; print('OK' if DarkerMarketDB().connect() else 'Failed')"
```

### Import Errors / 导入错误
```bash
# Make sure you're in the right directory / 确保在正确目录
pwd
# Should show: /path/to/Darker Market

# Check Python path / 检查Python路径
python -c "import sys; print('\\n'.join(sys.path))"
```

### Task Scheduler Issues / 任务调度问题
```bash
# Check if files exist / 检查文件是否存在
ls -la Ore/
ls -la Scheduler/

# Test individual components / 测试单个组件
python start_scheduler.py --run "../Ore/Iron_Ore_API.py"
```

## 📊 Expected Output / 预期输出

### Successful Data Collection / 成功的数据收集
```
🔌 连接数据库...
✅ 成功连接到PostgreSQL数据库
📚 已加载 5028 条现有记录到去重集合
去重后: 5 条新数据 (跳过 45 条重复数据)
✅ 第1页: 收集到 5 条数据
💾 开始统一插入数据到数据库...
✅ 批量插入完成: 新增 5 条数据
📊 本次运行总计新增数据: 5 条
```

### Successful Task Scheduling / 成功的任务调度
```
📋 从配置文件加载任务...
📅 已添加每日任务: Iron Ore API 在 23:00
📅 已添加每日任务: Gold Ore API 在 23:00
📅 已添加每日任务: Cobalt Ore API 在 23:00
🕐 智能定时任务调度器启动
📝 当前定时任务:
  1. run_script_async - 下次运行: 2025-09-22 23:00:00
  2. run_script_async - 下次运行: 2025-09-22 23:00:00
  3. run_script_async - 下次运行: 2025-09-22 23:00:00
```

## 🎯 Next Steps / 下一步

1. **Customize Schedule** / **自定义调度** - Modify `task_config.py` for your needs
2. **Add More Analysis** / **添加更多分析** - Extend the analysis modules
3. **Monitor Logs** / **监控日志** - Check `scheduler.log` for issues
4. **Scale Up** / **扩展** - Add more ore types or analysis methods

---

**Need Help? / 需要帮助？** Check the full [README.md](README.md) for detailed documentation.
