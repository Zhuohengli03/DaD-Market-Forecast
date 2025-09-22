# Darker Market Data Collection & Analysis System
# 暗黑市场数据收集与分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data collection and analysis system for Darker Market API, featuring automated data collection, time series analysis, machine learning predictions, and scheduled task management.

一个全面的暗黑市场API数据收集与分析系统，具有自动化数据收集、时间序列分析、机器学习预测和定时任务管理功能。

## 🌟 Features / 功能特性

### 📊 Data Collection / 数据收集
- **Automated API Data Collection** / **自动化API数据收集**
  - Real-time data fetching from Darker Market API / 从暗黑市场API实时获取数据
  - Support for multiple ore types (Iron, Gold, Cobalt) / 支持多种矿石类型（铁、金、钴）
  - Intelligent deduplication to prevent data redundancy / 智能去重防止数据冗余
  - Batch processing for efficient database operations / 批量处理提高数据库操作效率

### 🗄️ Database Management / 数据库管理
- **PostgreSQL Integration** / **PostgreSQL集成**
  - Robust database connection management / 强大的数据库连接管理
  - Automatic table creation and schema management / 自动表创建和模式管理
  - Data integrity and duplicate prevention / 数据完整性和重复预防
  - CSV export functionality / CSV导出功能

### 📈 Time Series Analysis / 时间序列分析
- **ARIMA Forecasting** / **ARIMA预测**
  - Advanced time series analysis / 高级时间序列分析
  - Price trend prediction / 价格趋势预测
  - Statistical analysis and visualization / 统计分析和可视化
  - Data quality assessment / 数据质量评估

### 🤖 Machine Learning / 机器学习
- **Price Prediction Models** / **价格预测模型**
  - Multiple ML algorithms (Random Forest, Gradient Boosting, SVR, etc.) / 多种ML算法（随机森林、梯度提升、SVR等）
  - Feature engineering and preprocessing / 特征工程和预处理
  - Model evaluation and hyperparameter tuning / 模型评估和超参数调优
  - Confidence intervals for predictions / 预测置信区间

### ⏰ Task Scheduling / 任务调度
- **Automated Task Management** / **自动化任务管理**
  - Flexible scheduling (daily, hourly, interval, weekly) / 灵活调度（每日、每小时、间隔、每周）
  - Configuration-based task management / 基于配置的任务管理
  - Real-time monitoring and logging / 实时监控和日志记录
  - Command-line interface / 命令行界面

## 🚀 Quick Start / 快速开始

### Prerequisites / 前置要求

```bash
# Python 3.8+
# PostgreSQL 12+
# Required packages / 所需包
pip install -r requirements.txt
```

### Environment Setup / 环境设置

1. **Copy environment template** / **复制环境模板**
```bash
cp env.template .env
```

2. **Configure your settings** / **配置你的设置**
Edit `.env` file with your actual values:
编辑`.env`文件为你的实际值：

```env
# Database Configuration / 数据库配置
DB_HOST=localhost
DB_DATABASE=darkerdb
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432

# API Configuration / API配置
DARKER_MARKET_API_KEY=your_api_key_here

# File Paths / 文件路径
DATA_DIR=/path/to/your/data/directory
```

**⚠️ Security Note / 安全提示**: The `.env` file is already included in `.gitignore` to prevent accidental commits of sensitive information.
**⚠️ 安全提示**: `.env`文件已包含在`.gitignore`中，以防止意外提交敏感信息。

### Installation / 安装

1. **Clone the repository** / **克隆仓库**
```bash
git clone <repository-url>
cd "Darker Market"
```

2. **Install dependencies** / **安装依赖**
```bash
pip install -r requirements.txt
```

3. **Configure database** / **配置数据库**
   - Set up PostgreSQL database / 设置PostgreSQL数据库
   - Create new file `.env` / 创建env环境文件
   - Update connection settings in `.env` / 在`.env`中更新连接设置
   - more detail check out `ENV_SETUP_GUIDE.md`

4. **Configure tasks** / **配置任务**
   - Edit `Scheduler/task_config.py` to customize scheduled tasks / 编辑`Scheduler/task_config.py`自定义定时任务

## 📁 Project Structure / 项目结构

```
Darker Market/
├── 📁 Ore/                          # API数据收集模块
│   ├── Iron_Ore_API.py             # 铁矿石API
│   ├── Gold_Ore_API.py             # 金矿石API
│   └── Cobalt_Ore_API.py           # 钴矿石API
├── 📁 Scheduler/                    # 任务调度模块
│   ├── smart_scheduler.py          # 智能调度器
│   ├── start_scheduler.py          # 启动脚本
│   ├── task_config.py              # 任务配置
│   └── scheduler.log               # 调度日志
├── 📁 Analysis/                     # 分析模块
│   └── Machine_learning_analysis.py # 机器学习分析
├── 📄 Database_connect.py           # 数据库连接
├── 📄 ARIMA_analysis.py            # ARIMA时间序列分析
├── 📄 requirements.txt              # 依赖包列表
└── 📄 README.md                    # 项目说明
```

## 🎯 Usage / 使用方法

### Data Collection / 数据收集

#### Manual Collection / 手动收集
```bash
# Run individual API collectors / 运行单个API收集器
python Ore/Iron_Ore_API.py
python Ore/Gold_Ore_API.py
python Ore/Cobalt_Ore_API.py
```

#### Scheduled Collection / 定时收集
```bash
# Start the scheduler / 启动调度器
cd Scheduler
python start_scheduler.py --start

# Check status / 查看状态
python start_scheduler.py --status

# List tasks / 列出任务
python start_scheduler.py --list

# Run specific task immediately / 立即运行特定任务
python start_scheduler.py --run "../Ore/Iron_Ore_API.py"
```

### Data Analysis / 数据分析

#### Time Series Analysis / 时间序列分析
```bash
# Run ARIMA analysis / 运行ARIMA分析
python ARIMA_analysis.py
```

#### Machine Learning Analysis / 机器学习分析
```bash
# Run ML analysis / 运行ML分析
python Analysis/Machine_learning_analysis.py
```

## ⚙️ Configuration / 配置

### Task Scheduling / 任务调度配置

Edit `Scheduler/task_config.py` to customize your scheduled tasks:

编辑`Scheduler/task_config.py`来自定义定时任务：

```python
TASKS = [
    {
        "name": "Iron Ore API",
        "script_path": os.path.join(CURRENT_DIR, "Ore", "Iron_Ore_API.py"),
        "schedule_type": "daily",      # daily, hourly, interval, weekly
        "schedule_value": "23:00",     # Time or interval
        "enabled": True,
        "description": "Daily Iron Ore data collection at 23:00"
    },
    # Add more tasks...
]
```

### Database Configuration / 数据库配置

Update database settings in `Database_connect.py`:

在`Database_connect.py`中更新数据库设置：

```python
# Database connection settings
DB_CONFIG = {
    'host': 'localhost',
    'database': 'darker_market',
    'user': 'your_username',
    'password': 'your_password',
    'port': 5432
}
```

## 📊 Data Output / 数据输出

### CSV Files / CSV文件
- `iron_ore.csv` - Iron ore market data / 铁矿石市场数据
- `gold_ore.csv` - Gold ore market data / 金矿石市场数据
- `cobalt_ore.csv` - Cobalt ore market data / 钴矿石市场数据

### Database Tables / 数据库表
- `iron_ore` - Iron ore data table / 铁矿石数据表
- `gold_ore` - Gold ore data table / 金矿石数据表
- `cobalt_ore` - Cobalt ore data table / 钴矿石数据表

### Logs / 日志
- `scheduler.log` - Task execution logs / 任务执行日志
- Console output with detailed progress information / 控制台输出包含详细进度信息

## 🔧 Advanced Features / 高级功能

### Intelligent Deduplication / 智能去重
- API-level deduplication during data collection / 数据收集期间的API级去重
- Database-level duplicate prevention / 数据库级重复预防
- Efficient memory usage with set-based operations / 基于集合操作的高效内存使用

### Error Handling / 错误处理
- Comprehensive error logging / 全面的错误日志记录
- Automatic retry mechanisms / 自动重试机制
- Graceful failure handling / 优雅的故障处理

### Performance Optimization / 性能优化
- Batch database operations / 批量数据库操作
- Asynchronous task execution / 异步任务执行
- Memory-efficient data processing / 内存高效的数据处理

## 📈 Analysis Capabilities / 分析能力

### Time Series Analysis / 时间序列分析
- **ARIMA Modeling** / **ARIMA建模**
  - Automatic parameter optimization / 自动参数优化
  - Stationarity testing / 平稳性测试
  - Trend and seasonality analysis / 趋势和季节性分析

### Machine Learning / 机器学习
- **Multiple Algorithms** / **多种算法**
  - Random Forest Regression / 随机森林回归
  - Gradient Boosting / 梯度提升
  - Support Vector Regression / 支持向量回归
  - Linear Regression variants / 线性回归变体

- **Feature Engineering** / **特征工程**
  - Time-based features / 基于时间的特征
  - Moving averages / 移动平均
  - Volatility indicators / 波动性指标
  - Price change patterns / 价格变化模式

## 🚨 Troubleshooting / 故障排除

### Common Issues / 常见问题

#### Database Connection Issues / 数据库连接问题
```bash
# Check PostgreSQL service / 检查PostgreSQL服务
sudo service postgresql status

# Verify connection settings / 验证连接设置
python -c "from Database_connect import DarkerMarketDB; db = DarkerMarketDB(); print(db.connect())"
```

#### Module Import Errors / 模块导入错误
```bash
# Ensure you're in the correct directory / 确保在正确的目录
cd "Darker Market"

# Check Python path / 检查Python路径
python -c "import sys; print(sys.path)"
```

#### Task Scheduling Issues / 任务调度问题
```bash
# Check task configuration / 检查任务配置
cd Scheduler
python start_scheduler.py --list

# Verify file paths / 验证文件路径
python start_scheduler.py --run "../Ore/Iron_Ore_API.py"
```

## 📝 Logs and Monitoring / 日志和监控

### Log Files / 日志文件
- **scheduler.log** - Task execution logs / 任务执行日志
- **Console output** - Real-time progress information / 控制台输出 - 实时进度信息

### Monitoring Commands / 监控命令
```bash
# View recent logs / 查看最近日志
tail -f scheduler.log

# Check task status / 检查任务状态
python start_scheduler.py --status

# View all scheduled tasks / 查看所有定时任务
python start_scheduler.py --list
```

## 🤝 Contributing / 贡献

1. Fork the repository / 分叉仓库
2. Create a feature branch / 创建功能分支
3. Make your changes / 进行更改
4. Add tests if applicable / 如适用则添加测试
5. Submit a pull request / 提交拉取请求

## 📄 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 📞 Support / 支持

For questions, issues, or contributions, please:

如有问题、问题或贡献，请：

- Open an issue on GitHub / 在GitHub上提出问题
- Contact the maintainers / 联系维护者
- Check the documentation / 查看文档

## 🔄 Version History / 版本历史

### v1.0.0
- Initial release / 初始版本
- Basic data collection functionality / 基本数据收集功能
- ARIMA time series analysis / ARIMA时间序列分析
- Machine learning price prediction / 机器学习价格预测
- Task scheduling system / 任务调度系统

---

**Happy Data Mining! / 祝数据挖掘愉快！** 🚀📊
