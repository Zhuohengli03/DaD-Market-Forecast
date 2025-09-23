# Darker Market Data Collection & Analysis System
# 暗黑市场数据收集与分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data collection and analysis system for Darker Market API, featuring automated data collection, machine learning predictions, time series analysis, and intelligent item management.

一个全面的暗黑市场API数据收集与分析系统，具有自动化数据收集、机器学习预测、时间序列分析和智能物品管理功能。

## 🌟 Features / 功能特性

### 🚀 Intelligent Item Management / 智能物品管理
- **Dynamic Item Configuration** / **动态物品配置**
  - JSON-based configuration system / 基于JSON的配置系统
  - Auto-discovery of new API files / 自动发现新API文件
  - Category-based organization (ore, consumable, equipment, material) / 按类别组织（矿石、消耗品、装备、材料）
  - Enable/disable items without code changes / 无需代码更改即可启用/禁用物品

### 📊 Data Collection / 数据收集
- **Automated API Data Collection** / **自动化API数据收集**
  - Real-time data fetching from Darker Market API / 从暗黑市场API实时获取数据
  - Support for multiple item types with extensible architecture / 支持多种物品类型的可扩展架构
  - Intelligent deduplication to prevent data redundancy / 智能去重防止数据冗余
  - Batch processing for efficient database operations / 批量处理提高数据库操作效率
  - Automatic stop mechanism after consecutive pages with no new data / 连续无新数据页面后自动停止机制

### 🗄️ Database Management / 数据库管理
- **PostgreSQL Integration** / **PostgreSQL集成**
  - Robust database connection management / 强大的数据库连接管理
  - Automatic table creation and schema management / 自动表创建和模式管理
  - Data integrity and duplicate prevention / 数据完整性和重复预防
  - Single connection batch processing / 单连接批量处理
  - CSV export functionality / CSV导出功能

### 🤖 Machine Learning Analysis / 机器学习分析
- **Advanced Price Prediction** / **高级价格预测**
  - Multiple algorithm ensemble (Random Forest, Gradient Boosting, Linear Models, SVR, MLP) / 多算法集成（随机森林、梯度提升、线性模型、SVR、MLP）
  - LSTM deep learning for time series / LSTM深度学习时间序列
  - ARIMA and Prophet time series models / ARIMA和Prophet时间序列模型
  - Dynamic confidence intervals / 动态置信区间
  - Feature engineering with lag features / 滞后特征工程
  - Model stability testing / 模型稳定性测试

### 📈 Time Series Analysis / 时间序列分析
- **Multi-Method Forecasting** / **多方法预测**
  - Ensemble prediction fusion / 集成预测融合
  - Outlier detection and handling / 异常值检测和处理
  - Trend analysis and visualization / 趋势分析和可视化
  - Risk assessment and investment advice / 风险评估和投资建议

### ⏰ Task Scheduling / 任务调度
- **Intelligent Scheduler** / **智能调度器**
  - Multiple scheduling options (daily, hourly, interval, weekly) / 多种调度选项（每日、每小时、间隔、每周）
  - Task management and monitoring / 任务管理和监控
  - Asynchronous execution / 异步执行
  - Enable/disable tasks dynamically / 动态启用/禁用任务

## 📁 Project Structure / 项目结构

```
Darker Market/
├── README.md                          # This file / 本文件
├── requirements.txt                   # Python dependencies / Python依赖
├── env.template                       # Environment variables template / 环境变量模板
├── items_config.json                  # Item configuration / 物品配置
├── *.csv                             # Data files / 数据文件
│
├── src/                              # Source code / 源代码
│   ├── api/                          # API modules / API模块
│   │   ├── Gold_Ore_API.py           # Gold ore data collection / 金矿数据收集
│   │   ├── Iron_Ore_API.py           # Iron ore data collection / 铁矿数据收集
│   │   └── Cobalt_Ore_API.py         # Cobalt ore data collection / 钴矿数据收集
│   │
│   ├── database/                     # Database modules / 数据库模块
│   │   ├── Database_connect.py       # Database connection / 数据库连接
│   │   └── config.py                 # Configuration management / 配置管理
│   │
│   ├── analysis/                     # Analysis modules / 分析模块
│   │   └── Machine_learning_analysis.py  # ML analysis system / 机器学习分析系统
│   │
│   └── scheduler/                    # Scheduling modules / 调度模块
│       ├── scheduler.py              # Main scheduler / 主调度器
│       ├── smart_scheduler.py        # Smart scheduler / 智能调度器
│       ├── start_scheduler.py        # Scheduler starter / 调度器启动器
│       └── task_config.py            # Task configuration / 任务配置
│
└── docs/                             # Documentation / 文档
    ├── guides/                       # User guides / 用户指南
    │   ├── QUICK_START.md            # Quick start guide / 快速开始指南
    │   ├── ENV_SETUP_GUIDE.md        # Environment setup / 环境设置
    │   └── ITEMS_CONFIG_GUIDE.md     # Item configuration guide / 物品配置指南
    │
    ├── api/                          # API documentation / API文档
    │   └── API_REFERENCE.md          # API reference / API参考
    │
    └── scheduler/                    # Scheduler documentation / 调度器文档
        └── SCHEDULER_README.md       # Scheduler guide / 调度器指南
```

## 🚀 Quick Start / 快速开始

### 1. Environment Setup / 环境设置

```bash
# Clone the repository / 克隆仓库
git clone <repository-url>
cd "Darker Market"

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Setup environment variables / 设置环境变量
cp env.template .env
# Edit .env with your credentials / 编辑.env文件添加你的凭证
```

### 2. Database Configuration / 数据库配置

Edit `.env` file with your database credentials:
编辑`.env`文件添加数据库凭证：

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# API Configuration
DARKER_MARKET_API_KEY=your_api_key
```

### 3. Run Analysis System / 运行分析系统

```bash
# Start the ML analysis system / 启动机器学习分析系统
python src/analysis/Machine_learning_analysis.py
```

**System Menu / 系统菜单:**
1. 🚀 Smart Mode (Item Selection → Data Collection → Analysis) / 智能模式（物品选择→数据收集→分析）
2. 📊 Traditional Mode (Analyze existing CSV files) / 传统模式（分析现有CSV文件）
3. 🛠️ Item Configuration Management / 物品配置管理
4. ❌ Exit / 退出

### 4. Item Management / 物品管理

The system automatically discovers API files and allows easy addition of new items:
系统自动发现API文件并允许轻松添加新物品：

**Adding New Items / 添加新物品:**
- Through configuration UI / 通过配置界面
- Auto-discovery from `src/api/` directory / 从`src/api/`目录自动发现
- Manual configuration in `items_config.json` / 在`items_config.json`中手动配置

## 📊 Usage Examples / 使用示例

### Smart Mode Analysis / 智能模式分析
```bash
python src/analysis/Machine_learning_analysis.py
# Select 1 for Smart Mode
# Choose item (Gold Ore, Iron Ore, Cobalt Ore)
# Choose whether to fetch new data
# Automatic analysis with predictions
```

### Direct API Collection / 直接API收集
```bash
python src/api/Gold_Ore_API.py
```

### Scheduled Tasks / 定时任务
```bash
python src/scheduler/smart_scheduler.py
```

## 🔧 Configuration / 配置

### Item Configuration / 物品配置
Edit `items_config.json` to manage items:
编辑`items_config.json`管理物品：

```json
{
  "items": [
    {
      "name": "Gold Ore",
      "file": "Gold_Ore_API.py",
      "csv": "gold_ore.csv",
      "category": "ore",
      "description": "Golden ore data",
      "enabled": true
    }
  ],
  "auto_discovery": {
    "enabled": true,
    "ore_directory": "src/api",
    "naming_pattern": "*_API.py"
  }
}
```

### Environment Variables / 环境变量
All sensitive information is stored in environment variables:
所有敏感信息存储在环境变量中：

- Database credentials / 数据库凭证
- API keys / API密钥
- Configuration paths / 配置路径

## 🤖 Machine Learning Features / 机器学习功能

### Models Used / 使用的模型
- **Traditional ML**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVR, Extra Trees, MLP / 传统机器学习
- **Deep Learning**: LSTM for time series / LSTM时间序列
- **Time Series**: ARIMA, Prophet / 时间序列模型
- **Ensemble**: Voting, Weighted Average / 集成方法

### Key Features / 主要功能
- **Feature Engineering**: Lag features, moving averages, volatility indicators / 特征工程
- **Model Selection**: Automatic best model selection / 自动最佳模型选择
- **Ensemble Methods**: Multiple model fusion / 多模型融合
- **Confidence Intervals**: Dynamic CI calculation / 动态置信区间计算
- **Stability Testing**: Model reliability assessment / 模型可靠性评估

### Investment Insights / 投资洞察
- **Price Predictions**: 7-day price forecasts / 7天价格预测
- **Trend Analysis**: Market direction indicators / 市场方向指标
- **Risk Assessment**: Confidence interval-based risk evaluation / 基于置信区间的风险评估
- **Investment Advice**: Buy/Sell/Hold recommendations / 买入/卖出/持有建议

## 📚 Documentation / 文档

Detailed documentation is available in the `docs/` directory:
详细文档位于`docs/`目录：

- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Get started quickly / 快速开始
- **[Environment Setup](docs/guides/ENV_SETUP_GUIDE.md)** - Detailed setup instructions / 详细设置说明
- **[Item Configuration](docs/guides/ITEMS_CONFIG_GUIDE.md)** - Item management guide / 物品管理指南
- **[API Reference](docs/api/API_REFERENCE.md)** - API documentation / API文档
- **[Scheduler Guide](docs/scheduler/SCHEDULER_README.md)** - Task scheduling / 任务调度

## 🔒 Security / 安全

- **Environment Variables**: All sensitive data in `.env` file / 所有敏感数据在`.env`文件中
- **Git Ignore**: Sensitive files excluded from version control / 敏感文件排除在版本控制外
- **Input Validation**: SQL injection prevention / SQL注入防护
- **Error Handling**: Graceful error management / 优雅的错误管理

## 🛠️ Development / 开发

### Adding New Items / 添加新物品
1. Create API file in `src/api/` following naming pattern `ItemName_API.py` / 在`src/api/`中创建API文件
2. System automatically discovers new files / 系统自动发现新文件
3. Or use configuration management UI / 或使用配置管理界面
4. Or manually edit `items_config.json` / 或手动编辑`items_config.json`

### Code Structure / 代码结构
- **Modular Design**: Separate modules for different functionalities / 模块化设计
- **Error Handling**: Comprehensive error management / 全面的错误管理
- **Logging**: Detailed logging for debugging / 详细的调试日志
- **Documentation**: Inline comments and docstrings / 内联注释和文档字符串

## 📈 Performance / 性能

- **Batch Processing**: Efficient database operations / 高效的数据库操作
- **Deduplication**: In-memory and database-level / 内存和数据库级别去重
- **Auto-stop**: Intelligent collection termination / 智能收集终止
- **Caching**: Model and data caching / 模型和数据缓存

## 🤝 Contributing / 贡献

1. Fork the repository / 分叉仓库
2. Create feature branch / 创建功能分支
3. Make changes / 进行更改
4. Test thoroughly / 彻底测试
5. Submit pull request / 提交拉取请求

## 📝 License / 许可

This project is licensed under the MIT License - see the LICENSE file for details.
本项目采用MIT许可证 - 详见LICENSE文件。

## 🆘 Support / 支持

- **Documentation**: Check `docs/` directory / 查看`docs/`目录
- **Issues**: Report bugs and feature requests / 报告错误和功能请求
- **Guides**: Step-by-step tutorials available / 提供逐步教程

## 🔄 Version History / 版本历史

- **v2.0**: Restructured project with intelligent item management / 智能物品管理的重构项目
- **v1.5**: Enhanced ML models and ensemble methods / 增强的机器学习模型和集成方法
- **v1.0**: Initial release with basic functionality / 基本功能的初始版本

---

**Ready to start? / 准备开始？**  
Run `python src/analysis/Machine_learning_analysis.py` and explore the intelligent market analysis system!  
运行 `python src/analysis/Machine_learning_analysis.py` 探索智能市场分析系统！