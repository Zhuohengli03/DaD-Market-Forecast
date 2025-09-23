# 环境变量设置指南 / Environment Variables Setup Guide

## 🔐 安全配置完成

你的项目已经配置了环境变量系统来保护敏感信息！

## 📋 快速设置

### 1. 复制环境变量模板
```bash
cp env.example .env
cp config.example.py config.py
```

### 2. 编辑 .env 文件
```bash
# 使用你喜欢的编辑器
nano .env
# 或者
code .env
```

### 3. 填入你的实际配置
```env
# Database Configuration / 数据库配置
DB_HOST=localhost
DB_DATABASE=darkerdb
DB_USER=your_actual_username
DB_PASSWORD=your_actual_password
DB_PORT=5432

# API Configuration / API配置
DARKER_MARKET_API_KEY=your_actual_api_key

# File Paths / 文件路径
DATA_DIR=/Users/zhuohengli/Cursor/darkerdb/data
```

## ✅ 验证配置

运行以下命令验证配置是否正确：

```bash
python -c "from config import config; config.validate_config(); print('✅ 配置验证成功')"
```

## 🔧 当前配置状态

- ✅ **数据库配置**: 从环境变量加载
- ✅ **API密钥**: 从环境变量加载  
- ✅ **文件路径**: 从环境变量加载
- ✅ **安全保护**: .env文件已添加到.gitignore

## 📁 相关文件

- `env.example` - 环境变量模板（可提交到Git）
- `.env` - 实际环境变量文件（已忽略，不会提交）
- `config.py` - 配置加载模块
- `.gitignore` - 已更新，保护敏感文件

## 🚀 使用方法

### 在代码中使用
```python
from config import config

# 获取数据库配置
db_config = config.get_db_config()

# 获取API请求头
headers = config.get_api_headers()

# 验证配置
config.validate_config()
```

### 环境变量优先级
1. 系统环境变量（最高）
2. .env文件中的变量
3. 代码中的默认值（最低）

## ⚠️ 重要提示

- **永远不要**将`.env`文件提交到Git
- **定期更新**API密钥和密码
- **使用强密码**和安全的API密钥
- **在团队中**通过安全渠道分享配置

## 🆘 故障排除

### 如果遇到导入错误
```bash
# 确保安装了python-dotenv
pip install python-dotenv

# 检查是否在正确目录
pwd
# 应该显示: /path/to/Darker Market

# 测试配置
python -c "from config import config; print('OK')"
```

### 如果数据库连接失败
```bash
# 检查.env文件是否存在
ls -la .env

# 检查数据库配置
python -c "from config import config; print(config.get_db_config())"
```

现在你的敏感信息已经安全地存储在环境变量中，可以安全地提交代码到GitHub了！🎉
