# API Reference / API参考文档

## 📋 Overview / 概述

This document provides detailed information about the Darker Market API integration modules.

本文档提供了暗黑市场API集成模块的详细信息。

## 🔗 API Modules / API模块

### Ore APIs / 矿石APIs

#### Gold Ore API
- **File**: `src/api/Gold_Ore_API.py`
- **Description**: Collects gold ore market data
- **Item ID**: Specific to gold ore
- **Output**: `gold_ore.csv`

#### Iron Ore API  
- **File**: `src/api/Iron_Ore_API.py`
- **Description**: Collects iron ore market data
- **Item ID**: Specific to iron ore
- **Output**: `iron_ore.csv`

#### Cobalt Ore API
- **File**: `src/api/Cobalt_Ore_API.py`
- **Description**: Collects cobalt ore market data
- **Item ID**: Specific to cobalt ore
- **Output**: `cobalt_ore.csv`

## 🔧 Configuration / 配置

### Items Configuration
- **File**: `items_config.json`
- **Purpose**: Manages item definitions and API mappings
- **Auto-discovery**: Automatically detects new API files

### Environment Variables
- **File**: `.env`
- **Template**: `env.template`
- **Required Variables**:
  - `DARKER_MARKET_API_KEY`
  - `DB_HOST`, `DB_PORT`, `DB_NAME`
  - `DB_USER`, `DB_PASSWORD`

## 📊 Data Structure / 数据结构

### CSV Output Format
```csv
timestamp,item_id,quantity,price_per_unit,seller,price,location
2023-09-24 10:30:15,item123,100,25.50,seller456,2550,city789
```

### Database Schema
```sql
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    item_id VARCHAR(50),
    quantity INTEGER,
    price_per_unit DECIMAL(10,2),
    seller VARCHAR(100),
    price DECIMAL(10,2),
    location VARCHAR(100)
);
```

## 🚀 Usage Examples / 使用示例

### Direct API Call
```bash
python src/api/Gold_Ore_API.py
```

### Through Analysis System
```python
from src.analysis.Machine_learning_analysis import collect_and_analyze
collect_and_analyze()
```

### Programmatic Item Addition
```python
from src.analysis.Machine_learning_analysis import add_new_item

add_new_item(
    name="Silver Ore",
    api_file="Silver_Ore_API.py",
    csv_file="silver_ore.csv",
    category="ore",
    description="Silver ore market data"
)
```

## 🔍 Auto-Discovery / 自动发现

The system automatically discovers new API files following this pattern:
系统自动发现遵循此模式的新API文件：

- **Pattern**: `*_API.py`
- **Location**: `src/api/` directory
- **Naming**: `ItemName_API.py` → `Item Name` + `itemname.csv`

## ⚡ Performance / 性能

### Batch Processing
- Collects data in batches for efficiency
- Single database connection per run
- Automatic duplicate detection

### Rate Limiting
- Respects API rate limits
- Configurable delay between requests
- Error handling and retry logic

## 🔐 Security / 安全

### API Key Management
- Stored in environment variables
- Never committed to version control
- Configurable through `.env` file

### Data Validation
- Input sanitization
- SQL injection prevention
- Error handling for malformed data
