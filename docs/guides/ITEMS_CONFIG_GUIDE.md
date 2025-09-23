# 物品配置管理指南

## 📋 概述

新的物品配置系统允许您轻松添加和管理新的市场物品，无需修改代码。

## 🗂️ 配置文件结构

### `items_config.json`
```json
{
  "items": [
    {
      "name": "Gold Ore",           // 显示名称
      "file": "Gold_Ore_API.py",   // API文件名
      "csv": "gold_ore.csv",       // CSV文件名
      "category": "ore",           // 物品类别
      "description": "黄金矿石",    // 描述
      "enabled": true              // 是否启用
    }
  ],
  "categories": {
    "ore": "矿石类",
    "consumable": "消耗品类",
    "equipment": "装备类",
    "material": "材料类"
  },
  "auto_discovery": {
    "enabled": true,              // 启用自动发现
    "ore_directory": "Ore",       // 矿石目录
    "consumable_directory": "Consumable",  // 消耗品目录
    "naming_pattern": "*_API.py"  // 文件名模式
  }
}
```

## 🚀 添加新物品的方法

### 方法1：通过配置管理界面 (推荐)
1. 运行机器学习分析系统
2. 选择 "3. 🛠️ 物品配置管理"
3. 选择 "2. 添加新物品"
4. 按提示输入信息

### 方法2：手动编辑配置文件
1. 编辑 `items_config.json`
2. 在 `items` 数组中添加新物品
3. 保存文件

### 方法3：程序化添加
```python
from Analysis.Machine_learning_analysis import add_new_item

# 添加新物品
add_new_item(
    name="Silver Ore",
    api_file="Silver_Ore_API.py", 
    csv_file="silver_ore.csv",
    category="ore",
    description="白银矿石 - 中等价值金属"
)
```

## 📁 目录结构要求

```
Darker Market/
├── items_config.json          # 配置文件
├── Ore/                       # 矿石类API
│   ├── Gold_Ore_API.py
│   ├── Iron_Ore_API.py
│   ├── Silver_Ore_API.py      # 新增物品
│   └── ...
├── Consumable/                # 消耗品类API (可选)
│   ├── Health_Potion_API.py
│   └── ...
├── gold_ore.csv              # CSV数据文件
├── iron_ore.csv
├── silver_ore.csv            # 新增数据
└── ...
```

## 🔍 自动发现功能

系统会自动扫描指定目录，发现新的API文件：

- **扫描目录**: `Ore/`, `Consumable/`
- **文件模式**: `*_API.py`
- **命名规则**: `物品名_API.py` → `物品名.csv`

例如：
- `Silver_Ore_API.py` → `Silver Ore` + `silver_ore.csv`
- `Health_Potion_API.py` → `Health Potion` + `health_potion.csv`

## ⚙️ 物品类别

支持的类别：
- **ore** (矿石类): 存放在 `Ore/` 目录
- **consumable** (消耗品类): 存放在 `Consumable/` 目录  
- **equipment** (装备类): 存放在 `Equipment/` 目录
- **material** (材料类): 存放在 `Material/` 目录

## 📝 添加新物品的完整流程

### 1. 创建API文件
在相应目录下创建 `新物品_API.py`:
```python
# 例：Ore/Silver_Ore_API.py
# 复制现有API文件并修改物品ID和参数
```

### 2. 添加配置 (三种方式任选一种)
- 使用配置管理界面
- 手动编辑配置文件
- 程序化添加

### 3. 测试
运行分析系统，新物品应该出现在列表中

## 🔧 高级配置

### 禁用物品
将 `enabled` 设为 `false`:
```json
{
  "name": "Old Item",
  "enabled": false
}
```

### 禁用自动发现
```json
{
  "auto_discovery": {
    "enabled": false
  }
}
```

### 自定义目录结构
```json
{
  "auto_discovery": {
    "enabled": true,
    "custom_directories": ["CustomAPI", "SpecialItems"],
    "naming_pattern": "*_api.py"
  }
}
```

## ❗ 注意事项

1. **API文件**: 确保新的API文件遵循现有格式
2. **CSV文件**: API运行后会生成对应的CSV文件
3. **命名一致性**: 保持文件名、配置名称的一致性
4. **测试**: 添加新物品后先测试API是否正常工作

## 🆘 故障排除

### 物品未显示
- 检查 `enabled: true`
- 确认API文件存在
- 检查文件路径是否正确

### API调用失败
- 确认API文件语法正确
- 检查物品ID和参数
- 查看错误输出

### CSV文件未生成
- 确认API成功执行
- 检查数据库连接
- 确认写入权限
