#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境变量配置模块
Environment Variables Configuration Module
"""

import os
from dotenv import load_dotenv

# 加载环境变量文件
load_dotenv()

class Config:
    """配置类 / Configuration Class"""
    
    # 数据库配置 / Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_DATABASE = os.getenv('DB_DATABASE', 'darkerdb')
    DB_USER = os.getenv('DB_USER', '')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_PORT = int(os.getenv('DB_PORT', '5432'))
    
    # API配置 / API Configuration
    DARKER_MARKET_API_KEY = os.getenv('DARKER_MARKET_API_KEY', '')
    
    # 文件路径配置 / File Path Configuration
    DATA_DIR = os.getenv('DATA_DIR', '/Users/zhuohengli/Cursor/darkerdb/data')
    
    @classmethod
    def get_db_config(cls):
        """获取数据库配置 / Get database configuration"""
        return {
            'host': cls.DB_HOST,
            'database': cls.DB_DATABASE,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'port': cls.DB_PORT
        }
    
    @classmethod
    def get_api_headers(cls):
        """获取API请求头 / Get API headers"""
        return {
            "Authorization": cls.DARKER_MARKET_API_KEY
        }
    
    @classmethod
    def validate_config(cls):
        """验证配置 / Validate configuration"""
        required_vars = [
            'DB_HOST', 'DB_DATABASE', 'DB_USER', 'DB_PASSWORD',
            'DARKER_MARKET_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

# 创建全局配置实例
config = Config()