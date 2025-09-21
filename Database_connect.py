import psycopg2
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


class DarkerMarketDB:
    """Darker Market 数据库连接器"""
    
    def __init__(self, host="localhost", database="darkerdb", user="zhuohengli", password="0728", items="items", df="df"):
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password
        }
        self.connector = None
        self.items = items
        self.df = df
        self.all_data = []  # 存储所有待插入的数据
        self.is_connected = False  # 连接状态标志

        self.path = f"/Users/zhuohengli/Cursor/darkerdb/data/{items}.csv"


    def connect(self):
        """连接到PostgreSQL数据库"""
        try:
            if not self.is_connected:
                self.connector = psycopg2.connect(**self.connection_params)
                self.is_connected = True
                print("✅ 成功连接到PostgreSQL数据库")
                self.create_table()
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {str(e)}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connector and self.is_connected:
            self.connector.close()
            self.is_connected = False
            print("🔌 数据库连接已断开")
    
    def create_table(self):
        """创建Dark_market_items表"""
        if not self.connector:
            print("❌ 请先连接数据库")
            return False
            
        try:
            cursor = self.connector.cursor()
            # 只创建表（如果不存在），不删除旧数据
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.items} ( 
                    item_id VARCHAR(255),
                    item VARCHAR(255),
                    quantity INTEGER,
                    price_per_unit FLOAT,
                    price FLOAT,
                    has_sold BOOLEAN,
                    created_at VARCHAR(255),
                    insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connector.commit()
            cursor.close()
            print("✅ 数据表创建成功")
            return True
        except Exception as e:
            print(f"❌ 创建表失败: {str(e)}")
            return False
    
    def insert_market_data(self):
        """插入市场数据（避免重复）"""
        if not self.connector:
            print("❌ 请先连接数据库")
            return False
            
        try:
            cursor = self.connector.cursor()
            
            # 准备数据
            data_to_insert = []
            new_count = 0
            
            for item in self.df:
                # 确保数据类型正确
                item_id = str(item.get('id', ''))
                item_name = str(item.get('item', ''))
                quantity = int(item.get('quantity', 0))
                price_per_unit = float(item.get('price_per_unit', 0.0))
                price = float(item.get('price', 0.0))
                has_sold = bool(item.get('has_sold', False))
                created_at = item.get('created_at', '')
                
                # 检查数据是否已存在（基于item_id和created_at）
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.items} 
                    WHERE item_id = %s AND created_at = %s
                """, (item_id, created_at))
                
                exists = cursor.fetchone()[0] > 0
                
                if not exists:
                    data_to_insert.append((
                        item_id,
                        item_name,
                        quantity,
                        price_per_unit,
                        price,
                        has_sold,
                        created_at,
                        datetime.now()
                    ))
                    new_count += 1
            
            # 只插入新数据
            if data_to_insert:
                insert_query = f"""
                    INSERT INTO {self.items} (item_id, item, quantity, price_per_unit, price, has_sold, created_at, insert_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, data_to_insert)
                self.connector.commit()
                print(f"✅ 成功插入 {new_count} 条新数据（跳过 {len(self.df) - new_count} 条重复数据）")
            else:
                print(f"ℹ️  所有 {len(self.df)} 条数据都已存在，无需插入")
            
            cursor.close()
            
            # 返回新插入的数据数量
            return new_count
            
        except Exception as e:
            print(f"❌ 插入数据失败: {str(e)}")
            return False
    
    def add_data(self, data_batch):
        """添加数据到待插入列表"""
        if isinstance(data_batch, list):
            self.all_data.extend(data_batch)
        else:
            self.all_data.append(data_batch)
        print(f"📝 已收集 {len(self.all_data)} 条数据待插入")
    
    def batch_insert_all_data(self):
        """批量插入所有收集的数据 - 优化版去重"""
        if not self.connector or not self.is_connected:
            print("❌ 请先连接数据库")
            return False
            
        if not self.all_data:
            print("ℹ️  没有数据需要插入")
            return 0
            
        try:
            cursor = self.connector.cursor()
            
            # 先获取所有现有的item_id和created_at组合
            print("🔍 检查数据库中现有数据...")
            cursor.execute(f"SELECT item_id, created_at FROM {self.items}")
            existing_records = set()
            for row in cursor.fetchall():
                existing_records.add(f"{row[0]}_{row[1]}")
            
            print(f"📊 数据库中现有 {len(existing_records)} 条记录")
            
            # 准备数据 - 只插入不存在的记录
            data_to_insert = []
            new_count = 0
            duplicate_count = 0
            
            for item in self.all_data:
                # 确保数据类型正确
                item_id = str(item.get('id', ''))
                item_name = str(item.get('item', ''))
                quantity = int(item.get('quantity', 0))
                price_per_unit = float(item.get('price_per_unit', 0.0))
                price = float(item.get('price', 0.0))
                has_sold = bool(item.get('has_sold', False))
                created_at = item.get('created_at', '')
                
                # 创建唯一标识符
                record_key = f"{item_id}_{created_at}"
                
                # 检查是否已存在
                if record_key not in existing_records:
                    data_to_insert.append((
                        item_id, item_name, quantity, price_per_unit, 
                        price, has_sold, created_at, datetime.now()
                    ))
                    new_count += 1
                    # 添加到现有记录集合中，避免同一批次内的重复
                    existing_records.add(record_key)
                else:
                    duplicate_count += 1
            
            # 批量插入新数据
            if data_to_insert:
                insert_query = f"""
                    INSERT INTO {self.items} (item_id, item, quantity, price_per_unit, price, has_sold, created_at, insert_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, data_to_insert)
                self.connector.commit()
                print(f"✅ 批量插入完成: 新增 {new_count} 条数据（跳过 {duplicate_count} 条重复数据）")
            else:
                print(f"ℹ️  所有 {len(self.all_data)} 条数据都已存在，无需插入")
            
            cursor.close()
            
            # 清空待插入数据列表
            self.all_data = []
            
            return new_count
            
        except Exception as e:
            print(f"❌ 批量插入数据失败: {str(e)}")
            return False
    
    def show_data_count(self):
        """显示数据库中的数据总数"""
        try:
            cursor = self.connector.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.items}")
            total_count = cursor.fetchone()[0]
            print(f"📊 数据库中总共有 {total_count} 条数据")
            cursor.close()
        except Exception as e:
            print(f"❌ 查询数据总数失败: {str(e)}")
    
    