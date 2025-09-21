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

        self.path = f"/Users/zhuohengli/Cursor/darkerdb/data/{items}.csv"


    def connect(self):
        """连接到PostgreSQL数据库"""
        try:
            self.connector = psycopg2.connect(**self.connection_params)
            print("✅ 成功连接到PostgreSQL数据库")
            self.create_table()
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {str(e)}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connector:
            self.connector.close()
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
    
    