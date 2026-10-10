import requests
import pandas
import time
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.Database_connect import DarkerMarketDB
from src.database.config import config



need_run = 200
time_sleep = 0

class DarkerMarketAPI:
    def __init__(self):
        self.item = "Iron Ore"
        self.page = 1
        self.limit = 50
        self.order = "desc"
        
        self.from_time = "2025-10-05"
        self.csv_filename = f"{self.item.replace(' ', '_').lower()}.csv"
        self.total_inserted = 0  # Track total inserted data
        self.no_new_data_count = 0  # 连续没有新数据的次数
        self.max_no_new_data = 3  # 最大允许连续没有新数据的次数
        self.db = None  # 数据库连接对象
        self.seen_records = set()  # 用于去重的已见记录集合
        
        self.headers = config.get_api_headers()

    def _load_existing_records(self):
        """加载数据库中已存在的记录到seen_records"""
        if not self.db.connector:
            print("❌ 数据库未连接，无法加载现有记录")
            return
        
        try:
            cursor = self.db.connector.cursor()
            cursor.execute(f"SELECT item_id, created_at FROM {self.db.items}")
            existing_count = 0
            for row in cursor.fetchall():
                record_key = f"{row[0]}_{row[1]}"
                self.seen_records.add(record_key)
                existing_count += 1
            cursor.close()
            print(f"📚 已加载 {existing_count} 条现有记录到去重集合")
        except Exception as e:
            print(f"❌ 加载现有记录失败: {str(e)}")

    def run(self):
        if need_run:
            # 连接数据库
            print("🔌 连接数据库...")
            print("🔌 Connecting to database...")
            self.db = DarkerMarketDB(items=self.item.replace(" ", "_").lower())
            if not self.db.connect():
                print("❌ 数据库连接失败\n❌ Database connection failed")
                print("❌ Database connection failed")
                return
            
            # 加载数据库中已存在的记录到去重集合
            self._load_existing_records()
            
            while need_run > self.page and self.no_new_data_count < self.max_no_new_data:
                new_data_count = self.get_market_data()
                
                # 检查是否有新数据
                if new_data_count is not None and new_data_count > 0:
                    self.no_new_data_count = 0  # 重置计数器
                    print(f"✅ 第{self.page}页: 收集到 {new_data_count} 条数据")
                else:
                    self.no_new_data_count += 1
                    print(f"⚠️  第{self.page}页: 没有新数据 (连续 {self.no_new_data_count}/{self.max_no_new_data} 次)")
                
                # 检查是否应该停止
                if self.no_new_data_count >= self.max_no_new_data:
                    print(f"\n🛑 连续 {self.max_no_new_data} 次没有新数据，自动停止数据收集")
                    break
                
                time.sleep(time_sleep)
                self.page += 1
                
            
            # 统一插入所有收集的数据
            print(f"\n💾 开始统一插入数据到数据库...")
            self.total_inserted = self.db.batch_insert_all_data()
            
            # 断开数据库连接
            self.db.disconnect()
            
            # 数据收集完成后，保存到CSV
            print(f"\n🎯 数据收集完成，开始保存到CSV...")
            print(f"📊 本次运行总计新增数据: {self.total_inserted} 条")
            print(f"📊 总共处理页数: {self.page - 1}")
            self.save_data_to_csv()
        else:
            # 单次运行模式
            self.db = DarkerMarketDB(items=self.item.replace(" ", "_").lower())
            if self.db.connect():
                new_data_count = self.get_market_data()
                self.total_inserted = self.db.batch_insert_all_data()
                self.db.disconnect()
                print(f"📊 本次运行总计新增数据: {self.total_inserted} 条")
                # 单次运行也保存到CSV
                self.save_data_to_csv()
            else:
                print("❌ 数据库连接失败\n❌ Database connection failed")


    def get_market_data(self):
        url  = "https://api.darkerdb.com/v1/market?"
        params = {
            "page": self.page,
            "limit": self.limit,
            "item": self.item,
            "order": self.order,
            "from": self.from_time,
        }
       
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # print(data)
            table = {
                "id": [],
                "item": [],
                "quantity": [],
                "price_per_unit": [],
                "price": [],
                "has_sold": [],
                "created_at": [],
            }

            i = 0
            for item in data["body"]:
                i += 1
                table["id"].append(item["id"])
                table["item"].append(item["item"])
                table["quantity"].append(item["quantity"])
                table["price_per_unit"].append(item["price_per_unit"])
                table["price"].append(item["price"])
                table["has_sold"].append(item["has_sold"])
                table["created_at"].append(item["created_at"])

            df = pandas.DataFrame(table).to_dict(orient="records")
            

            # 去重处理
            unique_data = []
            new_count = 0
            
            for record in df:
                # 创建唯一标识符
                record_key = f"{record['id']}_{record['created_at']}"
                
                if record_key not in self.seen_records:
                    self.seen_records.add(record_key)
                    unique_data.append(record)
                    new_count += 1
            
            print(f"去重后: {new_count} 条新数据 (跳过 {i - new_count} 条重复数据)")
            
            # 将去重后的数据添加到数据库对象的待插入列表
            if unique_data:
                self.db.add_data(unique_data)
            
            # 返回去重后的新数据数量
            return new_count
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Page {self.page}: API请求失败 - {str(e)}")
            return 0
        except Exception as e:
            print(f"❌ Page {self.page}: 处理数据时出错 - {str(e)}")
            return 0
        
        # 如果没有数据或API请求失败，返回0
        return 0 



    
    def save_data_to_csv(self):
        """从数据库导出数据到CSV文件"""
        try:
            print(f"\n💾 开始导出数据到CSV文件: {self.csv_filename}")
            
            # 连接数据库
            db = DarkerMarketDB(items=self.item.replace(" ", "_").lower())
            if not db.connect():
                print("❌ 数据库连接失败\n❌ Database connection failed")
                return False
            
            # 查询所有数据
            cursor = db.connector.cursor()
            cursor.execute(f"""
                SELECT item_id, item, quantity, price_per_unit, price, has_sold, created_at, insert_time
                FROM {self.item.replace(' ', '_').lower()}
                ORDER BY created_at DESC
            """)
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 获取所有数据
            data = cursor.fetchall()
            
            if not data:
                print("❌ 数据库中没有数据")
                cursor.close()
                db.disconnect()
                return False
            
            # 创建DataFrame
            df = pandas.DataFrame(data, columns=columns)
            
            # 保存到CSV
            df.to_csv(self.csv_filename, index=False, encoding='utf-8')
            
            print(f"✅ 成功导出 {len(data)} 条数据到 {self.csv_filename}")
            print(f"📁 文件路径: {os.path.abspath(self.csv_filename)}")
            
            # 显示文件大小
            file_size = os.path.getsize(self.csv_filename)
            if file_size > 1024 * 1024:
                print(f"📊 文件大小: {file_size / (1024 * 1024):.2f} MB")
            else:
                print(f"📊 文件大小: {file_size / 1024:.2f} KB")
            
            cursor.close()
            db.disconnect()
            return True
            
        except Exception as e:
            print(f"❌ 导出CSV失败: {str(e)}")
            return False
        

DarkerMarketAPI().run()