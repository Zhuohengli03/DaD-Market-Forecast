import requests
import pandas
from Database_connect import DarkerMarketDB
import time
import os



need_run = 200
time_sleep = 0

class DarkerMarketAPI:
    def __init__(self):
        self.item = "Cobalt Ore"
        self.page = 1
        self.limit = 50
        self.order = "desc"
        self.from_time = "30 days ago"
        self.csv_filename = f"{self.item.replace(' ', '_').lower()}.csv"
        self.total_inserted = 0  # Track total inserted data
        
        self.headers = {
            "Authorization": "960f723902200d13d5c7"
        }


    def run(self):
        if need_run:
            while need_run > self.page:
                self.get_market_data()
                time.sleep(time_sleep)
                self.page += 1
                print(f"Page: {self.page}")
            
            # 数据收集完成后，保存到CSV
            print(f"\n🎯 数据收集完成，开始保存到CSV...")
            print(f"📊 本次运行总计新增数据: {self.total_inserted} 条")
            self.save_data_to_csv()
        else:
            self.get_market_data()
            print(f"📊 本次运行总计新增数据: {self.total_inserted} 条")
            # 单次运行也保存到CSV
            self.save_data_to_csv()


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
            # print(df) # Commented out print statement
            print(f"Total items: {i}")

            replace_item = self.item.replace(" ", "_").lower()
            # 将API数据传递给数据库
            db = DarkerMarketDB(items=replace_item, df=df)

            # 连接数据库并插入数据
            if db.connect():
                new_count = db.insert_market_data()
                if new_count is not None:
                    self.total_inserted += new_count
                db.disconnect()
                print(f"✅ 数据已保存到数据库 (本页新增: {new_count if new_count is not None else 0}, 累计新增: {self.total_inserted})")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Page {self.page}: API请求失败 - {str(e)}")
        except Exception as e:
            print(f"❌ Page {self.page}: 处理数据时出错 - {str(e)}") 



    
    def save_data_to_csv(self):
        """从数据库导出数据到CSV文件"""
        try:
            print(f"\n💾 开始导出数据到CSV文件: {self.csv_filename}")
            
            # 连接数据库
            db = DarkerMarketDB(items=self.item.replace(" ", "_").lower())
            if not db.connect():
                print("❌ 数据库连接失败")
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