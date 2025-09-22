import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ARIMAAnalysis:
    def __init__(self, csv_path=None, df=None):
     
            # 默认路径
        self.df = pd.read_csv("/Users/zhuohengli/Cursor/Darker Market/cobalt_ore.csv")
        
        # 数据预处理
        self.prepare_data()
        
    def prepare_data(self):
        """数据预处理"""
        print("📊 开始数据预处理...")
        
        # 转换时间列
        if 'created_at' in self.df.columns:
            self.df['created_at'] = pd.to_datetime(self.df['created_at'])
            self.df = self.df.sort_values('created_at')
            self.df.set_index('created_at', inplace=True)
        
        # 选择价格列进行分析
        if 'price_per_unit' in self.df.columns:
            self.price_series = self.df['price_per_unit'].dropna()
        elif 'price' in self.df.columns:
            self.price_series = self.df['price'].dropna()
        else:
            raise ValueError("未找到价格列")
        
        # 处理异常值 - 使用IQR方法
        Q1 = self.price_series.quantile(0.25)
        Q3 = self.price_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"原始数据统计:")
        print(f"  数据点: {len(self.price_series)}")
        print(f"  最小值: {self.price_series.min():.2f}")
        print(f"  最大值: {self.price_series.max():.2f}")
        print(f"  平均值: {self.price_series.mean():.2f}")
        print(f"  中位数: {self.price_series.median():.2f}")
        
        # 标记异常值
        outliers = (self.price_series < lower_bound) | (self.price_series > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"发现 {outlier_count} 个异常值 ({outlier_count/len(self.price_series)*100:.1f}%)")
            print(f"异常值范围: {self.price_series[outliers].min():.2f} - {self.price_series[outliers].max():.2f}")
            
            # 限制异常值到边界值
            self.price_series = self.price_series.clip(lower_bound, upper_bound)
            print(f"✅ 限制异常值后，数据范围: {self.price_series.min():.2f} - {self.price_series.max():.2f}")
        
        print(f"✅ 数据预处理完成，共 {len(self.price_series)} 个数据点")
        
    def check_stationarity(self, timeseries):
        """检查时间序列的平稳性"""
        print("🔍 检查时间序列平稳性...")
        
        # ADF测试
        result = adfuller(timeseries.dropna())
        print(f"ADF统计量: {result[0]:.4f}")
        print(f"p值: {result[1]:.4f}")
        print(f"临界值:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.4f}")
        
        if result[1] <= 0.05:
            print("✅ 时间序列是平稳的")
            return True
        else:
            print("❌ 时间序列不是平稳的，需要进行差分")
            return False
    
    def make_stationary(self, timeseries):
        """使时间序列平稳"""
        print("🔄 进行差分使时间序列平稳...")
        
        # 一阶差分
        diff_series = timeseries.diff().dropna()
        
        # 检查差分后的平稳性
        if self.check_stationarity(diff_series):
            return diff_series, 1
        else:
            # 二阶差分
            diff2_series = diff_series.diff().dropna()
            if self.check_stationarity(diff2_series):
                return diff2_series, 2
            else:
                return diff2_series, 2  # 最多进行二阶差分
    
    def find_best_arima_params(self, timeseries, max_p=2, max_d=1, max_q=2):
        """寻找最佳ARIMA参数（进一步限制搜索范围）"""
        print("🔍 寻找最佳ARIMA参数...")
        print("⚠️  为了加快速度，限制搜索范围: p≤2, d≤1, q≤2")
        
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        # 先尝试简单的模型
        simple_models = [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]
        
        for p, d, q in simple_models:
            try:
                print(f"测试 ARIMA({p},{d},{q})...", end=" ")
                model = ARIMA(timeseries, order=(p, d, q))
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
                    best_model = fitted_model
                    print(f"✅ 新的最佳AIC: {best_aic:.2f}")
                else:
                    print(f"AIC: {fitted_model.aic:.2f}")
                    
            except Exception as e:
                print(f"❌ 失败: {str(e)[:30]}...")
                continue
        
        # 如果简单模型都失败，尝试更复杂的
        if best_model is None:
            print("简单模型失败，尝试更复杂的模型...")
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        if (p, d, q) in simple_models:
                            continue
                            
                        try:
                            print(f"测试 ARIMA({p},{d},{q})...", end=" ")
                            model = ARIMA(timeseries, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                best_model = fitted_model
                                print(f"✅ 新的最佳AIC: {best_aic:.2f}")
                            else:
                                print(f"AIC: {fitted_model.aic:.2f}")
                                
                        except Exception as e:
                            print(f"❌ 失败: {str(e)[:30]}...")
                            continue
        
        if best_model is None:
            print("❌ 所有ARIMA模型都拟合失败")
            return None, None
            
        print(f"✅ 最佳参数: ARIMA{best_params}, AIC: {best_aic:.2f}")
        return best_params, best_model
    
    def plot_analysis(self, timeseries, diff_series=None):
        """绘制分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始时间序列
        axes[0, 0].plot(timeseries.index, timeseries.values, color='blue', linewidth=1)
        axes[0, 0].set_title('Original Time Series', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Price', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加统计信息
        stats_text = f'Mean: {timeseries.mean():.2f}\nStd: {timeseries.std():.2f}\nMin: {timeseries.min():.2f}\nMax: {timeseries.max():.2f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 差分后的时间序列
        if diff_series is not None:
            axes[0, 1].plot(diff_series.index, diff_series.values, linewidth=1, color='orange')
            axes[0, 1].set_title('Differenced Time Series', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Price Difference', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Differencing Needed', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Differenced Time Series', fontsize=14, fontweight='bold')
        
        # ACF图
        plot_acf(timeseries.dropna(), ax=axes[1, 0], lags=20, alpha=0.05)
        axes[1, 0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # PACF图
        plot_pacf(timeseries.dropna(), ax=axes[1, 1], lags=20, alpha=0.05)
        axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def train_and_predict(self, test_size=0.2):
        """训练模型并进行预测"""
        print("🚀 开始训练ARIMA模型...")
        
        # 分割数据
        split_point = int(len(self.price_series) * (1 - test_size))
        train_data = self.price_series[:split_point]
        test_data = self.price_series[split_point:]
        
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        # 寻找最佳参数
        best_params, best_model = self.find_best_arima_params(train_data)
        
        if best_model is None:
            print("❌ 无法找到合适的ARIMA模型")
            return None
        
        # 进行预测
        try:
            forecast = best_model.forecast(steps=len(test_data))
            forecast_series = pd.Series(forecast, index=test_data.index)
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            # 使用最后已知值作为预测
            last_value = train_data.iloc[-1]
            forecast_series = pd.Series([last_value] * len(test_data), index=test_data.index)
        
        # 处理NaN值 - 使用更安全的方法
        if forecast_series.isna().any():
            print("⚠️  预测结果包含NaN值，使用最后已知值填充...")
            last_value = train_data.iloc[-1]
            forecast_series = forecast_series.fillna(last_value)
        
        # 确保数组长度一致
        min_length = min(len(forecast_series), len(test_data))
        forecast_series = forecast_series.iloc[:min_length]
        test_data_clean = test_data.iloc[:min_length]
        
        # 确保没有NaN值
        forecast_series = forecast_series.dropna()
        test_data_clean = test_data_clean.dropna()
        
        # 再次确保长度一致
        min_length = min(len(forecast_series), len(test_data_clean))
        if min_length > 0:
            forecast_series = forecast_series.iloc[:min_length]
            test_data_clean = test_data_clean.iloc[:min_length]
        
        # 检查数据是否为空
        if len(forecast_series) == 0 or len(test_data_clean) == 0:
            print("⚠️  预测数据为空，使用默认评估指标")
            mae = 0.0
            mse = 0.0
            rmse = 0.0
        else:
            # 确保最终长度一致
            if len(forecast_series) != len(test_data_clean):
                print(f"⚠️  长度不一致: forecast={len(forecast_series)}, test={len(test_data_clean)}")
                min_len = min(len(forecast_series), len(test_data_clean))
                forecast_series = forecast_series.iloc[:min_len]
                test_data_clean = test_data_clean.iloc[:min_len]
            
            # 计算评估指标
            mae = mean_absolute_error(test_data_clean, forecast_series)
            mse = mean_squared_error(test_data_clean, forecast_series)
            rmse = np.sqrt(mse)
        
        print(f"📈 模型评估结果:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        return {
            'model': best_model,
            'params': best_params,
            'forecast': forecast_series,
            'test_data': test_data,
            'train_data': train_data,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    def plot_predictions(self, results):
        """绘制预测结果"""
        plt.figure(figsize=(15, 8))
        
        # 绘制训练数据
        plt.plot(results['train_data'].index, results['train_data'].values, 
                label='Training Data', color='blue', alpha=0.7, linewidth=1)
        
        # 绘制测试数据
        plt.plot(results['test_data'].index, results['test_data'].values, 
                label='Actual Values', color='green', linewidth=2)
        
        # 绘制预测值
        plt.plot(results['forecast'].index, results['forecast'].values, 
                label='Predicted Values', color='red', linewidth=2, linestyle='--')
        
        # 添加分割线
        split_point = results['train_data'].index[-1]
        plt.axvline(x=split_point, color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
        
        # 添加统计信息
        plt.text(0.02, 0.98, f'Model: ARIMA{results["params"]}\nAIC: {results["model"].aic:.2f}\nRMSE: {results["rmse"]:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.title('ARIMA Price Prediction Results', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def future_forecast(self, model, steps=30):
        """未来预测"""
        print(f"🔮 进行未来 {steps} 步预测...")
        
        try:
            forecast = model.forecast(steps=steps)
            forecast_index = pd.date_range(
                start=self.price_series.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            # 处理NaN值
            if forecast_series.isna().any():
                print("⚠️  预测结果包含NaN值，使用最后已知值填充...")
                last_value = self.price_series.iloc[-1]
                forecast_series = forecast_series.fillna(last_value)
            
            return forecast_series
            
        except Exception as e:
            print(f"❌ 未来预测失败: {str(e)}")
            # 使用最后已知值作为预测
            last_value = self.price_series.iloc[-1]
            forecast_index = pd.date_range(
                start=self.price_series.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            forecast_series = pd.Series([last_value] * steps, index=forecast_index)
            return forecast_series
    
    def plot_future_trend(self, future_forecast, steps=30):
        """绘制未来趋势图"""
        plt.figure(figsize=(15, 8))
        
        # 获取所有历史数据来显示趋势
        recent_data = self.price_series  # 显示所有7天的数据
        
        # 绘制历史数据
        plt.plot(recent_data.index, recent_data.values, 
                label='Historical Data (7 days)', color='blue', linewidth=1.5, alpha=0.8)
        
        # 绘制未来预测
        plt.plot(future_forecast.index, future_forecast.values, 
                label=f'Future Forecast ({steps} days)', color='red', linewidth=3, linestyle='--')
        
        # 添加历史趋势线
        if len(recent_data) > 10:
            x_hist = range(len(recent_data))
            y_hist = recent_data.values
            z_hist = np.polyfit(x_hist, y_hist, 1)
            trend_line_hist = np.poly1d(z_hist)
            plt.plot(recent_data.index, trend_line_hist(x_hist), 
                    color='lightblue', linewidth=2, linestyle=':', alpha=0.7,
                    label=f'Historical Trend: {"上升" if z_hist[0] > 0 else "下降"} ({z_hist[0]:.3f}/day)')
        
        # 添加未来趋势线
        if len(future_forecast) > 1:
            x_future = range(len(future_forecast))
            y_future = future_forecast.values
            z_future = np.polyfit(x_future, y_future, 1)
            trend_line_future = np.poly1d(z_future)
            plt.plot(future_forecast.index, trend_line_future(x_future), 
                    color='orange', linewidth=2, linestyle=':', 
                    label=f'Future Trend: {"上升" if z_future[0] > 0 else "下降"} ({z_future[0]:.3f}/day)')
        
        # 添加价格区间
        price_std = recent_data.std()
        price_mean = recent_data.mean()
        plt.axhspan(price_mean - price_std, price_mean + price_std, 
                   alpha=0.2, color='gray', label=f'Normal Range (±1σ)')
        
        # 添加统计信息
        current_price = self.price_series.iloc[-1]
        future_price = future_forecast.iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # 计算价格波动性
        price_volatility = recent_data.std() / recent_data.mean() * 100
        
        stats_text = f'Current: {current_price:.2f}\nFuture: {future_price:.2f}\nChange: {price_change:+.2f} ({price_change_pct:+.1f}%)\nVolatility: {price_volatility:.1f}%'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.title('Price Trend Analysis & Future Forecast', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 打印详细趋势分析
        print(f"\n📈 详细趋势分析:")
        print(f"当前价格: {current_price:.2f}")
        print(f"预测价格: {future_price:.2f}")
        print(f"价格变化: {price_change:+.2f} ({price_change_pct:+.1f}%)")
        print(f"价格波动性: {price_volatility:.1f}%")
        
        if len(recent_data) > 10:
            print(f"\n历史趋势 (过去7天):")
            if z_hist[0] > 0.001:
                print(f"  上升趋势: 每日平均上涨 {z_hist[0]:.3f}")
            elif z_hist[0] < -0.001:
                print(f"  下降趋势: 每日平均下跌 {abs(z_hist[0]):.3f}")
            else:
                print(f"  平稳趋势: 变化很小 ({z_hist[0]:.3f}/day)")
        
        print(f"\n未来预测趋势:")
        if z_future[0] > 0.001:
            print(f"  预测上升: 每日平均上涨 {z_future[0]:.3f}")
        elif z_future[0] < -0.001:
            print(f"  预测下降: 每日平均下跌 {abs(z_future[0]):.3f}")
        else:
            print(f"  预测平稳: 变化很小 ({z_future[0]:.3f}/day)")
        
        # 给出投资建议
        print(f"\n💡 投资建议:")
        if price_change_pct > 5:
            print("  价格预计显著上涨，可考虑买入")
        elif price_change_pct < -5:
            print("  价格预计显著下跌，可考虑卖出")
        else:
            print("  价格预计相对稳定，可持有观望")
        
        if price_volatility > 20:
            print("  价格波动较大，注意风险控制")
        else:
            print("  价格波动较小，相对稳定")
    
    def analyze_sparse_data(self):
        """分析稀疏数据特征"""
        print("📊 分析稀疏数据特征...")
        
        # 统计非零值
        nonzero_data = self.price_series[self.price_series > 0]
        zero_count = len(self.price_series) - len(nonzero_data)
        
        print(f"总数据点: {len(self.price_series)}")
        print(f"非零值: {len(nonzero_data)} ({len(nonzero_data)/len(self.price_series)*100:.1f}%)")
        print(f"零值: {zero_count} ({zero_count/len(self.price_series)*100:.1f}%)")
        
        if len(nonzero_data) > 0:
            print(f"非零值统计:")
            print(f"  最小值: {nonzero_data.min():.2f}")
            print(f"  最大值: {nonzero_data.max():.2f}")
            print(f"  平均值: {nonzero_data.mean():.2f}")
            print(f"  中位数: {nonzero_data.median():.2f}")
            print(f"  标准差: {nonzero_data.std():.2f}")
        
        # 分析价格尖峰模式
        if len(nonzero_data) > 1:
            price_changes = nonzero_data.diff().dropna()
            print(f"\n价格变化统计:")
            print(f"  平均变化: {price_changes.mean():.2f}")
            print(f"  最大涨幅: {price_changes.max():.2f}")
            print(f"  最大跌幅: {price_changes.min():.2f}")
        
        return len(nonzero_data) / len(self.price_series)

    def analysis(self):
        """完整的ARIMA分析流程"""
        print("🎯 开始ARIMA时间序列分析")
        print("=" * 50)
        
        # 首先分析数据质量
        sparsity = self.analyze_sparse_data()
        print(f"\n数据稀疏度: {sparsity:.3f}")
        
        # 分析数据时间跨度
        time_span = (self.price_series.index[-1] - self.price_series.index[0]).days
        print(f"数据时间跨度: {time_span} 天")
        print(f"数据点数量: {len(self.price_series)}")
        print(f"平均每天数据点: {len(self.price_series) / max(time_span, 1):.1f}")
        
        if time_span < 7:
            print("⚠️  警告: 数据时间跨度太短，ARIMA预测可能不准确")
            print("建议: 至少需要7-30天的数据才能进行可靠预测")
        elif time_span < 30:
            print("⚠️  注意: 数据时间跨度较短，建议只做短期预测(1-3天)")
        else:
            print("✅ 数据时间跨度充足，适合ARIMA分析")
        
        if sparsity < 0.1:
            print("⚠️  警告: 数据非常稀疏，ARIMA模型可能不适合")
            print("建议: 考虑使用其他方法，如异常检测或事件预测模型")
        elif sparsity < 0.3:
            print("⚠️  注意: 数据较为稀疏，ARIMA模型效果可能有限")
        else:
            print("✅ 数据密度适中，适合ARIMA分析")
        
        # 1. 检查平稳性
        is_stationary = self.check_stationarity(self.price_series)
        
        # 2. 如果不平稳，进行差分
        if not is_stationary:
            diff_series, diff_order = self.make_stationary(self.price_series)
        else:
            diff_series = None
            diff_order = 0
        
        # 3. 训练模型并预测
        results = self.train_and_predict()
        
        if results is None:
            print("❌ 模型训练失败，无法继续分析")
            return None, None
        
        # 4. 未来预测 (基于7天数据，只预测3天)
        future_forecast = self.future_forecast(results['model'], steps=3)
        
        # 5. 绘制未来趋势图
        self.plot_future_trend(future_forecast, steps=3)
        
        print("\n📊 分析完成！")
        print(f"最佳模型: ARIMA{results['params']}")
        print(f"模型AIC: {results['model'].aic:.2f}")
        print(f"预测RMSE: {results['rmse']:.2f}")
        
        return results, future_forecast


# 使用示例
if __name__ == "__main__":
    # 创建ARIMA分析实例
    analyzer = ARIMAAnalysis()
    
    # 运行完整分析
    results, future = analyzer.analysis()
        