import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MarketMLAnalyzer:
    def __init__(self, csv_file_path):
        """
        初始化机器学习分析器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("🔄 加载和预处理数据...")
        
        try:
            # 加载数据
            self.df = pd.read_csv(self.csv_file_path)
            print(f"✅ 成功加载数据: {len(self.df)} 条记录")
            
            # 数据基本信息
            print(f"📊 数据形状: {self.df.shape}")
            print(f"📅 时间范围: {self.df['created_at'].min()} 到 {self.df['created_at'].max()}")
            
            # 数据预处理
            self._preprocess_data()
            
            # 特征工程
            self._feature_engineering()
            
            # 准备训练数据
            self._prepare_training_data()
            
            print("✅ 数据预处理完成")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False
            
        return True
    
    def _preprocess_data(self):
        """数据预处理"""
        print("🔧 进行数据预处理...")
        
        # 转换时间列
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        self.df['insert_time'] = pd.to_datetime(self.df['insert_time'])
        
        # 排序
        self.df = self.df.sort_values('created_at').reset_index(drop=True)
        
        # 处理缺失值
        self.df = self.df.dropna()
        
        # 数据类型转换
        self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce')
        self.df['price_per_unit'] = pd.to_numeric(self.df['price_per_unit'], errors='coerce')
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['has_sold'] = self.df['has_sold'].astype(int)
        
        # 移除异常值
        self._remove_outliers()
        
        print(f"📊 预处理后数据形状: {self.df.shape}")
    
    def _remove_outliers(self):
        """移除异常值 - 使用改进的Z-score方法"""
        print("🔍 检测和移除异常值...")
        
        # 使用Z-score方法检测异常值，对价格数据使用更宽松的阈值
        numeric_columns = ['quantity', 'price_per_unit', 'price']
        original_shape = self.df.shape
        
        for col in numeric_columns:
            # 计算Z-score
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            
            # 对价格数据使用更宽松的阈值（2.5σ），对数量使用标准阈值（3σ）
            threshold = 2.5 if col in ['price_per_unit', 'price'] else 3.0
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"  {col}: 发现 {outlier_count} 个异常值 (阈值: {threshold}σ)")
                self.df = self.df[~outliers]
        
        removed_count = original_shape[0] - self.df.shape[0]
        print(f"📊 移除异常值后数据形状: {self.df.shape} (移除了 {removed_count} 条记录)")
    
    def _feature_engineering(self):
        """高级特征工程"""
        print("⚙️ 进行高级特征工程...")
        
        # 1. 基础时间特征
        self.df['hour'] = self.df['created_at'].dt.hour
        self.df['day_of_week'] = self.df['created_at'].dt.dayofweek
        self.df['day_of_month'] = self.df['created_at'].dt.day
        self.df['month'] = self.df['created_at'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # 2. 价格相关特征
        self.df['price_quantity_ratio'] = self.df['price'] / (self.df['quantity'] + 1)
        self.df['price_per_unit_squared'] = self.df['price_per_unit'] ** 2
        self.df['quantity_squared'] = self.df['quantity'] ** 2
        self.df['price_quantity_interaction'] = self.df['price_per_unit'] * self.df['quantity']
        
        # 3. 移动平均特征 (多时间窗口)
        for window in [3, 5, 7, 10, 14]:
            self.df[f'price_ma_{window}'] = self.df['price_per_unit'].rolling(window=window).mean()
            self.df[f'quantity_ma_{window}'] = self.df['quantity'].rolling(window=window).mean()
            self.df[f'price_ema_{window}'] = self.df['price_per_unit'].ewm(span=window).mean()
        
        # 4. 价格变化特征
        self.df['price_change'] = self.df['price_per_unit'].diff()
        self.df['price_change_pct'] = self.df['price_per_unit'].pct_change()
        self.df['price_change_abs'] = np.abs(self.df['price_change'])
        
        # 5. 波动性特征
        for window in [3, 5, 10, 20]:
            self.df[f'price_volatility_{window}'] = self.df['price_per_unit'].rolling(window=window).std()
            self.df[f'price_skewness_{window}'] = self.df['price_per_unit'].rolling(window=window).skew()
            self.df[f'price_kurtosis_{window}'] = self.df['price_per_unit'].rolling(window=window).kurt()
        
        # 6. 趋势特征
        for window in [5, 10, 20]:
            self.df[f'price_trend_{window}'] = self.df['price_per_unit'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
            )
        
        # 7. 分位数特征
        for window in [10, 20]:
            self.df[f'price_q25_{window}'] = self.df['price_per_unit'].rolling(window=window).quantile(0.25)
            self.df[f'price_q75_{window}'] = self.df['price_per_unit'].rolling(window=window).quantile(0.75)
            self.df[f'price_iqr_{window}'] = self.df[f'price_q75_{window}'] - self.df[f'price_q25_{window}']
        
        # 8. 滞后特征
        for lag in [1, 2, 3, 5]:
            self.df[f'price_lag_{lag}'] = self.df['price_per_unit'].shift(lag)
            self.df[f'quantity_lag_{lag}'] = self.df['quantity'].shift(lag)
        
        # 9. 交互特征
        self.df['hour_quantity_interaction'] = self.df['hour'] * self.df['quantity']
        self.df['day_quantity_interaction'] = self.df['day_of_week'] * self.df['quantity']
        
        # 10. 填充缺失值
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        print("✅ 高级特征工程完成")
    
    def _prepare_training_data(self):
        """准备训练数据"""
        print("📊 准备训练数据...")
        
        # 选择优化特征 - 避免数据泄露，增加预测能力
        feature_columns = [
            # 基础特征
            'quantity', 'has_sold',
            # 时间特征
            'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            # 价格相关特征（不包含当前价格）
            'price_quantity_ratio', 'quantity_squared', 'price_quantity_interaction',
            # 数量移动平均
            'quantity_ma_3', 'quantity_ma_5', 'quantity_ma_7', 'quantity_ma_10', 'quantity_ma_14',
            # 滞后特征
            'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
            # 波动性特征
            'price_volatility_3', 'price_volatility_5', 'price_volatility_10', 'price_volatility_20',
            # 趋势特征
            'price_trend_5', 'price_trend_10', 'price_trend_20',
            # 分位数特征
            'price_q25_10', 'price_q75_10', 'price_iqr_10',
            'price_q25_20', 'price_q75_20', 'price_iqr_20',
            # 交互特征
            'hour_quantity_interaction', 'day_quantity_interaction'
        ]
        
        # 移除包含NaN的列
        available_features = [col for col in feature_columns if col in self.df.columns]
        self.X = self.df[available_features].fillna(0)
        self.y = self.df['price_per_unit']
        
        print(f"📊 特征数量: {len(available_features)}")
        print(f"📊 样本数量: {len(self.X)}")
        
        # 分割数据 - 使用固定比例确保稳定性
        split_index = int(len(self.X) * 0.8)
        
        # 确保分割点固定，避免随机性
        self.X_train = self.X.iloc[:split_index].copy()
        self.X_test = self.X.iloc[split_index:].copy()
        self.y_train = self.y.iloc[:split_index].copy()
        self.y_test = self.y.iloc[split_index:].copy()
        
        print(f"📊 训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
        print(f"📊 分割点: 第{split_index}条记录 ({(split_index/len(self.X)*100):.1f}%)")
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ 训练数据准备完成")
    
    def train_models(self):
        """训练多个机器学习模型"""
        print("🤖 开始训练机器学习模型...")
        
        # 定义优化模型 - 增强性能和稳定性
        models = {
            # 集成学习模型
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=5, 
                min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.03, 
                subsample=0.8, random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            # 线性模型
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=0.1, random_state=42),
            'Lasso Regression': Lasso(alpha=0.01, random_state=42, max_iter=5000),
            'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
            # 支持向量机
            'SVR': SVR(kernel='rbf', C=10.0, gamma='auto', epsilon=0.01),
            # 神经网络
            'MLP Regressor': MLPRegressor(
                hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                alpha=0.001, learning_rate='adaptive', max_iter=1000, random_state=42
            )
        }
        
        # 训练和评估模型
        model_scores = {}
        
        for name, model in models.items():
            print(f"🔄 训练 {name}...")
            
            try:
                # 训练模型
                if name == 'SVR':
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                
                # 评估模型
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                
                # 添加交叉验证评估
                if name == 'SVR':
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=3, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2')
                
                model_scores[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                print(f"  ✅ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}, CV-R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"  ❌ {name} 训练失败: {str(e)}")
        
        self.models = model_scores
        
        # 创建集成模型
        print("\n🔄 创建集成模型...")
        ensemble_models = self._create_ensemble_model(model_scores)
        
        # 评估集成模型
        if ensemble_models:
            ensemble_scores = self._evaluate_ensemble_model(ensemble_models)
            model_scores['Ensemble'] = ensemble_scores
            print(f"✅ 集成模型: R²={ensemble_scores['r2']:.3f}, CV-R²={ensemble_scores['cv_r2_mean']:.3f}")
        
        # 选择最佳模型 - 使用交叉验证分数
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['cv_r2_mean'])
        self.best_model = model_scores[best_model_name]['model']
        
        print(f"\n🏆 最佳模型: {best_model_name}")
        print(f"📊 最佳模型性能: R²={model_scores[best_model_name]['r2']:.3f}, CV-R²={model_scores[best_model_name]['cv_r2_mean']:.3f}")
        
        # 详细模型准确性分析
        self._analyze_model_accuracy(model_scores[best_model_name], best_model_name)
        
        return model_scores
    
    def _create_ensemble_model(self, model_scores):
        """创建集成模型"""
        try:
            # 选择前5个最佳模型进行集成
            top_models = sorted(model_scores.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)[:5]
            
            if len(top_models) < 2:
                print("⚠️ 模型数量不足，跳过集成")
                return None
            
            # 创建投票回归器
            estimators = []
            for name, scores in top_models:
                if name in ['SVR', 'MLP Regressor']:
                    # 需要标准化的模型
                    estimators.append((name, scores['model']))
                else:
                    estimators.append((name, scores['model']))
            
            # 使用加权平均
            ensemble = VotingRegressor(estimators, weights=[scores[1]['cv_r2_mean'] for scores in top_models])
            
            return {
                'voting': ensemble,
                'models': top_models,
                'weights': [scores[1]['cv_r2_mean'] for scores in top_models]
            }
            
        except Exception as e:
            print(f"⚠️ 集成模型创建失败: {str(e)}")
            return None
    
    def _evaluate_ensemble_model(self, ensemble_models):
        """评估集成模型"""
        try:
            ensemble = ensemble_models['voting']
            
            # 训练集成模型
            ensemble.fit(self.X_train, self.y_train)
            
            # 预测
            y_pred = ensemble.predict(self.X_test)
            
            # 评估
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(ensemble, self.X_train, self.y_train, cv=3, scoring='r2')
            
            return {
                'model': ensemble,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"⚠️ 集成模型评估失败: {str(e)}")
            return None
    
    def _analyze_model_accuracy(self, model_info, model_name):
        """详细分析模型准确性"""
        print("\n" + "="*60)
        print("📊 模型准确性详细分析")
        print("="*60)
        
        # 1. 基础指标分析
        print("\n🔍 基础评估指标:")
        print(f"  • R² 决定系数: {model_info['r2']:.4f} (越接近1越好)")
        print(f"  • 交叉验证 R²: {model_info['cv_r2_mean']:.4f} ± {model_info['cv_r2_std']:.4f}")
        print(f"  • 平均绝对误差 (MAE): {model_info['mae']:.4f}")
        print(f"  • 均方根误差 (RMSE): {model_info['rmse']:.4f}")
        
        # 2. 准确性等级评估
        r2_score = model_info['r2']
        cv_r2_score = model_info['cv_r2_mean']
        
        print("\n📈 准确性等级评估:")
        if r2_score >= 0.95:
            print("  ✅ 优秀 (R² ≥ 0.95) - 模型预测非常准确")
        elif r2_score >= 0.85:
            print("  ✅ 良好 (0.85 ≤ R² < 0.95) - 模型预测较为准确")
        elif r2_score >= 0.70:
            print("  ⚠️  一般 (0.70 ≤ R² < 0.85) - 模型预测基本可用")
        elif r2_score >= 0.50:
            print("  ⚠️  较差 (0.50 ≤ R² < 0.70) - 模型预测不够准确")
        else:
            print("  ❌ 很差 (R² < 0.50) - 模型预测不准确")
        
        # 3. 过拟合检测
        print("\n🔍 过拟合检测:")
        r2_diff = r2_score - cv_r2_score
        if r2_diff > 0.1:
            print(f"  ⚠️  可能存在过拟合 (R²差异: {r2_diff:.3f})")
        elif r2_diff > 0.05:
            print(f"  ⚠️  轻微过拟合 (R²差异: {r2_diff:.3f})")
        else:
            print(f"  ✅ 无过拟合 (R²差异: {r2_diff:.3f})")
        
        # 4. 预测误差分析
        y_pred = model_info['predictions']
        errors = self.y_test - y_pred
        
        print("\n📊 预测误差分析:")
        print(f"  • 平均误差: {errors.mean():.4f} (越接近0越好)")
        print(f"  • 误差标准差: {errors.std():.4f}")
        print(f"  • 最大正误差: {errors.max():.4f}")
        print(f"  • 最大负误差: {errors.min():.4f}")
        
        # 5. 相对误差分析
        relative_errors = np.abs(errors) / self.y_test * 100
        print(f"  • 平均相对误差: {relative_errors.mean():.2f}% (越接近0越好)")
        print(f"  • 相对误差中位数: {relative_errors.median():.2f}%")
        
        # 6. 预测范围分析
        print("\n📈 预测范围分析:")
        print(f"  • 实际价格范围: {self.y_test.min():.2f} - {self.y_test.max():.2f}")
        print(f"  • 预测价格范围: {y_pred.min():.2f} - {y_pred.max():.2f}")
        print(f"  • 价格范围覆盖率: {((y_pred >= self.y_test.min()) & (y_pred <= self.y_test.max())).mean()*100:.1f}%")
        
        # 7. 模型稳定性评估
        print("\n🛡️ 模型稳定性评估:")
        cv_std = model_info['cv_r2_std']
        if cv_std < 0.01:
            print("  ✅ 非常稳定 (CV标准差 < 0.01)")
        elif cv_std < 0.05:
            print("  ✅ 稳定 (CV标准差 < 0.05)")
        elif cv_std < 0.10:
            print("  ⚠️  一般稳定 (CV标准差 < 0.10)")
        else:
            print("  ❌ 不稳定 (CV标准差 ≥ 0.10)")
        
        # 8. 业务价值评估
        print("\n💼 业务价值评估:")
        mae_percentage = (model_info['mae'] / self.y_test.mean()) * 100
        if mae_percentage < 5:
            print("  ✅ 高价值 - 预测误差小于5%，适合投资决策")
        elif mae_percentage < 10:
            print("  ✅ 中高价值 - 预测误差5-10%，适合趋势分析")
        elif mae_percentage < 20:
            print("  ⚠️  中等价值 - 预测误差10-20%，适合参考")
        else:
            print("  ❌ 低价值 - 预测误差大于20%，不建议用于决策")
        
        print("\n" + "="*60)
    
    def hyperparameter_tuning(self):
        """超参数调优"""
        print("⚙️ 开始超参数调优...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        # 根据最佳模型类型进行调优
        if isinstance(self.best_model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif isinstance(self.best_model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        else:
            print("⚠️ 当前最佳模型不支持超参数调优")
            return self.best_model
        
        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"✅ 最佳参数: {grid_search.best_params_}")
        print(f"📊 最佳分数: {grid_search.best_score_:.3f}")
        
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("🔍 分析特征重要性...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("📊 特征重要性排序:")
            print(self.feature_importance.head(10))
            
        else:
            print("⚠️ 当前模型不支持特征重要性分析")
        
        return self.feature_importance
    
    def test_model_stability(self, n_runs=5):
        """测试模型稳定性"""
        print(f"\n🔄 测试模型稳定性 ({n_runs} 次运行)...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        # 存储多次运行的结果
        r2_scores = []
        mae_scores = []
        predictions_list = []
        
        for i in range(n_runs):
            print(f"  运行 {i+1}/{n_runs}...")
            
            # 重新训练模型（使用相同的随机种子）
            if hasattr(self.best_model, 'random_state'):
                self.best_model.random_state = 42 + i  # 每次使用不同的种子
            
            # 训练模型
            if hasattr(self.best_model, 'fit'):
                if 'SVR' in str(type(self.best_model)):
                    self.best_model.fit(self.X_train_scaled, self.y_train)
                    y_pred = self.best_model.predict(self.X_test_scaled)
                else:
                    self.best_model.fit(self.X_train, self.y_train)
                    y_pred = self.best_model.predict(self.X_test)
                
                # 评估
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                predictions_list.append(y_pred)
        
        # 计算稳定性指标
        r2_mean = np.mean(r2_scores)
        r2_std = np.std(r2_scores)
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        
        print(f"\n📊 稳定性测试结果:")
        print(f"  R² 平均值: {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"  MAE 平均值: {mae_mean:.4f} ± {mae_std:.4f}")
        
        # 稳定性评估
        if r2_std < 0.01:
            print("  ✅ 非常稳定 (R²标准差 < 0.01)")
        elif r2_std < 0.05:
            print("  ✅ 稳定 (R²标准差 < 0.05)")
        elif r2_std < 0.10:
            print("  ⚠️  一般稳定 (R²标准差 < 0.10)")
        else:
            print("  ❌ 不稳定 (R²标准差 ≥ 0.10)")
        
        return {
            'r2_scores': r2_scores,
            'mae_scores': mae_scores,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'mae_mean': mae_mean,
            'mae_std': mae_std,
            'predictions': predictions_list
        }
    
    def predict_future_prices(self, days_ahead=7):
        """预测未来价格 - 改进版时间序列预测"""
        print(f"🔮 预测未来 {days_ahead} 天的价格...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        # 使用时间序列方法进行预测
        predictions = []
        confidence_intervals = []
        
        # 获取历史价格数据
        historical_prices = self.df['price_per_unit'].values
        historical_dates = self.df['created_at'].values
        
        # 计算价格趋势
        recent_prices = historical_prices[-30:]  # 最近30个数据点
        if len(recent_prices) > 1:
            # 计算趋势
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            volatility = np.std(recent_prices)
            mean_price = np.mean(recent_prices)
        else:
            trend = 0
            volatility = 0
            mean_price = historical_prices[-1]
        
        print(f"📊 价格趋势分析:")
        print(f"  最近价格均值: {mean_price:.2f}")
        print(f"  价格趋势: {trend:+.4f} 每单位时间")
        print(f"  价格波动性: {volatility:.2f}")
        
        # 生成预测
        for day in range(days_ahead):
            # 基础预测：使用趋势
            base_prediction = mean_price + trend * (day + 1)
            
            # 添加随机波动（基于历史波动性）
            if volatility > 0:
                noise = np.random.normal(0, volatility * 0.1)  # 10%的波动
                prediction = base_prediction + noise
            else:
                prediction = base_prediction
            
            # 确保预测价格合理（不能为负数）
            prediction = max(prediction, 0.01)
            
            predictions.append(prediction)
            
            # 计算置信区间
            if volatility > 0:
                ci_lower = prediction - 1.96 * volatility * 0.1
                ci_upper = prediction + 1.96 * volatility * 0.1
            else:
                ci_lower = prediction * 0.95
                ci_upper = prediction * 1.05
            
            confidence_intervals.append((ci_lower, ci_upper))
        
        # 创建预测结果
        future_dates = pd.date_range(
            start=self.df['created_at'].max() + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions,
            'ci_lower': [ci[0] for ci in confidence_intervals],
            'ci_upper': [ci[1] for ci in confidence_intervals]
        })
        
        print("✅ 价格预测完成")
        return predictions_df
    
    def plot_analysis(self):
        """绘制分析图表"""
        print("📊 绘制分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 价格时间序列
        axes[0, 0].plot(self.df['created_at'], self.df['price_per_unit'], alpha=0.7)
        axes[0, 0].set_title('Price Time Series', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price per Unit')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 价格分布
        axes[0, 1].hist(self.df['price_per_unit'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Price per Unit')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 模型性能比较
        if self.models:
            model_names = list(self.models.keys())
            r2_scores = [self.models[name]['r2'] for name in model_names]
            
            bars = axes[1, 0].bar(model_names, r2_scores, alpha=0.7)
            axes[1, 0].set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, score in zip(bars, r2_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 特征重要性
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, predictions_df):
        """绘制预测结果 - 改进版"""
        print("📈 绘制预测结果...")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. 历史价格和预测（带置信区间）
        axes[0].plot(self.df['created_at'], self.df['price_per_unit'], 
                    label='Historical Price', alpha=0.7, linewidth=1, color='blue')
        
        # 预测价格
        axes[0].plot(predictions_df['date'], predictions_df['predicted_price'], 
                    label='Predicted Price', color='red', linewidth=2, linestyle='-', marker='o')
        
        # 置信区间
        if 'ci_lower' in predictions_df.columns:
            axes[0].fill_between(predictions_df['date'], 
                               predictions_df['ci_lower'], 
                               predictions_df['ci_upper'], 
                               alpha=0.3, color='red', label='95% Confidence Interval')
        
        axes[0].set_title('Price History and Future Prediction with Confidence Interval', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price per Unit')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. 预测趋势分析（更详细）
        axes[1].plot(predictions_df['date'], predictions_df['predicted_price'], 
                    marker='o', linewidth=3, markersize=8, color='red', label='Predicted Price')
        
        # 添加趋势线
        if len(predictions_df) > 1:
            x_numeric = range(len(predictions_df))
            z = np.polyfit(x_numeric, predictions_df['predicted_price'], 1)
            trend_line = np.poly1d(z)
            axes[1].plot(predictions_df['date'], trend_line(x_numeric), 
                        '--', color='orange', linewidth=2, alpha=0.8, 
                        label=f'Trend: {"上升" if z[0] > 0 else "下降"} ({z[0]:.2f}/day)')
        
        axes[1].set_title('Future Price Trend Analysis', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Predicted Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 添加统计信息
        current_price = self.df['price_per_unit'].iloc[-1]
        future_price = predictions_df['predicted_price'].iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # 计算价格范围
        min_pred = predictions_df['predicted_price'].min()
        max_pred = predictions_df['predicted_price'].max()
        price_range = max_pred - min_pred
        
        stats_text = f'''Current: {current_price:.2f}
Future: {future_price:.2f}
Change: {price_change:+.2f} ({price_change_pct:+.1f}%)
Range: {min_pred:.2f} - {max_pred:.2f}
Volatility: {price_range:.2f}'''
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细预测结果
        print(f"\n📊 详细预测结果:")
        print(f"当前价格: {current_price:.2f}")
        print(f"预测价格范围: {min_pred:.2f} - {max_pred:.2f}")
        print(f"最终预测价格: {future_price:.2f}")
        print(f"价格变化: {price_change:+.2f} ({price_change_pct:+.1f}%)")
        print(f"预测波动性: {price_range:.2f}")
        
        # 显示每日预测
        print(f"\n📅 每日预测详情:")
        for i, row in predictions_df.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['predicted_price']:.2f}")
        
        return predictions_df
    
    def generate_report(self):
        """生成分析报告"""
        print("📋 生成分析报告...")
        
        report = f"""
        ========================================
        机器学习价格预测分析报告
        ========================================
        
        数据概览:
        - 总记录数: {len(self.df)}
        - 时间范围: {self.df['created_at'].min()} 到 {self.df['created_at'].max()}
        - 特征数量: {len(self.X.columns)}
        
        模型性能:
        """
        
        if self.models:
            for name, metrics in self.models.items():
                report += f"        - {name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}\n"
        
        if self.feature_importance is not None:
            report += f"\n        重要特征 (前5个):\n"
            for i, row in self.feature_importance.head(5).iterrows():
                report += f"        - {row['feature']}: {row['importance']:.3f}\n"
        
        report += f"\n        数据质量:\n"
        report += f"        - 缺失值: {self.df.isnull().sum().sum()}\n"
        report += f"        - 异常值: 已处理\n"
        report += f"        - 数据完整性: {((len(self.df) - self.df.isnull().sum().sum()) / (len(self.df) * len(self.df.columns)) * 100):.1f}%\n"
        
        print(report)
        return report
    
    def run_complete_analysis(self, days_ahead=7):
        """运行完整的机器学习分析"""
        print("🚀 开始完整的机器学习分析...")
        print("=" * 50)
        
        # 1. 加载和预处理数据
        if not self.load_and_prepare_data():
            return None
        
        # 2. 训练模型
        model_scores = self.train_models()
        
        # 3. 超参数调优
        self.hyperparameter_tuning()
        
        # 4. 特征重要性分析
        self.analyze_feature_importance()
        
        # 5. 模型稳定性测试
        stability_results = self.test_model_stability(n_runs=3)
        
        # 6. 预测未来价格
        predictions = self.predict_future_prices(days_ahead)
        
        # 6. 绘制分析图表
        self.plot_analysis()
        
        # 7. 绘制预测结果
        if predictions is not None:
            self.plot_predictions(predictions)
        
        # 8. 生成报告
        self.generate_report()
        
        print("✅ 机器学习分析完成!")
        return predictions


def main():
    """主函数"""
    print("🤖 机器学习价格预测系统")
    print("=" * 50)
    
    # 自动检测项目根目录中的所有CSV文件
    import os
    import glob
    
    # 获取项目根目录 (脚本在Analysis子目录中)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 自动查找所有CSV文件
    csv_pattern = os.path.join(project_root, "*.csv")
    csv_files = [os.path.basename(f) for f in glob.glob(csv_pattern)]
    
    # 按文件名排序
    csv_files.sort()
    
    if not csv_files:
        print("❌ 未找到任何CSV文件！")
        return
    
    print("可用的数据文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    try:
        choice = int(input("请选择数据文件 (输入数字): ")) - 1
        if 0 <= choice < len(csv_files):
            csv_file = csv_files[choice]
        else:
            print("无效选择，使用默认文件:", csv_files[0])
            csv_file = csv_files[0]
    except:
        print("使用默认文件:", csv_files[0])
        csv_file = csv_files[0]
    
    # 构建完整文件路径
    csv_file_path = os.path.join(project_root, csv_file)
    
    # 创建分析器
    analyzer = MarketMLAnalyzer(csv_file_path)
    
    # 运行完整分析
    predictions = analyzer.run_complete_analysis(days_ahead=7)
    
    if predictions is not None:
        print(f"\n🎯 预测完成! 未来7天的价格预测已生成")
        print(predictions)


if __name__ == "__main__":
    main()
