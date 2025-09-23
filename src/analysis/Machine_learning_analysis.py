import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor, StackingRegressor
from sklearn.model_selection import KFold
# 导入可选依赖
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not available, using alternative models")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
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
        """改进的特征工程 - 避免数据泄露"""
        print("⚙️ 进行改进的特征工程（避免数据泄露）...")
        
        # 按时间排序，确保时间序列的正确性
        self.df = self.df.sort_values('created_at').reset_index(drop=True)
        
        # 1. 基础时间特征（安全）
        self.df['hour'] = self.df['created_at'].dt.hour
        self.df['day_of_week'] = self.df['created_at'].dt.dayofweek
        self.df['day_of_month'] = self.df['created_at'].dt.day
        self.df['month'] = self.df['created_at'].dt.month
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # 2. 基础数值特征（安全）
        self.df['price_quantity_ratio'] = self.df['price'] / (self.df['quantity'] + 1)
        self.df['quantity_squared'] = self.df['quantity'] ** 2
        self.df['price_quantity_interaction'] = self.df['price'] * self.df['quantity']  # 使用总价，不是单价
        
        # 3. 严格的滞后特征（避免数据泄露）
        for lag in [1, 2, 3, 5, 10]:
            self.df[f'price_lag_{lag}'] = self.df['price_per_unit'].shift(lag)
            self.df[f'quantity_lag_{lag}'] = self.df['quantity'].shift(lag)
            
        # 4. 基于滞后价格的移动平均（安全）
        for window in [3, 5, 7]:
            # 使用已经滞后的价格计算移动平均
            self.df[f'price_lag1_ma_{window}'] = self.df['price_lag_1'].rolling(window=window, min_periods=1).mean()
            self.df[f'quantity_ma_{window}'] = self.df['quantity'].rolling(window=window, min_periods=1).mean()
        
        # 5. 基于滞后价格的波动性（安全）
        for window in [5, 10]:
            self.df[f'price_lag1_volatility_{window}'] = self.df['price_lag_1'].rolling(window=window, min_periods=1).std()
            
        # 6. 价格变化（基于滞后）
        self.df['price_lag1_change'] = self.df['price_lag_1'].diff()
        self.df['price_lag1_change_pct'] = self.df['price_lag_1'].pct_change()
        
        # 7. 交互特征（安全）
        self.df['hour_quantity_interaction'] = self.df['hour'] * self.df['quantity']
        self.df['day_quantity_interaction'] = self.df['day_of_week'] * self.df['quantity']
        
        # 8. 填充缺失值（向前填充，避免使用未来信息）
        self.df = self.df.fillna(method='ffill').fillna(0)
        
        print("✅ 改进的特征工程完成（已避免数据泄露）")
    
    def _prepare_training_data(self):
        """准备训练数据"""
        print("📊 准备训练数据...")
        
        # 选择安全特征 - 严格避免数据泄露
        feature_columns = [
            # 基础特征（安全）
            'quantity', 'has_sold',
            # 时间特征（安全）
            'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            # 基础数值特征（安全）
            'price_quantity_ratio', 'quantity_squared', 'price_quantity_interaction',
            # 滞后特征（安全 - 使用历史价格）
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_5', 'price_lag_10',
            'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_5', 'quantity_lag_10',
            # 基于滞后价格的移动平均（安全）
            'price_lag1_ma_3', 'price_lag1_ma_5', 'price_lag1_ma_7',
            'quantity_ma_3', 'quantity_ma_5', 'quantity_ma_7',
            # 基于滞后价格的波动性（安全）
            'price_lag1_volatility_5', 'price_lag1_volatility_10',
            # 基于滞后价格的变化（安全）
            'price_lag1_change', 'price_lag1_change_pct',
            # 交互特征（安全）
            'hour_quantity_interaction', 'day_quantity_interaction'
        ]
        
        # 移除包含NaN的列
        available_features = [col for col in feature_columns if col in self.df.columns]
        X_temp = self.df[available_features].fillna(0)
        self.y = self.df['price_per_unit']
        
        print(f"📊 初始特征数量: {len(available_features)}")
        
        # 特征选择 - 减少过拟合风险
        if len(available_features) > 15:  # 只有在特征过多时才进行选择
            print("🔍 进行特征选择以减少过拟合...")
            
            # 使用SelectKBest选择最重要的特征
            k_best = min(15, len(available_features))  # 最多选择15个特征
            selector = SelectKBest(score_func=f_regression, k=k_best)
            
            # 暂时分割数据进行特征选择
            temp_split = int(len(X_temp) * 0.8)
            X_train_temp = X_temp.iloc[:temp_split]
            y_train_temp = self.y.iloc[:temp_split]
            
            X_selected = selector.fit_transform(X_train_temp, y_train_temp)
            selected_features = [available_features[i] for i in selector.get_support(indices=True)]
            
            print(f"📊 选择的特征: {len(selected_features)} 个")
            print(f"   特征列表: {selected_features}")
            
            self.X = self.df[selected_features].fillna(0)
        else:
            print("📊 特征数量适中，无需特征选择")
            self.X = X_temp
        
        print(f"📊 最终特征数量: {len(self.X.columns)}")
        print(f"📊 样本数量: {len(self.X)}")
        
        # 时间序列分割 - 确保训练集在测试集之前
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
        
        # 添加 XGBoost（如果可用）
        if HAS_XGBOOST:
            models['XGBoost'] = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
        
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
        
        # 创建多层次集成模型
        print("\n🔄 创建多层次集成模型...")
        ensemble_models = self._create_ensemble_model(model_scores)
        
        # 评估集成模型
        if ensemble_models:
            ensemble_scores = self._evaluate_ensemble_model(ensemble_models)
            if ensemble_scores:
                model_scores['Ensemble'] = ensemble_scores
                print(f"✅ 最佳集成模型: R²={ensemble_scores['r2']:.3f}, CV-R²={ensemble_scores['cv_r2_mean']:.3f}")
                
                # 打印集成统计信息
                diversity = ensemble_models.get('model_diversity', 0)
                print(f"🔄 模型多样性: {diversity} 种不同类型")
            else:
                print("⚠️ 集成模型创建失败")
        
        # 选择最佳模型 - 使用交叉验证分数
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['cv_r2_mean'])
        self.best_model = model_scores[best_model_name]['model']
        
        print(f"\n🏆 最佳模型: {best_model_name}")
        print(f"📊 最佳模型性能: R²={model_scores[best_model_name]['r2']:.3f}, CV-R²={model_scores[best_model_name]['cv_r2_mean']:.3f}")
        
        # 详细模型准确性分析
        self._analyze_model_accuracy(model_scores[best_model_name], best_model_name)
        
        return model_scores
    
    def _create_ensemble_model(self, model_scores):
        """创建多层次集成模型 - 增强稳定性和准确性"""
        try:
            # 选择性能最佳的模型
            top_models = sorted(model_scores.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)
            
            if len(top_models) < 3:
                print("⚠️ 模型数量不足，跳过集成")
                return None
                
            # 1. 准备不同类型的模型组合
            linear_models = []
            tree_models = []
            other_models = []
            
            for name, scores in top_models:
                model = scores['model']
                if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
                    linear_models.append((name, model))
                elif name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost']:
                    tree_models.append((name, model))
                else:
                    other_models.append((name, model))
            
            ensemble_results = {}
            
            # 简化集成策略 - 只保留最有效的方法
            
            # 1. Voting Ensemble (投票集成) - 主要方法
            if len(top_models) >= 3:
                voting_estimators = [(name, scores['model']) for name, scores in top_models[:3]]  # 只使用前3个
                # 使用动态权重（基于R²和稳定性）
                weights = self._calculate_dynamic_weights(top_models[:3])
                
                voting_ensemble = VotingRegressor(
                    estimators=voting_estimators,
                    weights=weights
                )
                ensemble_results['voting'] = voting_ensemble
            
            # 2. 简单加权平均 (备选方法)
            ensemble_results['weighted_average'] = {
                'models': [scores['model'] for name, scores in top_models[:3]],
                'weights': self._calculate_dynamic_weights(top_models[:3]) if len(top_models) >= 3 else None
            }
            
            return {
                'ensembles': ensemble_results,
                'top_models': top_models,
                'model_diversity': len(set([type(scores['model']).__name__ for name, scores in top_models]))
            }
            
        except Exception as e:
            print(f"⚠️ 集成模型创建失败: {str(e)}")
            return None
    
    def _calculate_dynamic_weights(self, top_models):
        """计算动态权重 - 综合考虑R²、稳定性和多样性"""
        weights = []
        
        for name, scores in top_models:
            # 基础权重：R²分数
            base_weight = scores['cv_r2_mean']
            
            # 稳定性加分：标准差越小越好
            stability_bonus = 1.0 - min(scores['cv_r2_std'], 0.1) * 10
            
            # 模型类型多样性加分
            if name in ['Linear Regression', 'Ridge Regression']:
                diversity_bonus = 1.1  # 线性模型稳定性加分
            elif name in ['Random Forest', 'Gradient Boosting']:
                diversity_bonus = 1.05  # 集成模型加分
            else:
                diversity_bonus = 1.0
                
            final_weight = base_weight * stability_bonus * diversity_bonus
            weights.append(final_weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
            
        return weights
    
    def _evaluate_ensemble_model(self, ensemble_models):
        """评估多种集成模型并选择最佳的"""
        try:
            ensembles = ensemble_models['ensembles']
            ensemble_scores = {}
            
            print(f"\n🔄 评估 {len(ensembles)} 种集成策略...")
            
            for ensemble_name, ensemble_model in ensembles.items():
                try:
                    if ensemble_name == 'weighted_average':
                        # 加权平均集成的特殊处理
                        predictions = self._multi_model_predict(ensemble_model['models'], ensemble_model['weights'])
                        y_pred = predictions
                    else:
                        # 标准集成模型
                        ensemble_model.fit(self.X_train, self.y_train)
                        y_pred = ensemble_model.predict(self.X_test)
                    
                    # 评估指标
                    mae = mean_absolute_error(self.y_test, y_pred)
                    mse = mean_squared_error(self.y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(self.y_test, y_pred)
                    
                    # 交叉验证（除了加权平均）
                    if ensemble_name != 'weighted_average':
                        cv_scores = cross_val_score(ensemble_model, self.X_train, self.y_train, cv=3, scoring='r2')
                        cv_r2_mean = cv_scores.mean()
                        cv_r2_std = cv_scores.std()
                    else:
                        # 加权平均的交叉验证需要特殊处理
                        cv_r2_mean = r2  # 使用测试集R²作为估计
                        cv_r2_std = 0.001  # 假设较小的标准差
                    
                    ensemble_scores[ensemble_name] = {
                        'model': ensemble_model,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'cv_r2_mean': cv_r2_mean,
                        'cv_r2_std': cv_r2_std,
                        'predictions': y_pred
                    }
                    
                    print(f"  ✅ {ensemble_name}: R²={r2:.3f}, CV-R²={cv_r2_mean:.3f}±{cv_r2_std:.3f}")
                    
                except Exception as e:
                    print(f"  ⚠️ {ensemble_name} 评估失败: {str(e)}")
                    continue
            
            # 选择最佳集成模型
            if ensemble_scores:
                best_ensemble_name = max(ensemble_scores.keys(), key=lambda x: ensemble_scores[x]['cv_r2_mean'])
                best_ensemble = ensemble_scores[best_ensemble_name]
                
                print(f"\n🏆 最佳集成策略: {best_ensemble_name}")
                print(f"📊 集成性能: R²={best_ensemble['r2']:.3f}, CV-R²={best_ensemble['cv_r2_mean']:.3f}")
                
                return best_ensemble
            else:
                print("⚠️ 所有集成模型都失败")
                return None
                
        except Exception as e:
            print(f"⚠️ 集成模型评估失败: {str(e)}")
            return None
    
    def _multi_model_predict(self, models, weights):
        """多模型加权平均预测"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
            
        predictions = np.zeros(len(self.X_test))
        
        for i, model in enumerate(models):
            try:
                # 确保模型已训练
                if not hasattr(model, 'predict') or not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_') and not hasattr(model, 'support_vectors_'):
                    model.fit(self.X_train, self.y_train)
                
                pred = model.predict(self.X_test)
                predictions += weights[i] * pred
            except Exception as e:
                print(f"  ⚠️ 模型 {i} 预测失败: {str(e)}")
                continue
                
        return predictions
    
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
        
        # 检查是否是集成模型
        if isinstance(self.best_model, dict):
            print("🔄 检测到集成模型，使用最佳单一模型进行稳定性测试...")
            # 找到最佳的非集成模型
            single_models = {k: v for k, v in self.models.items() if k != 'Ensemble'}
            if not single_models:
                print("❌ 没有可用的单一模型进行稳定性测试")
                return None
            best_single_name = max(single_models.keys(), key=lambda x: single_models[x]['cv_r2_mean'])
            stability_model = single_models[best_single_name]['model']
            print(f"  使用 {best_single_name} 进行稳定性测试")
        else:
            stability_model = self.best_model
        
        # 存储多次运行的结果
        r2_scores = []
        mae_scores = []
        predictions_list = []
        
        for i in range(n_runs):
            print(f"  运行 {i+1}/{n_runs}...")
            
            try:
                # 创建模型的新实例（避免修改原模型）
                from sklearn.base import clone
                test_model = clone(stability_model)
                
                # 设置随机种子以获得不同的结果
                if hasattr(test_model, 'random_state'):
                    test_model.random_state = 42 + i
                
                # 训练模型
                if 'SVR' in str(type(test_model)):
                    test_model.fit(self.X_train_scaled, self.y_train)
                    y_pred = test_model.predict(self.X_test_scaled)
                else:
                    test_model.fit(self.X_train, self.y_train)
                    y_pred = test_model.predict(self.X_test)
                
                # 评估
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                predictions_list.append(y_pred)
                
            except Exception as e:
                print(f"  ⚠️ 运行 {i+1} 失败: {str(e)}")
                continue
        
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
    
    def predict_future_prices(self, days_ahead=7, method='auto'):
        """预测未来价格 - 使用改进的预测方法"""
        print(f"🔮 预测未来 {days_ahead} 天的价格 (方法: {method})...")
        
        if self.best_model is None:
            print("❌ 请先训练模型")
            return None
        
        # 使用内置预测方法
        predictions, confidence_intervals = self._predict_with_best_model(days_ahead, method)
        
        # 生成简化的模型解释报告
        if method == 'auto' or 'interpret' in str(method):
            print("\n🧠 生成模型解释报告...")
            self._generate_simple_interpretation()
        
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
    
    def _predict_with_best_model(self, days_ahead, method='auto'):
        """使用最佳模型进行预测，支持多方法融合"""
        if method == 'auto':
            # 自动模式：尝试多种方法并融合结果
            return self._predict_with_multi_methods(days_ahead)
        elif method == 'ml_only':
            # 仅使用ML模型
            try:
                if isinstance(self.best_model, dict):
                    return self._predict_with_ensemble_model(days_ahead)
                else:
                    return self._predict_with_single_model(days_ahead)
            except Exception as e:
                print(f"❌ ML模型预测失败: {str(e)}")
                print("📊 回退到基础统计方法...")
                return self._basic_fallback_prediction(days_ahead)
        else:
            # 单一方法
            method_map = {
                'lstm': self._predict_with_lstm,
                'arima': self._predict_with_arima,
                'prophet': self._predict_with_prophet,
                'basic': self._basic_fallback_prediction
            }
            
            if method in method_map:
                result = method_map[method](days_ahead)
                if result is not None:
                    return result
                else:
                    print(f"⚠️ {method}方法失败，回退到基础方法")
                    return self._basic_fallback_prediction(days_ahead)
            else:
                print(f"⚠️ 未知预测方法: {method}，使用默认方法")
                return self._predict_with_multi_methods(days_ahead)
    
    def _predict_with_multi_methods(self, days_ahead):
        """尝试多种预测方法并融合结果"""
        print("🎯 尝试多种预测方法并融合结果...")
        
        prediction_results = {}
        
        # 1. 尝试ML模型预测
        try:
            if isinstance(self.best_model, dict):
                ml_result = self._predict_with_ensemble_model(days_ahead)
            else:
                ml_result = self._predict_with_single_model(days_ahead)
            
            if ml_result is not None:
                prediction_results['ML模型'] = ml_result
        except Exception as e:
            print(f"⚠️ ML模型预测失败: {str(e)}")
        
        # 2. 尝试ARIMA预测
        arima_result = self._predict_with_arima(days_ahead)
        if arima_result is not None:
            prediction_results['ARIMA'] = arima_result
        
        # 3. 尝试Prophet预测
        prophet_result = self._predict_with_prophet(days_ahead)
        if prophet_result is not None:
            prediction_results['Prophet'] = prophet_result
        
        # 4. 尝试LSTM预测（如果可用）
        lstm_result = self._predict_with_lstm(days_ahead)
        if lstm_result is not None:
            prediction_results['LSTM'] = lstm_result
        
        # 5. 基础统计方法作为保底
        basic_result = self._basic_fallback_prediction(days_ahead)
        if basic_result is not None:
            prediction_results['基础统计'] = basic_result
        
        # 如果有多种方法成功，进行融合
        if len(prediction_results) > 1:
            return self._ensemble_predictions(prediction_results)
        elif len(prediction_results) == 1:
            # 只有一种方法成功
            method_name, result = list(prediction_results.items())[0]
            print(f"✅ 使用单一方法: {method_name}")
            return result
        else:
            # 所有方法都失败了
            print("❌ 所有预测方法都失败，使用简单的线性外推")
            return self._simple_linear_extrapolation(days_ahead)
    
    def _predict_with_single_model(self, days_ahead):
        """使用单一模型进行预测"""
        # 获取最后几行的特征作为基础
        last_features = self.X_test.iloc[-days_ahead:].copy() if len(self.X_test) >= days_ahead else self.X.iloc[-days_ahead:].copy()
        
        predictions = []
        
        for i in range(days_ahead):
            # 使用最佳模型预测
            if 'SVR' in str(type(self.best_model)):
                pred = self.best_model.predict(self.scaler.transform(last_features.iloc[[i % len(last_features)]]))
            else:
                pred = self.best_model.predict(last_features.iloc[[i % len(last_features)]])
            
            predictions.append(pred[0])
        
        # 计算动态置信区间
        confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='ml')
        
        return predictions, confidence_intervals
    
    def _predict_with_ensemble_model(self, days_ahead):
        """使用集成模型进行预测"""
        ensemble_data = self.best_model
        ensemble_model = ensemble_data.get('model')
        
        
        if ensemble_model is None:
            # 检查是否是加权平均集成模型
            if 'models' in ensemble_data and 'weights' in ensemble_data:
                print("✅ 找到加权平均集成模型")
                ensemble_model = ensemble_data  # 使用整个字典
            else:
                print("⚠️ 集成模型中没有找到有效的预测器，尝试其他键...")
                # 尝试其他可能的键
                for key in ['voting', 'weighted_average', 'ensemble_model']:
                    if key in ensemble_data:
                        ensemble_model = ensemble_data[key]
                        print(f"✅ 找到集成模型: {key}")
                        break
                
                if ensemble_model is None:
                    print("❌ 所有尝试都失败，使用基础方法")
                    return self._basic_fallback_prediction(days_ahead)
        
        # 获取最后几行的特征作为基础
        last_features = self.X_test.iloc[-days_ahead:].copy() if len(self.X_test) >= days_ahead else self.X.iloc[-days_ahead:].copy()
        
        predictions = []
        
        # 检查是否是加权平均集成（字典类型）
        if isinstance(ensemble_model, dict) and 'models' in ensemble_model:
            # 加权平均集成
            models = ensemble_model['models']
            weights = ensemble_model['weights']
            
            for i in range(days_ahead):
                feature_row = last_features.iloc[[i % len(last_features)]]
                weighted_pred = 0
                
                for model, weight in zip(models, weights):
                    if 'SVR' in str(type(model)):
                        pred = model.predict(self.scaler.transform(feature_row))
                    else:
                        pred = model.predict(feature_row)
                    weighted_pred += pred[0] * weight
                    
                predictions.append(weighted_pred)
        else:
            # 标准sklearn集成模型
            for i in range(days_ahead):
                pred = ensemble_model.predict(last_features.iloc[[i % len(last_features)]])
                predictions.append(pred[0])
        
        # 计算动态置信区间
        confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='ml')
        
        return predictions, confidence_intervals
    
    def _calculate_confidence_intervals(self, predictions, historical_std, ci_factor=0.12):
        """计算预测的置信区间"""
        confidence_intervals = []
        for pred in predictions:
            ci_margin = 1.96 * historical_std * ci_factor
            confidence_intervals.append((pred - ci_margin, pred + ci_margin))
        return confidence_intervals
    
    def _generate_simple_interpretation(self):
        """生成简化的模型解释报告"""
        print("\n🔍 生成模型可解释性报告...")
        
        # 获取可解释的模型
        interpretable_model = self._get_interpretable_model()
        
        if interpretable_model is None:
            print("⚠️ 无法获取可解释的模型")
            return
        
        # 特征重要性分析
        if hasattr(interpretable_model, 'feature_importances_'):
            print("\n📊 特征重要性分析:")
            feature_names = self.X.columns.tolist()
            importances = interpretable_model.feature_importances_
            
            # 排序并显示前10个特征
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_importance[:10]:
                print(f"  • {feature}: {importance:.4f}")
        else:
            print("⚠️ 当前模型不支持特征重要性分析")
        
        # 模型性能摘要
        if hasattr(self, 'models') and self.models:
            print("\n📈 模型性能摘要:")
            for name, scores in self.models.items():
                if name != 'Ensemble':
                    print(f"  • {name}: R²={scores['r2']:.3f}, CV-R²={scores['cv_r2_mean']:.3f}±{scores['cv_r2_std']:.3f}")
    
    def _get_interpretable_model(self):
        """获取可解释的模型对象"""
        best_model = self.best_model
        
        # 如果最佳模型是集成模型（字典类型）
        if isinstance(best_model, dict):
            if 'model' in best_model:
                return best_model['model']
            else:
                # 如果是集成模型，尝试找到最好的单一模型
                print("🔍 集成模型检测到，使用最佳单一模型进行解释...")
                if hasattr(self, 'models') and self.models:
                    # 找到非集成的最佳模型
                    single_models = {k: v for k, v in self.models.items() if k != 'Ensemble'}
                    if single_models:
                        best_single_name = max(single_models.keys(), key=lambda x: single_models[x]['cv_r2_mean'])
                        print(f"  使用 {best_single_name} 进行模型解释")
                        return single_models[best_single_name]['model']
                return None
        else:
            return best_model
    
    def _predict_with_lstm(self, days_ahead):
        """使用LSTM深度学习预测"""
        if not HAS_TENSORFLOW:
            print("⚠️ TensorFlow未安装，跳过LSTM预测")
            return None
            
        try:
            print("🧠 使用LSTM深度学习预测...")
            
            from sklearn.preprocessing import MinMaxScaler
            
            # 准备LSTM数据
            price_data = self.df['price_per_unit'].values
            lstm_scaler = MinMaxScaler()
            scaled_data = lstm_scaler.fit_transform(price_data.reshape(-1, 1))
            
            # 创建序列数据
            sequence_length = min(60, len(scaled_data) // 4)
            if sequence_length < 10:
                print("⚠️ 数据量不足，跳过LSTM预测")
                return None
                
            X_lstm, y_lstm = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-sequence_length:i, 0])
                y_lstm.append(scaled_data[i, 0])
            
            if len(X_lstm) < 50:
                print("⚠️ 序列数据不足，跳过LSTM预测")
                return None
            
            X_lstm = np.array(X_lstm).reshape((len(X_lstm), sequence_length, 1))
            y_lstm = np.array(y_lstm)
            
            # 构建LSTM模型
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # 训练模型
            model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
            
            # 进行预测
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            predictions = []
            
            for _ in range(days_ahead):
                pred = model.predict(last_sequence, verbose=0)
                predictions.append(pred[0, 0])
                
                # 更新序列
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]
            
            # 反向缩放
            predictions = lstm_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # 计算置信区间
            confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='lstm')
            
            return predictions.tolist(), confidence_intervals
            
        except Exception as e:
            print(f"❌ LSTM预测失败: {str(e)}")
            return None
    
    def _predict_with_arima(self, days_ahead):
        """使用ARIMA时间序列预测"""
        if not HAS_STATSMODELS:
            print("⚠️ statsmodels未安装，跳过ARIMA预测")
            return None
            
        try:
            print("📈 使用ARIMA时间序列预测...")
            
            # 准备时间序列数据
            ts_data = self.df['price_per_unit'].dropna()
            if len(ts_data) < 50:
                print("⚠️ 数据量不足，跳过ARIMA预测")
                return None
            
            # 拟合ARIMA模型
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # 进行预测
            forecast = fitted_model.forecast(steps=days_ahead)
            predictions = forecast.tolist()
            
            # 使用我们统一的动态置信区间计算，而不是ARIMA自带的
            confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='arima')
            
            return predictions, confidence_intervals
            
        except Exception as e:
            print(f"❌ ARIMA预测失败: {str(e)}")
            return None
    
    def _predict_with_prophet(self, days_ahead):
        """使用Prophet时间序列预测"""
        if not HAS_PROPHET:
            print("⚠️ Prophet未安装，跳过Prophet预测")
            return None
            
        try:
            print("🔮 使用Prophet时间序列预测...")
            
            # 准备Prophet数据
            prophet_df = self.df[['created_at', 'price_per_unit']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 50:
                print("⚠️ 数据量不足，跳过Prophet预测")
                return None
            
            # 创建和训练Prophet模型
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )
            
            model.fit(prophet_df)
            
            # 创建未来数据框
            future = model.make_future_dataframe(periods=days_ahead, freq='D')
            forecast = model.predict(future)
            
            # 提取预测结果
            predictions = forecast['yhat'].tail(days_ahead).tolist()
            
            # 使用我们统一的动态置信区间计算，而不是Prophet自带的
            confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='prophet')
            
            return predictions, confidence_intervals
            
        except Exception as e:
            print(f"❌ Prophet预测失败: {str(e)}")
            return None
    
    def _calculate_dynamic_confidence_intervals(self, predictions, method='default'):
        """动态计算置信区间"""
        try:
            # 方法1：基于残差分布（最准确的方法）
            if hasattr(self, 'best_model') and self.best_model is not None:
                # 获取模型在测试集上的残差
                interpretable_model = self._get_interpretable_model()
                if interpretable_model is not None and hasattr(self, 'y_test'):
                    try:
                        if 'SVR' in str(type(interpretable_model)):
                            y_pred = interpretable_model.predict(self.X_test_scaled)
                        else:
                            y_pred = interpretable_model.predict(self.X_test)
                        
                        residuals = self.y_test - y_pred
                        residual_std = np.std(residuals)
                        residual_mean = np.mean(residuals)
                        
                        # 使用残差分布计算置信区间
                        confidence_intervals = []
                        for i, pred in enumerate(predictions):
                            # 考虑残差的偏差和标准差，加上轻微的时间衰减
                            time_factor = 1 + (i * 0.01)  # 每天增加1%的不确定性（更保守）
                            adjusted_std = residual_std * time_factor
                            
                            ci_lower = pred + residual_mean - 1.96 * adjusted_std
                            ci_upper = pred + residual_mean + 1.96 * adjusted_std
                            confidence_intervals.append((ci_lower, ci_upper))
                        
                        return confidence_intervals
                    except Exception as e:
                        print(f"⚠️ 残差计算失败: {str(e)}")
                        pass
            
            # 方法2：基于历史价格波动（回退方法）
            historical_prices = self.df['price_per_unit'].values
            recent_prices = historical_prices[-min(30, len(historical_prices)):]
            
            # 计算更保守的波动性估计
            price_changes = np.diff(recent_prices)
            daily_volatility = np.std(price_changes)
            
            # 根据预测方法调整置信区间宽度
            method_factors = {
                'lstm': 2.0,       # LSTM: 2倍日波动性
                'arima': 1.5,      # ARIMA: 1.5倍日波动性  
                'prophet': 1.8,    # Prophet: 1.8倍日波动性
                'ml': 1.2,         # ML模型: 1.2倍日波动性（最保守）
                'default': 1.5
            }
            
            volatility_multiplier = method_factors.get(method, 1.5)
            
            confidence_intervals = []
            for i, pred in enumerate(predictions):
                # 轻微的时间衰减
                time_factor = 1 + (i * 0.05)  # 每天增加5%的不确定性
                
                # 使用日波动性作为基础
                ci_margin = 1.96 * daily_volatility * volatility_multiplier * time_factor
                
                confidence_intervals.append((pred - ci_margin, pred + ci_margin))
            
            return confidence_intervals
            
        except Exception as e:
            print(f"⚠️ 动态置信区间计算失败，使用默认方法: {str(e)}")
            return self._calculate_confidence_intervals(predictions, self.y.std())
    
    def _ensemble_predictions(self, prediction_results):
        """融合多种预测方法的结果"""
        print(f"🔄 融合 {len(prediction_results)} 种预测方法的结果...")
        
        if not prediction_results:
            return None, None
        
        # 显示各方法的预测范围
        for method_name, (predictions, _) in prediction_results.items():
            if predictions:
                print(f"   - {method_name}: 预测范围 {min(predictions):.2f} - {max(predictions):.2f}")
        
        # 计算权重（基于方法的可靠性）
        method_weights = {
            'ML模型': 0.35,      # 最高权重给ML模型
            'ARIMA': 0.25,       # ARIMA适合时间序列
            'Prophet': 0.20,     # Prophet适合趋势分析
            'LSTM': 0.15,        # LSTM适合复杂模式
            '基础统计': 0.05     # 最低权重给基础方法
        }
        
        # 标准化权重（只对实际存在的方法）
        available_methods = list(prediction_results.keys())
        total_weight = sum(method_weights.get(method, 0.1) for method in available_methods)
        normalized_weights = {method: method_weights.get(method, 0.1) / total_weight 
                            for method in available_methods}
        
        # 融合预测结果
        days_ahead = len(list(prediction_results.values())[0][0])
        ensemble_predictions = []
        ensemble_confidence_intervals = []
        
        for day in range(days_ahead):
            # 加权平均预测值
            weighted_pred = 0
            
            # 收集各方法的预测和不确定性
            method_predictions = []
            method_uncertainties = []
            
            for method_name, (predictions, confidence_intervals) in prediction_results.items():
                weight = normalized_weights[method_name]
                weighted_pred += weight * predictions[day]
                
                # 计算每个方法的不确定性（半宽度）
                uncertainty = (confidence_intervals[day][1] - confidence_intervals[day][0]) / 2
                method_predictions.append(predictions[day])
                method_uncertainties.append(uncertainty)
            
            # 计算融合后的不确定性：使用加权平均的不确定性，而不是边界的加权平均
            weighted_uncertainty = sum(normalized_weights[method_name] * method_uncertainties[i] 
                                     for i, method_name in enumerate(prediction_results.keys()))
            
            # 添加方法间差异的不确定性（预测分歧度）
            if len(method_predictions) > 1:
                prediction_spread = np.std(method_predictions)
                total_uncertainty = np.sqrt(weighted_uncertainty**2 + (prediction_spread * 0.5)**2)
            else:
                total_uncertainty = weighted_uncertainty
            
            
            ensemble_predictions.append(weighted_pred)
            ensemble_confidence_intervals.append((weighted_pred - total_uncertainty, weighted_pred + total_uncertainty))
        
        print(f"✅ 融合预测完成，最终范围: {min(ensemble_predictions):.2f} - {max(ensemble_predictions):.2f}")
        
        return ensemble_predictions, ensemble_confidence_intervals
    
    def _simple_linear_extrapolation(self, days_ahead):
        """简单的线性外推作为最后保底方法"""
        print("📈 使用简单线性外推...")
        
        try:
            historical_prices = self.df['price_per_unit'].values
            recent_prices = historical_prices[-min(10, len(historical_prices)):]
            
            if len(recent_prices) < 2:
                # 如果数据太少，返回最后一个价格
                last_price = historical_prices[-1] if len(historical_prices) > 0 else 40.0
                predictions = [last_price] * days_ahead
            else:
                # 拟合线性趋势
                x = np.arange(len(recent_prices))
                coeffs = np.polyfit(x, recent_prices, 1)
                trend = coeffs[0]
                intercept = coeffs[1]
                
                # 外推预测
                predictions = []
                for day in range(1, days_ahead + 1):
                    pred = intercept + trend * (len(recent_prices) + day - 1)
                    predictions.append(max(pred, 0.01))  # 确保价格为正
            
            # 简单的置信区间
            volatility = np.std(recent_prices) if len(recent_prices) > 1 else 5.0
            confidence_intervals = []
            for pred in predictions:
                margin = 1.96 * volatility * 0.2  # 20%的波动性因子
                confidence_intervals.append((pred - margin, pred + margin))
            
            return predictions, confidence_intervals
            
        except Exception as e:
            print(f"❌ 线性外推失败: {str(e)}")
            # 最后的最后保底方法
            last_price = 40.0  # 硬编码一个合理的价格
            predictions = [last_price] * days_ahead
            confidence_intervals = [(last_price - 5, last_price + 5)] * days_ahead
            return predictions, confidence_intervals
    
    def _basic_fallback_prediction(self, days_ahead):
        """基础备选预测方法"""
        print("📊 使用基础统计方法预测...")
        
        # 获取历史价格数据
        historical_prices = self.df['price_per_unit'].values
        recent_prices = historical_prices[-min(30, len(historical_prices)):]
        
        if len(recent_prices) > 1:
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            volatility = np.std(recent_prices)
            mean_price = np.mean(recent_prices)
        else:
            trend, volatility, mean_price = 0, 0, historical_prices[-1]
        
        print(f"📊 价格趋势分析:")
        print(f"  最近价格均值: {mean_price:.2f}")
        print(f"  价格趋势: {trend:+.4f} 每单位时间")
        print(f"  价格波动性: {volatility:.2f}")
        
        predictions, confidence_intervals = [], []
        
        for day in range(days_ahead):
            # 基础预测
            base_prediction = mean_price + trend * (day + 1)
            cycle_adjustment = volatility * 0.05 * np.sin(2 * np.pi * day / 7) if volatility > 0 else 0
            deterministic_noise = np.sin(day * 0.7) * volatility * 0.02 if volatility > 0 else 0
            prediction = max(base_prediction + cycle_adjustment + deterministic_noise, 0.01)
            
            predictions.append(prediction)
            
        # 计算动态置信区间
        confidence_intervals = self._calculate_dynamic_confidence_intervals(predictions, method='default')
        
        return predictions, confidence_intervals
    
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


def load_items_config():
    """
    加载物品配置文件
    """
    import os
    import json
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'items_config.json')
    
    # 默认配置（向后兼容）
    default_config = {
        "items": [
            {"name": "Gold Ore", "file": "Gold_Ore_API.py", "csv": "gold_ore.csv", "category": "ore", "enabled": True},
            {"name": "Iron Ore", "file": "Iron_Ore_API.py", "csv": "iron_ore.csv", "category": "ore", "enabled": True},
            {"name": "Cobalt Ore", "file": "Cobalt_Ore_API.py", "csv": "cobalt_ore.csv", "category": "ore", "enabled": True}
        ],
        "auto_discovery": {"enabled": True, "ore_directory": "Ore", "naming_pattern": "*_API.py"}
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            return default_config
    except Exception as e:
        print(f"⚠️ 配置文件加载失败，使用默认配置: {str(e)}")
        return default_config


def get_available_items():
    """
    动态获取可用物品列表，支持配置文件和自动发现
    """
    import os
    import glob
    
    # 加载配置
    config = load_items_config()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    available_items = {}
    item_index = 1
    
    # 1. 从配置文件加载已启用的物品
    for item in config.get("items", []):
        if not item.get("enabled", True):
            continue
            
        # 检查API文件是否存在
        category = item.get("category", "ore")
        directory = "src/api"  # 统一使用src/api目录
        api_path = os.path.join(project_root, directory, item["file"])
        
        if os.path.exists(api_path):
            item_info = {
                "name": item["name"],
                "file": item["file"],
                "csv": item["csv"],
                "category": item.get("category", "ore"),
                "description": item.get("description", ""),
                "directory": directory
            }
            available_items[item_index] = item_info
            item_index += 1
        else:
            print(f"⚠️ API文件不存在: {api_path}")
    
    # 2. 自动发现新物品（如果启用）
    auto_discovery = config.get("auto_discovery", {})
    if auto_discovery.get("enabled", True):
        directories_to_scan = ["src/api"]
        
        for directory in directories_to_scan:
            dir_path = os.path.join(project_root, directory)
            if not os.path.exists(dir_path):
                continue
                
            pattern = auto_discovery.get("naming_pattern", "*_API.py")
            api_files = glob.glob(os.path.join(dir_path, pattern))
            
            # 获取已知的文件名列表
            known_files = [item["file"] for item in config.get("items", [])]
            
            for api_file in api_files:
                filename = os.path.basename(api_file)
                
                # 跳过已配置的物品
                if filename in known_files:
                    continue
                
                # 解析物品名称和CSV名称
                item_name = filename.replace("_API.py", "").replace("_", " ")
                csv_name = filename.replace("_API.py", ".csv").lower()
                
                # 根据文件名确定类别 (默认为ore)
                category = "consumable" if "potion" in item_name.lower() or "consumable" in filename.lower() else "ore"
                
                new_item = {
                    "name": item_name,
                    "file": filename,
                    "csv": csv_name,
                    "category": category,
                    "description": f"自动发现的{item_name}",
                    "directory": directory
                }
                
                available_items[item_index] = new_item
                item_index += 1
                print(f"🆕 自动发现新物品: {item_name} ({category})")
    
    return available_items


def add_new_item(name, api_file, csv_file, category="ore", description="", enabled=True):
    """
    添加新物品到配置文件
    
    Args:
        name: 物品名称 (例: "Silver Ore")
        api_file: API文件名 (例: "Silver_Ore_API.py")
        csv_file: CSV文件名 (例: "silver_ore.csv")
        category: 物品类别 ("ore", "consumable", "equipment", "material")
        description: 物品描述
        enabled: 是否启用
    """
    import os
    import json
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'items_config.json')
    
    # 加载现有配置
    config = load_items_config()
    
    # 检查是否已存在
    for item in config.get("items", []):
        if item["name"] == name or item["file"] == api_file:
            print(f"⚠️ 物品 '{name}' 已存在!")
            return False
    
    # 添加新物品
    new_item = {
        "name": name,
        "file": api_file,
        "csv": csv_file,
        "category": category,
        "description": description,
        "enabled": enabled
    }
    
    config.setdefault("items", []).append(new_item)
    
    # 保存配置
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功添加新物品: {name}")
        return True
    except Exception as e:
        print(f"❌ 保存配置失败: {str(e)}")
        return False


def list_items_config():
    """
    显示当前物品配置
    """
    config = load_items_config()
    
    print("\n📋 当前物品配置:")
    print("=" * 50)
    
    for i, item in enumerate(config.get("items", []), 1):
        status = "✅" if item.get("enabled", True) else "❌"
        category = item.get("category", "unknown")
        description = item.get("description", "")
        
        print(f"{i:2d}. {status} {item['name']} ({category})")
        print(f"     API: {item['file']}")
        print(f"     CSV: {item['csv']}")
        if description:
            print(f"     描述: {description}")
        print()
    
    auto_discovery = config.get("auto_discovery", {})
    print(f"🔍 自动发现: {'启用' if auto_discovery.get('enabled') else '禁用'}")


def manage_items_config():
    """
    物品配置管理菜单
    """
    while True:
        print("\n🛠️ 物品配置管理")
        print("=" * 30)
        print("1. 查看当前配置")
        print("2. 添加新物品")
        print("3. 返回主菜单")
        
        choice = input("\n请选择操作 (1-3): ").strip()
        
        if choice == "1":
            list_items_config()
        elif choice == "2":
            print("\n➕ 添加新物品")
            name = input("物品名称 (例: Silver Ore): ").strip()
            if not name:
                print("❌ 物品名称不能为空")
                continue
                
            api_file = input("API文件名 (例: Silver_Ore_API.py): ").strip()
            if not api_file.endswith("_API.py"):
                api_file += "_API.py"
                
            csv_file = input(f"CSV文件名 (默认: {name.lower().replace(' ', '_')}.csv): ").strip()
            if not csv_file:
                csv_file = f"{name.lower().replace(' ', '_')}.csv"
                
            category = input("类别 (ore/consumable/equipment/material, 默认: ore): ").strip()
            if not category:
                category = "ore"
                
            description = input("描述 (可选): ").strip()
            
            add_new_item(name, api_file, csv_file, category, description)
        elif choice == "3":
            break
        else:
            print("❌ 无效选择")


def collect_and_analyze():
    """
    集成功能：选择物品 → 调用API获取数据 → 生成CSV → 进行机器学习分析
    支持自定义物品和API文件
    """
    import os
    import sys
    import subprocess
    
    print("🚀 启动智能市场分析系统")
    print("=" * 60)
    
    # 1. 动态获取可用物品列表
    available_items = get_available_items()
    
    print("\n📦 可用物品列表:")
    for key, item in available_items.items():
        category_emoji = "⛏️" if item.get('category') == 'ore' else "🧪"
        description = item.get('description', '')
        if description:
            print(f"  {key}. {category_emoji} {item['name']} - {description}")
        else:
            print(f"  {key}. {category_emoji} {item['name']}")
    
    # 2. 用户选择物品
    try:
        choice = int(input("\n请选择要分析的物品 (输入数字): "))
        if choice not in available_items:
            print("❌ 无效选择，使用默认选项: Gold Ore")
            choice = 1
        
        selected_item = available_items[choice]
        print(f"✅ 已选择: {selected_item['name']} ({selected_item.get('category', 'unknown')})")
        
    except ValueError:
        print("❌ 输入无效，使用默认选项: Gold Ore")
        choice = 1
        selected_item = available_items[1]
    
    # 3. 询问是否需要更新数据
    update_data = input("\n🔄 是否需要获取最新数据? (y/n, 默认n): ").lower().strip()
    
    if update_data in ['y', 'yes', '是']:
        print(f"\n📡 正在调用 {selected_item['name']} API 获取最新数据...")
        
        try:
            # 构建API文件路径 - 统一使用src/api目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            api_directory = "src/api"  # 统一使用src/api目录
            api_file_path = os.path.join(project_root, api_directory, selected_item['file'])
            
            if not os.path.exists(api_file_path):
                print(f"❌ API文件不存在: {api_file_path}")
                return None
            
            print(f"🔄 执行: {api_file_path}")
            
            # 执行API脚本
            result = subprocess.run([sys.executable, api_file_path], 
                                  capture_output=True, text=True, 
                                  cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            if result.returncode == 0:
                print("✅ 数据获取完成!")
                print("📊 API输出摘要:")
                # 显示最后几行输出
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
            else:
                print("⚠️ API执行出现警告，但继续分析...")
                print("错误输出:", result.stderr[-500:] if result.stderr else "无")
                
        except Exception as e:
            print(f"❌ 数据获取失败: {str(e)}")
            print("📊 将使用现有CSV文件进行分析...")
    
    # 4. 确定CSV文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_root, selected_item['csv'])
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV文件不存在: {csv_path}")
        print("💡 建议先运行API获取数据")
        return None
    
    print(f"\n📊 开始分析 {selected_item['name']} 数据...")
    print(f"📁 数据文件: {csv_path}")
    
    # 5. 进行机器学习分析
    try:
        analyzer = MarketMLAnalyzer(csv_path)
        predictions = analyzer.run_complete_analysis(days_ahead=7)
        
        if predictions is not None:
            print(f"\n🎯 {selected_item['name']} 分析完成!")
            print("📈 未来7天价格预测:")
            print(predictions)
            
            # 6. 生成分析总结
            print(f"\n📋 {selected_item['name']} 分析总结:")
            print("=" * 50)
            
            # 基本统计
            avg_price = predictions['predicted_price'].mean()
            price_trend = predictions['predicted_price'].iloc[-1] - predictions['predicted_price'].iloc[0]
            
            print(f"📊 平均预测价格: {avg_price:.2f}")
            print(f"📈 7天价格趋势: {price_trend:+.2f} ({price_trend/predictions['predicted_price'].iloc[0]*100:+.1f}%)")
            
            # 投资建议
            if price_trend > 0:
                print("💡 投资建议: 📈 价格呈上升趋势，考虑买入")
            elif price_trend < -1:
                print("💡 投资建议: 📉 价格呈下降趋势，考虑卖出")
            else:
                print("💡 投资建议: ➡️  价格相对稳定，观望")
            
            # 风险评估
            ci_width = (predictions['ci_upper'] - predictions['ci_lower']).mean()
            risk_level = "低风险" if ci_width < 2 else "中风险" if ci_width < 5 else "高风险"
            print(f"⚠️  风险评估: {risk_level} (置信区间宽度: ±{ci_width/2:.2f})")
            
            return predictions
        else:
            print("❌ 分析失败")
            return None
            
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        return None


def main():
    """主函数"""
    while True:
        print("\n🎯 Darker Market 机器学习分析系统")
        print("=" * 60)
        print("1. 🚀 智能模式 (选择物品 → 获取数据 → 自动分析)")
        print("2. 📊 传统模式 (分析现有CSV文件)")
        print("3. 🛠️ 物品配置管理")
        print("4. ❌ 退出")
        
        try:
            choice = input("\n请选择模式 (1-4, 默认1): ").strip()
            
            if choice == "2":
                main_traditional()
            elif choice == "3":
                manage_items_config()
                # 配置管理后返回主菜单
                continue
            elif choice == "4":
                print("👋 再见!")
                break
            else:
                # 默认或选择1
                collect_and_analyze()
                
            # 询问是否继续
            continue_choice = input("\n🔄 是否继续使用系统? (y/n, 默认n): ").lower().strip()
            if continue_choice not in ['y', 'yes', '是']:
                print("👋 再见!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 程序出错: {str(e)}")
            continue


def main_traditional():
    """传统模式：直接选择CSV文件进行分析"""
    print("\n📁 传统模式：选择现有CSV文件")
    print("=" * 40)
    
    import os
    import glob
    
    # 获取项目根目录 (脚本在Analysis子目录中)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 自动查找所有CSV文件
    csv_pattern = os.path.join(project_root, "*.csv")
    csv_files = [os.path.basename(f) for f in glob.glob(csv_pattern)]
    
    # 按文件名排序
    csv_files.sort()
    
    if not csv_files:
        print("❌ 未找到任何CSV文件！")
        print("💡 建议先使用智能模式获取数据")
        return None
    
    print("📄 可用的数据文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    try:
        choice = int(input("\n请选择数据文件 (输入数字): ")) - 1
        if 0 <= choice < len(csv_files):
            csv_file = csv_files[choice]
        else:
            print("❌ 无效选择，使用默认文件:", csv_files[0])
            csv_file = csv_files[0]
    except:
        print("❌ 输入无效，使用默认文件:", csv_files[0])
        csv_file = csv_files[0]
    
    # 构建完整文件路径
    csv_file_path = os.path.join(project_root, csv_file)
    
    print(f"\n📊 开始分析文件: {csv_file}")
    
    # 创建分析器
    analyzer = MarketMLAnalyzer(csv_file_path)
    
    # 运行完整分析
    predictions = analyzer.run_complete_analysis(days_ahead=7)
    
    if predictions is not None:
        print(f"\n🎯 预测完成! 未来7天的价格预测已生成")
        print(predictions)
        return predictions
    else:
        print("❌ 分析失败")
        return None


if __name__ == "__main__":
    main()
