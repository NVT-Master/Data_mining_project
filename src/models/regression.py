"""
Regression Models Module - Xây dựng và huấn luyện các mô hình regression

Module này bao gồm:
- Baseline models (Linear Regression, Ridge)
- Advanced models (Random Forest, XGBoost)
- Training, prediction, cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class RegressionModels:
    """
    Lớp để xây dựng và huấn luyện các mô hình regression.
    
    Attributes:
        models (dict): Dictionary lưu các models
        trained_models (dict): Dictionary lưu trained models
        predictions (dict): Dictionary lưu predictions
    """
    
    def __init__(self):
        """Khởi tạo RegressionModels."""
        self.models = {}
        self.trained_models = {}
        self.predictions = {}
        self.train_scores = {}
        self.test_scores = {}
        
    def create_models(self) -> Dict:
        """
        Tạo các models cả baseline và advanced.
        
        Returns:
            dict: Dictionary chứa các models
        """
        print("\n" + "="*60)
        print("🏗️  TẠO CÁC MÔ HÌNH REGRESSION")
        print("="*60)
        
        # Baseline Models
        print("\n   📌 BASELINE MODELS:")
        self.models['Linear Regression'] = LinearRegression()
        print("      • Linear Regression")
        
        self.models['Ridge Regression'] = Ridge(alpha=5.0)
        print("      • Ridge Regression (alpha=5.0)")
        
        # Advanced Models
        print("\n   🚀 ADVANCED MODELS:")
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        print("      • Random Forest (300 estimators, tuned)")

        self.models['Extra Trees'] = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        print("      • Extra Trees (400 estimators)")

        self.models['Hist Gradient Boosting'] = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_depth=12,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        )
        print("      • Hist Gradient Boosting (400 iterations)")
        
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            print("      • XGBoost (100 estimators)")
        else:
            print("      • XGBoost (skip - chưa cài package xgboost)")
        
        print(f"\n   ✓ Đã tạo {len(self.models)} models")
        
        return self.models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Huấn luyện tất cả các models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            dict: Dictionary chứa trained models
        """
        print("\n" + "="*60)
        print("🎯 HUẤN LUYỆN CÁC MÔ HÌNH")
        print("="*60)
        
        print(f"\n   Training data shape: {X_train.shape}")

        use_log_target = bool((pd.Series(y_train) >= 0).all())
        if use_log_target:
            print("   ℹ️  Áp dụng log1p target transform để giảm ảnh hưởng outliers")
        
        for model_name, model in self.models.items():
            print(f"\n   {model_name}...")
            
            try:
                model_to_fit = model
                if use_log_target:
                    model_to_fit = TransformedTargetRegressor(
                        regressor=model,
                        func=np.log1p,
                        inverse_func=np.expm1,
                        check_inverse=False
                    )

                model_to_fit.fit(X_train, y_train)
                self.trained_models[model_name] = model_to_fit
                print(f"      ✓ Hoàn thành!")
                
            except Exception as e:
                print(f"      ✗ Lỗi: {str(e)}")
        
        print(f"\n   ✓ Đã huấn luyện {len(self.trained_models)}/{len(self.models)} models")
        
        return self.trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> pd.DataFrame:
        """
        Đánh giá các models trên test set.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            pd.DataFrame: Bảng kết quả đánh giá
        """
        print("\n" + "="*60)
        print("📊 ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TEST SET")
        print("="*60)
        
        print(f"\n   Test data shape: {X_test.shape}")
        
        results = []
        
        for model_name, model in self.trained_models.items():
            print(f"\n   {model_name}:")
            
            # Prediction
            y_pred = model.predict(X_test)
            self.predictions[model_name] = y_pred
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            self.test_scores[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            
            print(f"      • MAE:  {mae:.4f}")
            print(f"      • RMSE: {rmse:.4f}")
            print(f"      • R²:   {r2:.4f}")
            print(f"      • MAPE: {mape:.2f}%")
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            })
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def cross_validation(self, X: pd.DataFrame, y: pd.Series,
                        cv_folds: int = 5) -> pd.DataFrame:
        """
        Thực hiện cross-validation cho các models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Số folds
            
        Returns:
            pd.DataFrame: Bảng kết quả CV
        """
        print("\n" + "="*60)
        print(f"🔄 CROSS-VALIDATION ({cv_folds}-FOLD)")
        print("="*60)
        
        print(f"\n   Data shape: {X.shape}")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        use_log_target = bool((pd.Series(y) >= 0).all())
        
        cv_results = []
        
        for model_name, model in self.models.items():
            print(f"\n   {model_name}:")
            
            model_for_cv = model
            if use_log_target:
                model_for_cv = TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False
                )

            # R² scores
            r2_scores = cross_val_score(model_for_cv, X, y, cv=kfold, scoring='r2')
            neg_mae_scores = -cross_val_score(model_for_cv, X, y, cv=kfold, scoring='neg_mean_absolute_error')
            neg_rmse_scores = np.sqrt(-cross_val_score(model_for_cv, X, y, cv=kfold, scoring='neg_mean_squared_error'))
            
            print(f"      • R² Mean:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
            print(f"      • MAE Mean:  {neg_mae_scores.mean():.4f} ± {neg_mae_scores.std():.4f}")
            print(f"      • RMSE Mean: {neg_rmse_scores.mean():.4f} ± {neg_rmse_scores.std():.4f}")
            
            cv_results.append({
                'Model': model_name,
                'R2_Mean': r2_scores.mean(),
                'R2_Std': r2_scores.std(),
                'MAE_Mean': neg_mae_scores.mean(),
                'MAE_Std': neg_mae_scores.std(),
                'RMSE_Mean': neg_rmse_scores.mean(),
                'RMSE_Std': neg_rmse_scores.std()
            })
        
        self.cv_results_df = pd.DataFrame(cv_results)
        return self.cv_results_df
    
    def get_feature_importance(self, model_name: str = 'Random Forest',
                              feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Lấy feature importance từ model.
        
        Args:
            model_name (str): Tên model
            feature_names (list): Tên các features
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} chưa được huấn luyện!")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} không hỗ trợ feature importance!")
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Lưu model vào file.
        
        Args:
            model_name (str): Tên model
            filepath (str): Đường dẫn file
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} chưa được huấn luyện!")
        
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)
        print(f"💾 Đã lưu {model_name}: {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Tải model từ file.
        
        Args:
            model_name (str): Tên model
            filepath (str): Đường dẫn file
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"📂 Đã tải {model_name}: {filepath}")
    
    def get_best_model(self) -> Tuple[str, float]:
        """
        Lấy model có R² score tốt nhất.
        
        Returns:
            tuple: (model_name, r2_score)
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Chưa có kết quả đánh giá! Gọi evaluate_models() trước.")
        
        best_idx = self.results_df['R2'].idxmax()
        best_model = self.results_df.loc[best_idx]
        
        return best_model['Model'], best_model['R2']
    
    def compare_models_summary(self) -> pd.DataFrame:
        """
        Lấy bảng so sánh tóm tắt các models.
        
        Returns:
            pd.DataFrame: Bảng so sánh
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Chưa có kết quả! Gọi evaluate_models() trước.")
        
        return self.results_df.sort_values('R2', ascending=False)
    
    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt models.
        
        Returns:
            dict: Thông tin tóm tắt
        """
        return {
            'total_models': len(self.models),
            'trained_models': len(self.trained_models),
            'has_predictions': len(self.predictions) > 0,
            'has_cv_results': hasattr(self, 'cv_results_df')
        }


def main():
    """Hàm chính để test module."""
    print("📚 Example: Regression Models\n")
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'Feature1': np.random.rand(n_samples) * 10,
        'Feature2': np.random.rand(n_samples) * 20,
        'Feature3': np.random.rand(n_samples) * 15,
        'Feature4': np.random.rand(n_samples) * 8
    })
    
    y = pd.Series(
        X['Feature1'] * 2 + 
        X['Feature2'] * 1.5 + 
        X['Feature3'] * 0.8 + 
        np.random.randn(n_samples) * 5
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("📊 Sample Data:")
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}\n")
    
    # Khởi tạo models
    regressor = RegressionModels()
    
    # Tạo models
    regressor.create_models()
    
    # Huấn luyện
    regressor.train_models(X_train, y_train)
    
    # Đánh giá
    results = regressor.evaluate_models(X_test, y_test)
    print("\n📊 Kết quả:")
    print(results)
    
    # Cross-validation
    cv_results = regressor.cross_validation(X, y, cv_folds=5)
    print("\n🔄 Cross-validation Results:")
    print(cv_results)


if __name__ == "__main__":
    main()
