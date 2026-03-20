"""
Feature Engineering Module - Tạo và chọn features

Module này chịu trách nhiệm:
- Tạo features mới từ dữ liệu hiện có
- Lựa chọn features quan trọng
- Thống kê feature importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """
    Lớp để tạo và chọn features.
    
    Attributes:
        feature_importance (dict): Lưu feature importance scores
        selected_features (list): Danh sách features được chọn
    """
    
    def __init__(self):
        """Khởi tạo FeatureBuilder."""
        self.feature_importance = {}
        self.selected_features = []
        self.selector = None
        
    def create_polynomial_features(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2) -> pd.DataFrame:
        """
        Tạo polynomial features.
        
        Args:
            df (pd.DataFrame): DataFrame input
            columns (list): Danh sách cột (nếu None thì tất cả numeric)
            degree (int): Bậc của polynomial
            
        Returns:
            pd.DataFrame: DataFrame với polynomial features
        """
        print("\n" + "="*60)
        print("🔢 TẠO POLYNOMIAL FEATURES")
        print("="*60)
        
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n   Số features gốc: {len(columns)}")
        print(f"   Bậc: {degree}")
        
        new_features = 0
        for col in columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_pow{d}"
                df_copy[new_col] = df_copy[col] ** d
                new_features += 1
        
        print(f"   ✓ Tạo được {new_features} features mới")
        print(f"   Shape sau: {df_copy.shape}")
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   max_interactions: int = None) -> pd.DataFrame:
        """
        Tạo interaction features (tích của 2 features).
        
        Args:
            df (pd.DataFrame): DataFrame input
            columns (list): Danh sách cột (nếu None thì top features)
            max_interactions (int): Số tương tác tối đa
            
        Returns:
            pd.DataFrame: DataFrame với interaction features
        """
        print("\n" + "="*60)
        print("🔗 TẠO INTERACTION FEATURES")
        print("="*60)
        
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()[:5]
        
        print(f"\n   Số features được xét: {len(columns)}")
        
        new_features = 0
        count = 0
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if max_interactions and count >= max_interactions:
                    break
                
                col1, col2 = columns[i], columns[j]
                new_col = f"{col1}_x_{col2}"
                df_copy[new_col] = df_copy[col1] * df_copy[col2]
                new_features += 1
                count += 1
        
        print(f"   ✓ Tạo được {new_features} interaction features")
        print(f"   Shape sau: {df_copy.shape}")
        
        return df_copy
    
    def create_ratio_features(self, df: pd.DataFrame,
                             pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Tạo ratio features (tỷ lệ của 2 features).
        
        Args:
            df (pd.DataFrame): DataFrame input
            pairs (list): Danh sách các cặp (col1, col2) để tính tỷ lệ
            
        Returns:
            pd.DataFrame: DataFrame với ratio features
        """
        print("\n" + "="*60)
        print("📐 TẠO RATIO FEATURES")
        print("="*60)
        
        df_copy = df.copy()
        
        if pairs is None:
            # Tự động tạo một số ratio features
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                pairs = [(numeric_cols[0], numeric_cols[1])]
        
        if pairs is None:
            print("   Không có cặp features để tạo ratio!")
            return df_copy
        
        print(f"\n   Số cặp được xét: {len(pairs)}")
        
        new_features = 0
        for col1, col2 in pairs:
            if col1 in df_copy.columns and col2 in df_copy.columns:
                # Tránh division by zero
                new_col = f"{col1}_ratio_{col2}"
                df_copy[new_col] = df_copy[col1] / (df_copy[col2] + 1e-8)
                new_features += 1
        
        print(f"   ✓ Tạo được {new_features} ratio features")
        print(f"   Shape sau: {df_copy.shape}")
        
        return df_copy
    
    def calculate_feature_importance_rf(self, X: pd.DataFrame, y: pd.Series,
                                       n_estimators: int = 100) -> Dict[str, float]:
        """
        Tính feature importance sử dụng Random Forest.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_estimators (int): Số trees trong forest
            
        Returns:
            dict: Dictionary với feature names và importance scores
        """
        print("\n" + "="*60)
        print("🌲 TÍNH FEATURE IMPORTANCE - RANDOM FOREST")
        print("="*60)
        
        # Kiểm tra nếu có missing values
        X_clean = X.fillna(X.mean())
        
        print(f"\n   Training Random Forest ({n_estimators} trees)...")
        
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                     random_state=42,
                                     n_jobs=-1)
        model.fit(X_clean, y)
        
        # Lấy importance
        importance_dict = dict(zip(X.columns, model.feature_importances_))
        
        # Sắp xếp theo độ quan trọng
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        self.feature_importance = importance_dict
        
        # In top features
        print(f"\n   Top 15 Features:")
        for i, (feature, importance) in enumerate(list(importance_dict.items())[:15], 1):
            bar = "█" * int(importance * 100)
            print(f"   {i:2d}. {feature:30s} {bar} {importance:.4f}")
        
        return importance_dict
    
    def calculate_feature_importance_univariate(self, X: pd.DataFrame, 
                                               y: pd.Series,
                                               method: str = 'f_regression') -> Dict[str, float]:
        """
        Tính feature importance sử dụng univariate statistical tests.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): 'f_regression' hoặc 'mutual_info'
            
        Returns:
            dict: Dictionary với feature names và scores
        """
        print("\n" + "="*60)
        print("📊 TÍNH FEATURE IMPORTANCE - UNIVARIATE")
        print("="*60)
        
        X_clean = X.fillna(X.mean())
        
        if method == 'f_regression':
            print(f"\n   Sử dụng F-test regression...")
            scores = f_regression(X_clean, y)[0]
        elif method == 'mutual_info':
            print(f"\n   Sử dụng Mutual Information...")
            scores = mutual_info_regression(X_clean, y, random_state=42)
        else:
            raise ValueError(f"Method {method} không hỗ trợ!")
        
        importance_dict = dict(zip(X.columns, scores))
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        # In top features
        print(f"\n   Top 15 Features:")
        for i, (feature, score) in enumerate(list(importance_dict.items())[:15], 1):
            bar = "█" * int(score / max(scores) * 50) if max(scores) > 0 else ""
            print(f"   {i:2d}. {feature:30s} {bar} {score:.4f}")
        
        return importance_dict
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'selectkbest',
                       n_features: int = 15,
                       importance_method: str = 'f_regression') -> List[str]:
        """
        Chọn top N features quan trọng nhất.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): 'selectkbest', 'importance_threshold'
            n_features (int): Số features cần chọn
            importance_method (str): Phương pháp tính importance
            
        Returns:
            list: Danh sách features được chọn
        """
        print("\n" + "="*60)
        print("🎯 CHỌN TOP FEATURES")
        print("="*60)
        
        X_clean = X.fillna(X.mean())
        
        if method == 'selectkbest':
            print(f"\n   Phương pháp: SelectKBest")
            print(f"   Số features: {n_features}")
            print(f"   Scoring: {importance_method}")
            
            if importance_method == 'f_regression':
                score_func = f_regression
            else:
                score_func = mutual_info_regression
            
            self.selector = SelectKBest(score_func=score_func, k=n_features)
            self.selector.fit(X_clean, y)
            
            # Lấy danh sách features được chọn
            selected_mask = self.selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
        
        elif method == 'importance_threshold':
            # Sử dụng importance từ Random Forest
            importance_dict = self.calculate_feature_importance_rf(X, y)
            
            # Lấy top N
            self.selected_features = list(importance_dict.keys())[:n_features]
        
        print(f"\n   ✓ Đã chọn {len(self.selected_features)} features:")
        for i, feat in enumerate(self.selected_features, 1):
            print(f"      {i:2d}. {feat}")
        
        return self.selected_features
    
    def get_selected_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Lấy DataFrame chỉ với features được chọn.
        
        Args:
            X (pd.DataFrame): DataFrame gốc
            
        Returns:
            pd.DataFrame: DataFrame chỉ với selected features
        """
        if not self.selected_features:
            raise ValueError("Chưa chọn features! Gọi select_features() trước.")
        
        return X[self.selected_features].copy()
    
    def get_importance_dataframe(self) -> pd.DataFrame:
        """
        Lấy DataFrame hiển thị feature importance.
        
        Returns:
            pd.DataFrame: DataFrame với feature names và importance scores
        """
        if not self.feature_importance:
            raise ValueError("Chưa tính feature importance!")
        
        return pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['Feature', 'Importance']
        )
    
    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt feature engineering.
        
        Returns:
            dict: Thông tin tóm tắt
        """
        return {
            'total_features_calculated': len(self.feature_importance),
            'features_selected': len(self.selected_features),
            'top_feature': list(self.feature_importance.keys())[0] if self.feature_importance else None,
            'top_feature_score': list(self.feature_importance.values())[0] if self.feature_importance else None
        }


def main():
    """Hàm chính để test module."""
    print("📚 Example: Feature Engineering Module\n")
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'Rainfall': np.random.rand(n_samples) * 200,
        'Temperature': np.random.rand(n_samples) * 40,
        'Humidity': np.random.rand(n_samples) * 100,
        'Pesticides': np.random.rand(n_samples) * 50
    })
    
    # Target variable (Yield) có dependency với features
    y = pd.Series(
        X['Rainfall'] * 0.5 + 
        X['Temperature'] * 2 + 
        X['Humidity'] * 0.3 + 
        np.random.randn(n_samples) * 10
    )
    
    print("📊 Sample Data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Khởi tạo builder
    builder = FeatureBuilder()
    
    # Tạo polynomial features
    X_poly = builder.create_polynomial_features(X, degree=2)
    
    # Tạo interaction features
    X_interaction = builder.create_interaction_features(X_poly)
    
    # Tính feature importance
    builder.calculate_feature_importance_rf(X_interaction, y)
    
    # Chọn top features
    builder.select_features(X_interaction, y, n_features=10)
    
    # Lấy selected data
    X_selected = builder.get_selected_dataframe(X_interaction)
    
    print(f"\n✓ Final shape: {X_selected.shape}")


if __name__ == "__main__":
    main()
