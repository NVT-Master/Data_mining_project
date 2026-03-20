"""
Data Cleaner Module - Xử lý dữ liệu

Module này chịu trách nhiệm:
- Xử lý missing values
- Xử lý outliers
- Encoding dữ liệu categorical
- Scaling dữ liệu numeric
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Lớp để xử lý và làm sạch dữ liệu.
    
    Attributes:
        scaler (object): Object để scaling dữ liệu
        encoders (dict): Dictionary lưu các encoder
        imputers (dict): Dictionary lưu các imputer
    """
    
    def __init__(self):
        """Khởi tạo DataCleaner."""
        self.scaler = None
        self.encoders = {}
        self.imputers = {}
        self.numeric_features = []
        self.categorical_features = []
        
    def identify_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Xác định features numeric và categorical.
        
        Args:
            df (pd.DataFrame): DataFrame input
            
        Returns:
            dict: Dictionary chứa danh sách numeric và categorical features
        """
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        print("\n" + "="*60)
        print("🔍 XÁC ĐỊNH LOẠI FEATURES")
        print("="*60)
        print(f"\n📊 Numeric features ({len(self.numeric_features)}):")
        for feat in self.numeric_features:
            print(f"   • {feat}")
        
        print(f"\n📝 Categorical features ({len(self.categorical_features)}):")
        for feat in self.categorical_features:
            print(f"   • {feat}")
        
        return {
            'numeric': self.numeric_features,
            'categorical': self.categorical_features
        }
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             method: str = 'mean',
                             numeric_method: str = 'mean',
                             categorical_method: str = 'most_frequent') -> pd.DataFrame:
        """
        Xử lý missing values.
        
        Args:
            df (pd.DataFrame): DataFrame input
            method (str): Phương pháp: 'mean', 'median', 'drop', 'forward_fill'
            numeric_method (str): Phương pháp cho numeric
            categorical_method (str): Phương pháp cho categorical
            
        Returns:
            pd.DataFrame: DataFrame đã xử lý
        """
        print("\n" + "="*60)
        print("🧹 XỬ LÝ MISSING VALUES")
        print("="*60)
        
        df_copy = df.copy()
        
        missing_before = df_copy.isnull().sum().sum()
        print(f"\n   Missing values trước: {missing_before}")
        
        if method == 'drop':
            df_copy = df_copy.dropna()
            print(f"   Đã xóa các dòng có missing values")
            print(f"   Shape sau: {df_copy.shape}")
        
        else:
            # Xử lý numeric features
            if len(self.numeric_features) > 0:
                numeric_imputer = SimpleImputer(strategy=numeric_method)
                df_copy[self.numeric_features] = numeric_imputer.fit_transform(
                    df_copy[self.numeric_features]
                )
                print(f"   ✓ Xử lý numeric ({numeric_method}): {len(self.numeric_features)} features")
            
            # Xử lý categorical features
            if len(self.categorical_features) > 0:
                categorical_imputer = SimpleImputer(strategy=categorical_method)
                df_copy[self.categorical_features] = categorical_imputer.fit_transform(
                    df_copy[self.categorical_features]
                )
                print(f"   ✓ Xử lý categorical ({categorical_method}): {len(self.categorical_features)} features")
            
            # Forward fill nếu cần
            if method == 'forward_fill':
                df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
                print(f"   ✓ Sử dụng forward fill")
        
        missing_after = df_copy.isnull().sum().sum()
        print(f"   Missing values sau: {missing_after}")
        print(f"   ✓ Hoàn thành!")
        
        return df_copy
    
    def handle_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr',
                       zscore_threshold: float = 3,
                       iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Xử lý outliers.
        
        Args:
            df (pd.DataFrame): DataFrame input
            method (str): 'iqr', 'zscore', 'isolation_forest'
            zscore_threshold (float): Ngưỡng cho Z-score
            iqr_multiplier (float): Hệ số nhân cho IQR
            
        Returns:
            pd.DataFrame: DataFrame đã xử lý
        """
        print("\n" + "="*60)
        print("🎯 XỬ LÝ OUTLIERS")
        print("="*60)
        
        df_copy = df.copy()
        
        if method == 'iqr':
            print(f"\n   Sử dụng phương pháp IQR (hệ số: {iqr_multiplier})")
            
            for col in self.numeric_features:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
                
                # Clip outliers thay vì xóa
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
                
                if outliers > 0:
                    print(f"   • {col}: {outliers} outliers → clipped")
        
        elif method == 'zscore':
            print(f"\n   Sử dụng phương pháp Z-score (threshold: {zscore_threshold})")
            
            for col in self.numeric_features:
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                outliers = (z_scores > zscore_threshold).sum()
                
                # Clip based on z-score
                lower_bound = df_copy[col].mean() - zscore_threshold * df_copy[col].std()
                upper_bound = df_copy[col].mean() + zscore_threshold * df_copy[col].std()
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
                
                if outliers > 0:
                    print(f"   • {col}: {outliers} outliers → clipped")
        
        print(f"   ✓ Hoàn thành!")
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame,
                          method: str = 'onehot',
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encoding dữ liệu categorical.
        
        Args:
            df (pd.DataFrame): DataFrame input
            method (str): 'label' hoặc 'onehot'
            columns (list): Danh sách cột cần encoding (nếu None thì tất cả categorical)
            
        Returns:
            pd.DataFrame: DataFrame đã encoding
        """
        print("\n" + "="*60)
        print("🔤 ENCODING CATEGORICAL FEATURES")
        print("="*60)
        
        df_copy = df.copy()
        
        if columns is None:
            columns = self.categorical_features
        
        print(f"\n   Phương pháp: {method.upper()}")
        print(f"   Số cột: {len(columns)}")
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self.encoders[col] = le
                print(f"   • {col}: {len(le.classes_)} classes")
        
        elif method == 'onehot':
            df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
            print(f"   • Sử dụng One-Hot Encoding")
            print(f"   • New shape: {df_copy.shape}")
        
        print(f"   ✓ Hoàn thành!")
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame,
                      method: str = 'standard',
                      columns: Optional[List[str]] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scaling các numeric features.
        
        Args:
            df (pd.DataFrame): DataFrame input
            method (str): 'standard', 'minmax', 'robust'
            columns (list): Danh sách cột cần scaling (nếu None thì tất cả numeric)
            fit (bool): Có fit scaler hay không
            
        Returns:
            pd.DataFrame: DataFrame đã scaling
        """
        print("\n" + "="*60)
        print("📊 SCALING NUMERIC FEATURES")
        print("="*60)
        
        df_copy = df.copy()
        
        if columns is None:
            columns = self.numeric_features
        
        # Chọn scaler
        if method == 'standard':
            self.scaler = StandardScaler()
            print(f"\n   Phương pháp: StandardScaler (mean=0, std=1)")
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            print(f"\n   Phương pháp: MinMaxScaler (range 0-1)")
        elif method == 'robust':
            self.scaler = RobustScaler()
            print(f"\n   Phương pháp: RobustScaler (robust to outliers)")
        
        print(f"   Số features: {len(columns)}")
        
        # Fit và transform
        if fit:
            df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        else:
            df_copy[columns] = self.scaler.transform(df_copy[columns])
        
        print(f"   ✓ Hoàn thành!")
        
        return df_copy
    
    def complete_pipeline(self, df: pd.DataFrame,
                         handle_missing: bool = True,
                         handle_outliers: bool = True,
                         encode: bool = True,
                         scale: bool = True,
                         target_column: str = None) -> pd.DataFrame:
        """
        Chạy toàn bộ quy trình xử lý dữ liệu.
        
        Args:
            df (pd.DataFrame): DataFrame input
            handle_missing (bool): Có xử lý missing không
            handle_outliers (bool): Có xử lý outliers không
            encode (bool): Có encoding không
            scale (bool): Có scaling không
            target_column (str): Tên cột target (không xử lý)
            
        Returns:
            pd.DataFrame: DataFrame đã xử lý đầy đủ
        """
        print("\n" + "="*80)
        print("🚀 PIPELINE XỬ LÝ DỮ LIỆU HOÀN CHỈNH")
        print("="*80)
        
        df_processed = df.copy()
        
        # Xác định features
        self.identify_features(df_processed)
        
        # Xử lý missing values
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
        
        # Xử lý outliers (chỉ numeric)
        if handle_outliers:
            df_processed = self.handle_outliers(df_processed)
        
        # Encoding categorical
        if encode:
            df_processed = self.encode_categorical(df_processed)
        
        # Scaling numeric
        if scale:
            # Cập nhật numeric features sau encoding
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Loại bỏ target column nếu có
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            df_processed = self.scale_features(df_processed, columns=numeric_cols)
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH PIPELINE XỬ LÝ DỮ LIỆU")
        print("="*80)
        print(f"\nShape input: {df.shape}")
        print(f"Shape output: {df_processed.shape}")
        
        return df_processed
    
    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt xử lý dữ liệu.
        
        Returns:
            dict: Thông tin tóm tắt
        """
        return {
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features),
            'encoders_saved': len(self.encoders),
            'scaler_fitted': self.scaler is not None
        }


def main():
    """Hàm chính để test module."""
    # Ví dụ sử dụng
    print("📚 Example: Data Cleaner Module")
    
    # Tạo sample data
    data = {
        'Year': [2019, 2019, 2020, 2020, 2021],
        'State': ['Punjab', 'Haryana', 'Punjab', 'UP', 'Haryana'],
        'Crop': ['Rice', 'Wheat', 'Cotton', 'Rice', 'Wheat'],
        'Production': [5.2, 4.8, None, 6.1, 5.5],
        'Yield': [45.2, 38.5, 42.1, 46.8, 39.2]
    }
    df = pd.DataFrame(data)
    
    print("\n📊 Sample Data:")
    print(df)
    
    # Khởi tạo cleaner
    cleaner = DataCleaner()
    
    # Chạy pipeline
    df_cleaned = cleaner.complete_pipeline(
        df,
        handle_missing=True,
        handle_outliers=False,
        encode=True,
        scale=True,
        target_column='Yield'
    )
    
    print("\n🔄 Data sau xử lý:")
    print(df_cleaned)


if __name__ == "__main__":
    main()
