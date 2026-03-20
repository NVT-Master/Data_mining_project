"""
Data Loader Module - Đọc và kiểm tra dữ liệu

Module này chịu trách nhiệm:
- Đọc các file CSV
- Kiểm tra schema dữ liệu
- Thống kê thông tin dữ liệu
- Kiểm tra missing values
- Kết hợp dữ liệu từ nhiều nguồn
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Lớp để tải và kiểm tra dữ liệu từ các file CSV.
    
    Attributes:
        data_dir (str): Đường dẫn thư mục chứa dữ liệu
        datasets (dict): Lưu các dataframe đã tải
        use_cleaned (bool): Sử dụng cleaned data nếu có
    """
    
    def __init__(self, data_dir: str = "./data/raw", use_cleaned: bool = True):
        """
        Khởi tạo DataLoader.
        
        Args:
            data_dir (str): Đường dẫn thư mục chứa dữ liệu
            use_cleaned (bool): Sử dụng cleaned data nếu có
        """
        self.data_dir = Path(data_dir)
        self.use_cleaned = use_cleaned
        self.datasets = {}

    def _resolve_csv_path(self, filename: str) -> Tuple[Path, str]:
        """
        Resolve path cho file CSV theo thứ tự ưu tiên cleaned/raw.

        Returns:
            tuple: (filepath, source_label)
        """
        normalized = filename.strip()
        stem = Path(normalized).stem
        cleaned_name = normalized if stem.endswith('_cleaned') else f"{stem}_cleaned.csv"

        candidates = []

        if self.use_cleaned:
            candidates.extend([
                (self.data_dir / 'cleaned' / cleaned_name, 'CLEANED'),
                (self.data_dir / 'cleaned' / normalized, 'CLEANED'),
                (self.data_dir / cleaned_name, 'CLEANED'),
                (self.data_dir / normalized, 'RAW'),
            ])
        else:
            candidates.extend([
                (self.data_dir / normalized, 'RAW'),
                (self.data_dir / cleaned_name, 'CLEANED'),
            ])

        for path, label in candidates:
            if path.exists():
                return path, label

        # Trả về candidate đầu để message lỗi rõ ràng
        return candidates[0][0], candidates[0][1]
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Đọc file CSV (ưu tiên cleaned data nếu có).
        
        Args:
            filename (str): Tên file CSV
            **kwargs: Các tham số khác để truyền cho pd.read_csv()
            
        Returns:
            pd.DataFrame: DataFrame đã đọc
        """
        filepath, source_label = self._resolve_csv_path(filename)
        print(f"📂 Đang đọc file ({source_label}): {filepath.name}...")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File không tồn tại: {filepath}")
        
        df = pd.read_csv(filepath, **kwargs)
        print(f"   ✓ Thành công! Shape: {df.shape}")
        
        return df
    
    def load_all_datasets(self, 
                         yield_file: str = "yield.csv",
                         pesticides_file: str = "pesticides.csv",
                         rainfall_file: str = "rainfall.csv",
                         temperature_file: str = "temp.csv") -> Dict[str, pd.DataFrame]:
        """
        Tải tất cả các file dữ liệu.
        
        Args:
            yield_file (str): Tên file năng suất cây trồng
            pesticides_file (str): Tên file dữ liệu thuốc trừ sâu
            rainfall_file (str): Tên file dữ liệu lượng mưa
            temperature_file (str): Tên file dữ liệu nhiệt độ
            
        Returns:
            dict: Dictionary chứa các DataFrame
        """
        print("\n" + "="*60)
        print("📊 BẮT ĐẦU TẢI DỮ LIỆU")
        print("="*60)
        
        self.datasets['yield'] = self.load_csv(yield_file)
        self.datasets['pesticides'] = self.load_csv(pesticides_file)
        self.datasets['rainfall'] = self.load_csv(rainfall_file)
        self.datasets['temperature'] = self.load_csv(temperature_file)
        
        print(f"\n✓ Đã tải {len(self.datasets)} file dữ liệu!")
        
        return self.datasets
    
    def get_dataset_info(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        """
        In thông tin chi tiết về dataset.
        
        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra
            name (str): Tên dataset để hiển thị
        """
        print(f"\n{'='*60}")
        print(f"📋 THÔNG TIN DATASET: {name}")
        print(f"{'='*60}")
        
        # Kích thước
        print(f"\n🔹 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
        
        # Tên cột
        print(f"\n🔹 Tên các cột:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col} ({df[col].dtype})")
        
        # Dòng đầu tiên
        print(f"\n🔹 5 dòng đầu tiên:")
        print(df.head())
        
    def check_missing_values(self, df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
        """
        Kiểm tra và trả về thông tin về missing values.
        
        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra
            name (str): Tên dataset để hiển thị
            
        Returns:
            pd.DataFrame: Bảng thống kê missing values
        """
        print(f"\n{'='*60}")
        print(f"⚠️  KIỂM TRA MISSING VALUES: {name}")
        print(f"{'='*60}")
        
        missing_info = pd.DataFrame({
            'Cột': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        missing_info = missing_info[missing_info['Missing Count'] > 0]
        
        if len(missing_info) == 0:
            print("\n✓ Không có missing values!")
        else:
            print(missing_info.to_string(index=False))
        
        return missing_info
    
    def get_statistical_summary(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        """
        In thống kê tổng quan về dataset.
        
        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra
            name (str): Tên dataset để hiển thị
        """
        print(f"\n{'='*60}")
        print(f"📊 THỐNG KÊ TỔNG QUAN: {name}")
        print(f"{'='*60}")
        print(df.describe().T)
        
    def check_data_types(self, df: pd.DataFrame) -> None:
        """
        Kiểm tra và in loại dữ liệu của các cột.
        
        Args:
            df (pd.DataFrame): DataFrame cần kiểm tra
        """
        print(f"\n{'='*60}")
        print("🔍 KIỂM TRA LOẠI DỮ LIỆU")
        print(f"{'='*60}")
        
        type_info = pd.DataFrame({
            'Cột': df.columns,
            'Loại': df.dtypes.values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        print(type_info.to_string(index=False))
    
    def merge_datasets(self, on: str = 'Year', how: str = 'inner') -> pd.DataFrame:
        """
        Kết hợp các dataset lại với nhau.
        
        Args:
            on (str): Cột để join
            how (str): Loại join ('inner', 'outer', 'left', 'right')
            
        Returns:
            pd.DataFrame: Dataset đã kết hợp
        """
        print(f"\n{'='*60}")
        print("🔗 KẾT HỢP CÁC DATASET")
        print(f"{'='*60}")
        
        if not self.datasets:
            raise ValueError("Chưa tải dữ liệu! Gọi load_all_datasets() trước.")
        
        # Chuẩn hóa tên cột
        yield_df = self.datasets['yield'].copy()
        pest_df = self.datasets['pesticides'].copy()
        rain_df = self.datasets['rainfall'].copy()
        temp_df = self.datasets['temperature'].copy()

        yield_df.columns = yield_df.columns.str.strip()
        pest_df.columns = pest_df.columns.str.strip()
        rain_df.columns = rain_df.columns.str.strip()
        temp_df.columns = temp_df.columns.str.strip()

        temp_df = temp_df.rename(columns={'year': 'Year', 'country': 'Area', 'avg_temp': 'Temperature'})

        rain_col_map = {
            'average_rain_fall_mm_per_year': 'Rainfall',
            'Average_Rainfall': 'Rainfall'
        }
        rain_df = rain_df.rename(columns=rain_col_map)

        yield_merge = yield_df[['Year', 'Area', 'Item', 'Value']].rename(columns={'Value': 'Yield'})
        pest_merge = pest_df[['Year', 'Area', 'Value']].rename(columns={'Value': 'Pesticides'})
        rain_merge = rain_df[['Year', 'Area', 'Rainfall']]
        temp_merge = temp_df[['Year', 'Area', 'Temperature']]

        # Tránh nở dữ liệu khi nguồn phụ có nhiều bản ghi cùng (Year, Area).
        pest_merge = pest_merge.groupby(['Year', 'Area'], as_index=False)['Pesticides'].mean()
        rain_merge = rain_merge.groupby(['Year', 'Area'], as_index=False)['Rainfall'].mean()
        temp_merge = temp_merge.groupby(['Year', 'Area'], as_index=False)['Temperature'].mean()

        print(f"   Bắt đầu: yield {yield_merge.shape}")

        merged_df = pd.merge(yield_merge, pest_merge, on=['Year', 'Area'], how=how)
        print(f"   Sau merge pesticides: {merged_df.shape}")

        merged_df = pd.merge(merged_df, rain_merge, on=['Year', 'Area'], how=how)
        print(f"   Sau merge rainfall: {merged_df.shape}")

        merged_df = pd.merge(merged_df, temp_merge, on=['Year', 'Area'], how=how)
        print(f"   Sau merge temperature: {merged_df.shape}")
        
        merged_df = merged_df.dropna(subset=['Yield', 'Pesticides', 'Rainfall', 'Temperature'])

        print(f"\n✓ Kết hợp xong! Shape cuối cùng: {merged_df.shape}")
        
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                          output_dir: str = "./data/processed") -> None:
        """
        Lưu DataFrame đã xử lý vào file CSV.
        
        Args:
            df (pd.DataFrame): DataFrame cần lưu
            filename (str): Tên file output
            output_dir (str): Đường dẫn thư mục output
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"💾 Đã lưu: {filepath}")
        
    def get_complete_dataset(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Lấy dataset hoàn chỉnh đã xử lý cơ bản.
        
        Returns:
            tuple: (merged_df, original_datasets)
        """
        if not self.datasets:
            self.load_all_datasets()
        
        merged_df = self.merge_datasets()
        
        return merged_df, self.datasets


def main():
    """
    Hàm chính để test module.
    """
    # Khởi tạo DataLoader
    loader = DataLoader(data_dir="./data/raw")
    
    # Tải tất cả các file
    datasets = loader.load_all_datasets()
    
    # Kiểm tra từng dataset
    for name, df in datasets.items():
        loader.get_dataset_info(df, name.upper())
        loader.check_missing_values(df, name.upper())
        loader.get_statistical_summary(df, name.upper())
        loader.check_data_types(df)
    
    # Kết hợp dataset
    merged_df = loader.merge_datasets()
    
    # Lưu dataset đã kết hợp
    loader.save_processed_data(merged_df, "merged_data.csv")


if __name__ == "__main__":
    main()
