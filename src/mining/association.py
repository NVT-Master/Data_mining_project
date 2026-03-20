"""
Association Rule Mining Module - Tìm các quy tắc liên kết

Module này thực hiện:
- Rời rạc hóa dữ liệu continuous thành categorical
- Áp dụng Apriori algorithm
- Tính support, confidence, lift
- Tìm các quy tắc liên quan đến Yield cao
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AssociationRuleMining:
    """
    Lớp để thực hiện association rule mining.
    
    Attributes:
        frequent_itemsets (DataFrame): Các itemsets thường xuyên
        rules (DataFrame): Các quy tắc được tìm thấy
    """
    
    def __init__(self):
        """Khởi tạo AssociationRuleMining."""
        self.frequent_itemsets = None
        self.rules = None
        self.discretizers = {}
        
    def discretize_features(self, df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           n_bins: int = 5,
                           strategy: str = 'quantile') -> pd.DataFrame:
        """
        Rời rạc hóa các features continuous thành categorical.
        
        Args:
            df (pd.DataFrame): DataFrame input
            columns (list): Danh sách cột cần rời rạc hóa
            n_bins (int): Số bins
            strategy (str): 'quantile', 'uniform', 'kmeans'
            
        Returns:
            pd.DataFrame: DataFrame đã rời rạc hóa
        """
        print("\n" + "="*60)
        print("📊 RỜI RẠC HÓA DỮ LIỆU")
        print("="*60)
        
        df_copy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n   Số features: {len(columns)}")
        print(f"   Số bins: {n_bins}")
        print(f"   Strategy: {strategy}")
        
        for col in columns:
            if col in df_copy.columns:
                # Tạo bins
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, 
                    encode='ordinal', 
                    strategy=strategy
                )
                
                # Fit và transform
                df_copy[col] = discretizer.fit_transform(
                    df_copy[[col]]
                ).astype(int)
                
                self.discretizers[col] = discretizer
                print(f"   ✓ {col}: discretized thành {n_bins} bins")
        
        return df_copy
    
    def create_itemsets(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Tạo itemsets từ dữ liệu rời rạc.
        
        Args:
            df (pd.DataFrame): DataFrame đã rời rạc hóa
            columns (list): Danh sách cột để tạo itemsets
            
        Returns:
            pd.DataFrame: Binary matrix (transactions x items)
        """
        print("\n" + "="*60)
        print("🛒 TẠO ITEMSETS")
        print("="*60)
        
        if columns is None:
            columns = df.columns.tolist()
        
        print(f"\n   Số features: {len(columns)}")
        
        # Tạo binary matrix
        itemset_matrix = pd.DataFrame()
        
        for col in columns:
            unique_vals = sorted(df[col].unique())
            
            for val in unique_vals:
                item_name = f"{col}={val}"
                itemset_matrix[item_name] = (df[col] == val).astype(int)
        
        print(f"   Items được tạo: {itemset_matrix.shape[1]}")
        print(f"   Transactions: {itemset_matrix.shape[0]}")
        print(f"   Matrix shape: {itemset_matrix.shape}")
        
        return itemset_matrix
    
    def find_frequent_itemsets(self, itemset_matrix: pd.DataFrame,
                              min_support: float = 0.3) -> pd.DataFrame:
        """
        Tìm các itemsets thường xuyên sử dụng Apriori algorithm.
        
        Args:
            itemset_matrix (pd.DataFrame): Binary matrix của itemsets
            min_support (float): Ngưỡng support tối thiểu
            
        Returns:
            pd.DataFrame: Frequent itemsets
        """
        print("\n" + "="*60)
        print("🔍 APRIORI ALGORITHM - TÌM ITEMSETS THƯỜNG XUYÊN")
        print("="*60)
        
        print(f"\n   Min support: {min_support}")
        print(f"   Đang chạy Apriori...")
        
        self.frequent_itemsets = apriori(
            itemset_matrix,
            min_support=min_support,
            use_colnames=True
        )
        
        print(f"\n   ✓ Tìm được {len(self.frequent_itemsets)} frequent itemsets")
        
        # In top itemsets
        print(f"\n   Top 15 Frequent Itemsets:")
        self.frequent_itemsets_sorted = self.frequent_itemsets.sort_values(
            'support', ascending=False
        )
        
        for idx, (_, row) in enumerate(self.frequent_itemsets_sorted.head(15).iterrows(), 1):
            itemset = list(row['itemsets'])
            support = row['support']
            print(f"   {idx:2d}. {itemset} → support: {support:.4f}")
        
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence: float = 0.5,
                      min_lift: float = 1.0) -> pd.DataFrame:
        """
        Tạo association rules từ frequent itemsets.
        
        Args:
            min_confidence (float): Ngưỡng confidence tối thiểu
            min_lift (float): Ngưỡng lift tối thiểu
            
        Returns:
            pd.DataFrame: Association rules
        """
        print("\n" + "="*60)
        print("🔗 TẠO ASSOCIATION RULES")
        print("="*60)
        
        if self.frequent_itemsets is None:
            raise ValueError("Chưa tìm frequent itemsets! Gọi find_frequent_itemsets() trước.")
        
        print(f"\n   Min confidence: {min_confidence}")
        print(f"   Min lift: {min_lift}")
        print(f"   Đang tạo rules...")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # Thêm lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        print(f"\n   ✓ Tìm được {len(self.rules)} rules")
        
        # Tính support, confidence, lift
        print(f"\n   Metrics Statistics:")
        print(f"   • Support:    min={self.rules['support'].min():.4f}, max={self.rules['support'].max():.4f}")
        print(f"   • Confidence: min={self.rules['confidence'].min():.4f}, max={self.rules['confidence'].max():.4f}")
        print(f"   • Lift:       min={self.rules['lift'].min():.4f}, max={self.rules['lift'].max():.4f}")
        
        return self.rules
    
    def find_rules_for_high_yield(self, rules: pd.DataFrame,
                                 yield_value: str = "Yield=4",
                                 top_n: int = 10) -> pd.DataFrame:
        """
        Tìm các quy tắc liên quan đến Yield cao.
        
        Args:
            rules (pd.DataFrame): Dataframe chứa rules
            yield_value (str): Giá trị yield cần tìm (e.g., "Yield=4")
            top_n (int): Số rules top cần trả về
            
        Returns:
            pd.DataFrame: Rules liên quan đến high yield
        """
        print("\n" + "="*60)
        print("🎯 TÌM RULES LIÊN QUAN ĐẾN YIELD CAO")
        print("="*60)
        
        print(f"\n   Tìm rules có consequent: {yield_value}")
        
        # Tìm rules có Yield cao trong consequent
        high_yield_rules = rules[
            rules['consequents'].apply(lambda x: yield_value in str(x))
        ].copy()
        
        if len(high_yield_rules) == 0:
            print(f"   ⚠️  Không tìm thấy rules cho {yield_value}")
            return pd.DataFrame()
        
        # Sắp xếp theo lift/confidence
        high_yield_rules = high_yield_rules.sort_values('lift', ascending=False)
        
        print(f"\n   ✓ Tìm được {len(high_yield_rules)} rules")
        print(f"\n   Top {min(top_n, len(high_yield_rules))} Rules:")
        
        for idx, (_, row) in enumerate(high_yield_rules.head(top_n).iterrows(), 1):
            antecedent = list(row['antecedents'])
            consequent = list(row['consequents'])
            support = row['support']
            confidence = row['confidence']
            lift = row['lift']
            
            print(f"\n   Rule {idx}:")
            print(f"      Antecedent (điều kiện): {antecedent}")
            print(f"      Consequent (kết quả):  {consequent}")
            print(f"      Support:   {support:.4f}")
            print(f"      Confidence: {confidence:.4f}")
            print(f"      Lift:      {lift:.4f}")
        
        return high_yield_rules.head(top_n)
    
    def get_rules_summary(self) -> pd.DataFrame:
        """
        Lấy tóm tắt các rules.
        
        Returns:
            pd.DataFrame: Bảng tóm tắt rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'Antecedent': self.rules['antecedents'].apply(lambda x: ', '.join(list(x))),
            'Consequent': self.rules['consequents'].apply(lambda x: ', '.join(list(x))),
            'Support': self.rules['support'].round(4),
            'Confidence': self.rules['confidence'].round(4),
            'Lift': self.rules['lift'].round(4)
        })
        
        return summary.sort_values('lift', ascending=False)
    
    def export_rules_to_csv(self, filepath: str) -> None:
        """
        Xuất rules ra file CSV.
        
        Args:
            filepath (str): Đường dẫn file output
        """
        summary = self.get_rules_summary()
        
        if len(summary) > 0:
            summary.to_csv(filepath, index=False)
            print(f"💾 Đã xuất rules: {filepath}")
        else:
            print("⚠️  Không có rules để xuất!")


def main():
    """Hàm chính để test module."""
    print("📚 Example: Association Rule Mining\n")
    
    # Tạo sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'Rainfall': np.random.rand(100) * 200,
        'Temperature': np.random.rand(100) * 40,
        'Humidity': np.random.rand(100) * 100,
        'Yield': np.random.rand(100) * 50
    })
    
    print("📊 Sample Data:")
    print(f"Shape: {df.shape}\n")
    
    # Khởi tạo mining
    mining = AssociationRuleMining()
    
    # Rời rạc hóa
    df_disc = mining.discretize_features(df, n_bins=4)
    
    # Tạo itemsets
    itemset_matrix = mining.create_itemsets(df_disc)
    
    # Tìm frequent itemsets
    mining.find_frequent_itemsets(itemset_matrix, min_support=0.2)
    
    # Tạo rules
    mining.generate_rules(min_confidence=0.3, min_lift=1.0)
    
    # Xuất rules
    print(mining.get_rules_summary())


if __name__ == "__main__":
    main()
