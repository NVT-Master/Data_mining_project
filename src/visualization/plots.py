"""
Visualization Module - Tạo các biểu đồ trực quan

Module này cung cấp các hàm vẽ:
- Biểu đồ phân bố Yield
- Heatmap tương quan
- Biểu đồ clusters
- Feature importance
- Residual plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PlotterLibrary:
    """
    Lớp chứa các hàm vẽ biểu đồ.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 6)):
        """
        Khởi tạo PlotterLibrary.
        
        Args:
            style (str): Style cho matplotlib
            figsize (tuple): Kích thước hình mặc định
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_yield_distribution(self, df: pd.DataFrame, 
                               target_column: str = 'Yield',
                               output_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ phân bố Yield.
        
        Args:
            df (pd.DataFrame): DataFrame input
            target_column (str): Tên cột target
            output_path (str): Đường dẫn lưu hình
        """
        print("📊 Vẽ biểu đồ phân bố Yield...")
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        axes[0].hist(df[target_column], bins=30, color=self.colors[0], edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(target_column, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution of {target_column}', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # KDE Plot
        df[target_column].plot(kind='kde', ax=axes[1], color=self.colors[1], linewidth=2)
        axes[1].set_xlabel(target_column, fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title(f'KDE Plot of {target_column}', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_boxplot(self, df: pd.DataFrame,
                    columns: Optional[List[str]] = None,
                    output_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ boxplot.
        
        Args:
            df (pd.DataFrame): DataFrame input
            columns (list): Danh sách cột (nếu None thì tất cả numeric)
            output_path (str): Đường dẫn lưu hình
        """
        print("📊 Vẽ biểu đồ Boxplot...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        df[columns].boxplot(ax=ax)
        ax.set_title('Boxplot of Features', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                output_path: Optional[str] = None) -> None:
        """
        Vẽ heatmap tương quan.
        
        Args:
            df (pd.DataFrame): DataFrame input
            output_path (str): Đường dẫn lưu hình
        """
        print("📊 Vẽ Correlation Heatmap...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Chỉ lấy numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Tính correlation
        corr = numeric_df.corr()
        
        # Vẽ heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_scatter_with_target(self, df: pd.DataFrame,
                                feature_x: str, feature_y: str,
                                target_column: str = 'Yield',
                                output_path: Optional[str] = None) -> None:
        """
        Vẽ scatter plot của 2 features với color coding từ target.
        
        Args:
            df (pd.DataFrame): DataFrame input
            feature_x (str): Feature cho trục X
            feature_y (str): Feature cho trục Y
            target_column (str): Cột để color coding
            output_path (str): Đường dẫn lưu hình
        """
        print(f"📊 Vẽ Scatter plot: {feature_x} vs {feature_y}...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(df[feature_x], df[feature_y], 
                            c=df[target_column], cmap='viridis',
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_title(f'{feature_x} vs {feature_y} (colored by {target_column})',
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(target_column, fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_clusters(self, X: pd.DataFrame, labels: np.ndarray,
                     feature_x: str = None, feature_y: str = None,
                     output_path: Optional[str] = None) -> None:
        """
        Vẽ clusters (2D scatter plot).
        
        Args:
            X (pd.DataFrame): Feature matrix
            labels (np.ndarray): Cluster labels
            feature_x (str): Feature cho trục X
            feature_y (str): Feature cho trục Y
            output_path (str): Đường dẫn lưu hình
        """
        print("📊 Vẽ biểu đồ Clusters...")
        
        if feature_x is None:
            feature_x = X.columns[0]
        if feature_y is None:
            feature_y = X.columns[1] if len(X.columns) > 1 else X.columns[0]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Vẽ từng cluster
        unique_labels = np.unique(labels)
        colors_map = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors_map):
            mask = labels == label
            if label == -1:
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
            
            ax.scatter(X.loc[mask, feature_x], X.loc[mask, feature_y],
                      c=[color], label=label_name, marker=marker,
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_title('Clustering Results', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               output_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ feature importance.
        
        Args:
            importance_df (pd.DataFrame): DataFrame với columns 'Feature' và 'Importance'
            top_n (int): Số features top cần hiển thị
            output_path (str): Đường dẫn lưu hình
        """
        print("📊 Vẽ biểu đồ Feature Importance...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        top_features = importance_df.head(top_n)
        
        ax.barh(top_features['Feature'], top_features['Importance'], 
               color=self.colors[0], edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')
        
        # Thêm giá trị lên các bar
        for i, v in enumerate(top_features['Importance']):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = 'Model',
                      output_path: Optional[str] = None) -> None:
        """
        Vẽ residual plots.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự báo
            model_name (str): Tên model
            output_path (str): Đường dẫn lưu hình
        """
        print(f"📊 Vẽ Residual Plots cho {model_name}...")
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Histogram of Residuals
        axes[0, 1].hist(residuals, bins=30, color=self.colors[0], edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)),
                          alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 1].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = 'Model',
                                output_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ Actual vs Predicted.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự báo
            model_name (str): Tên model
            output_path (str): Đường dẫn lưu hình
        """
        print(f"📊 Vẽ Actual vs Predicted cho {model_name}...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', 
                  linewidth=0.5, color=self.colors[0])
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Lưu: {output_path}")
        
        plt.show()


def main():
    """Hàm chính để test module."""
    print("📚 Example: Visualization Module\n")
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        'Feature1': np.random.randn(n_samples) * 10 + 50,
        'Feature2': np.random.randn(n_samples) * 15 + 70,
        'Feature3': np.random.randn(n_samples) * 20 + 100,
        'Yield': np.random.randn(n_samples) * 5 + 50
    })
    
    # Khởi tạo plotter
    plotter = PlotterLibrary()
    
    # Vẽ các biểu đồ
    # plotter.plot_yield_distribution(df, 'Yield')
    # plotter.plot_correlation_heatmap(df)
    # plotter.plot_scatter_with_target(df, 'Feature1', 'Feature2', 'Yield')
    
    # Clusters
    # X = df[['Feature1', 'Feature2']]
    # labels = np.random.randint(0, 3, n_samples)
    # plotter.plot_clusters(X, labels)


if __name__ == "__main__":
    main()
