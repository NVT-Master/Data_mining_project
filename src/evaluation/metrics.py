"""
Evaluation Module - Đánh giá chi tiết kết quả mô hình

Module này bao gồm:
- Tính toán các metrics (MAE, RMSE, R², MAPE)
- Phân tích residuals
- Tạo báo cáo chi tiết
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                           r2_score, mean_absolute_percentage_error)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelMetrics:
    """
    Lớp để tính toán và lưu các metrics đánh giá.
    
    Attributes:
        metrics (dict): Dictionary lưu metrics cho mỗi model
    """
    
    def __init__(self):
        """Khởi tạo ModelMetrics."""
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = 'Model') -> Dict[str, float]:
        """
        Tính tất cả các metrics.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự báo
            model_name (str): Tên model
            
        Returns:
            dict: Dictionary chứa các metrics
        """
        # Tính MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # Tính RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Tính R²
        r2 = r2_score(y_true, y_pred)
        
        # Tính MAPE
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Tính Mean Absolute Scaled Error
        naive_forecast_mase = np.mean(np.abs(np.diff(y_true)))
        mase = mae / (naive_forecast_mase + 1e-8)
        
        # Tính Mean Bias Error
        mbe = np.mean(y_true - y_pred)
        
        # Tính Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        metrics_dict = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2,
            'MAPE': mape,
            'MASE': mase,
            'MBE': mbe,
            'Median_AE': median_ae
        }
        
        self.metrics[model_name] = metrics_dict
        
        return metrics_dict
    
    def print_metrics(self, model_name: str) -> None:
        """
        In metrics chi tiết cho một model.
        
        Args:
            model_name (str): Tên model
        """
        if model_name not in self.metrics:
            print(f"⚠️  Không tìm thấy metrics cho {model_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 METRICS: {model_name}")
        print(f"{'='*60}")
        
        metrics = self.metrics[model_name]
        
        print(f"\n   🔹 Regression Metrics:")
        print(f"      • MAE (Mean Absolute Error):     {metrics['MAE']:.4f}")
        print(f"      • RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
        print(f"      • MSE (Mean Squared Error):       {metrics['MSE']:.4f}")
        print(f"      • R² (Coefficient of Determination): {metrics['R2']:.4f}")
        
        print(f"\n   🔹 Percentage Errors:")
        print(f"      • MAPE (Mean Absolute % Error): {metrics['MAPE']:.4f}")
        
        print(f"\n   🔹 Other Metrics:")
        print(f"      • MASE (Mean Absolute Scaled Error): {metrics['MASE']:.4f}")
        print(f"      • MBE (Mean Bias Error):              {metrics['MBE']:.4f}")
        print(f"      • Median AE (Median Absolute Error): {metrics['Median_AE']:.4f}")
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Lấy DataFrame chứa tất cả metrics.
        
        Returns:
            pd.DataFrame: Bảng metrics
        """
        return pd.DataFrame(list(self.metrics.values()))


class ResidualAnalysis:
    """
    Lớp để phân tích residuals (phần dư).
    """
    
    def __init__(self):
        """Khởi tạo ResidualAnalysis."""
        self.residuals = {}
        
    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model') -> np.ndarray:
        """
        Tính residuals.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự báo
            model_name (str): Tên model
            
        Returns:
            np.ndarray: Residuals
        """
        residuals = y_true - y_pred
        self.residuals[model_name] = residuals
        
        return residuals
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = 'Model') -> Dict:
        """
        Phân tích chi tiết residuals.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự báo
            model_name (str): Tên model
            
        Returns:
            dict: Thông tin phân tích residuals
        """
        residuals = self.calculate_residuals(y_true, y_pred, model_name)
        
        print(f"\n{'='*60}")
        print(f"📈 PHÂN TÍCH RESIDUALS: {model_name}")
        print(f"{'='*60}")
        
        print(f"\n   🔹 Thống kê Residuals:")
        print(f"      • Mean:       {np.mean(residuals):.4f}")
        print(f"      • Std Dev:    {np.std(residuals):.4f}")
        print(f"      • Min:        {np.min(residuals):.4f}")
        print(f"      • Max:        {np.max(residuals):.4f}")
        print(f"      • Median:     {np.median(residuals):.4f}")
        
        # Kiểm tra normality
        print(f"\n   🔹 Kiểm tra Tính Chuẩn:")
        skewness = (np.mean(residuals) - np.median(residuals)) / (np.std(residuals) + 1e-8)
        print(f"      • Skewness:   {skewness:.4f} {'(roughly normal)' if abs(skewness) < 0.5 else '(skewed)'}")
        
        # Kiểm tra heteroscedasticity
        first_half = np.std(residuals[:len(residuals)//2])
        second_half = np.std(residuals[len(residuals)//2:])
        print(f"\n   🔹 Kiểm tra Homoscedasticity:")
        print(f"      • Std Dev First Half:  {first_half:.4f}")
        print(f"      • Std Dev Second Half: {second_half:.4f}")
        print(f"      • Ratio: {max(first_half, second_half) / (min(first_half, second_half) + 1e-8):.4f}")
        
        analysis_dict = {
            'model': model_name,
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'skewness': skewness
        }
        
        return analysis_dict


class EvaluationReport:
    """
    Lớp để tạo báo cáo đánh giá chi tiết.
    """
    
    def __init__(self):
        """Khởi tạo EvaluationReport."""
        self.metrics_calculator = ModelMetrics()
        self.residual_analyzer = ResidualAnalysis()
        
    def create_comparison_table(self, results_dict: Dict) -> pd.DataFrame:
        """
        Tạo bảng so sánh các models.
        
        Args:
            results_dict (dict): Dictionary chứa kết quả từng model
                                {model_name: {'y_true': ..., 'y_pred': ...}, ...}
            
        Returns:
            pd.DataFrame: Bảng so sánh
        """
        print("\n" + "="*80)
        print("📊 TẠO BẢNG SO SÁNH CÁC MODELS")
        print("="*80)
        
        comparison_rows = []
        
        for model_name, result in results_dict.items():
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, model_name)
            comparison_rows.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        return comparison_df.sort_values('R2', ascending=False)
    
    def generate_report(self, y_true: Dict, y_pred: Dict,
                       output_file: Optional[str] = None) -> str:
        """
        Tạo báo cáo text chi tiết.
        
        Args:
            y_true (dict): {model_name: y_true_array}
            y_pred (dict): {model_name: y_pred_array}
            output_file (str): Đường dẫn file output
            
        Returns:
            str: Nội dung báo cáo
        """
        print("\n" + "="*80)
        print("📄 TẠO BÁO CÁO CHI TIẾT")
        print("="*80)
        
        report = "="*80 + "\n"
        report += "BÁO CÁO ĐÁNH GIÁ MÔ HÌNH REGRESSION\n"
        report += "="*80 + "\n\n"
        
        for model_name in y_true.keys():
            if model_name not in y_pred:
                continue
            
            yt = y_true[model_name]
            yp = y_pred[model_name]
            
            # Metrics
            metrics = self.metrics_calculator.calculate_metrics(yt, yp, model_name)
            
            report += f"Model: {model_name}\n"
            report += "-" * 40 + "\n"
            report += f"MAE:  {metrics['MAE']:.6f}\n"
            report += f"RMSE: {metrics['RMSE']:.6f}\n"
            report += f"R²:   {metrics['R2']:.6f}\n"
            report += f"MAPE: {metrics['MAPE']:.6f}\n"
            report += "\n"
            
            # Residuals
            residuals = self.residual_analyzer.calculate_residuals(yt, yp, model_name)
            report += f"Residuals Mean: {np.mean(residuals):.6f}\n"
            report += f"Residuals Std:  {np.std(residuals):.6f}\n"
            report += "\n\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"💾 Đã lưu báo cáo: {output_file}")
        
        return report
    
    def get_best_model_recommendation(self, comparison_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Đưa ra khuyến nghị về model tốt nhất.
        
        Args:
            comparison_df (pd.DataFrame): Bảng so sánh models
            
        Returns:
            tuple: (best_model_name, best_r2_score)
        """
        print("\n" + "="*60)
        print("💡 KHUYẾN NGHỊ MÔ HÌNH TỐT NHẤT")
        print("="*60)
        
        best_idx = comparison_df['R2'].idxmax()
        best_model = comparison_df.loc[best_idx]
        
        print(f"\n   🏆 Model được khuyến nghị: {best_model['Model']}")
        print(f"      • R²:   {best_model['R2']:.4f}")
        print(f"      • RMSE: {best_model['RMSE']:.4f}")
        print(f"      • MAE:  {best_model['MAE']:.4f}")
        
        return best_model['Model'], best_model['R2']


def main():
    """Hàm chính để test module."""
    print("📚 Example: Evaluation Module\n")
    
    # Tạo sample predictions
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.linspace(10, 100, n_samples) + np.random.randn(n_samples) * 5
    
    # Predictions từ các models
    results = {
        'Linear Regression': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 3
        },
        'Random Forest': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 2
        },
        'XGBoost': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 1.5
        }
    }
    
    # Tạo báo cáo
    report = EvaluationReport()
    
    # Bảng so sánh
    comparison = report.create_comparison_table(results)
    print("\n📊 Bảng So Sánh:")
    print(comparison)
    
    # Phân tích residuals
    analyzer = ResidualAnalysis()
    for model_name, result in results.items():
        analyzer.analyze_residuals(result['y_true'], result['y_pred'], model_name)
    
    # Khuyến nghị
    report.get_best_model_recommendation(comparison)


if __name__ == "__main__":
    main()
