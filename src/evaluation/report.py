"""
Report Module - Tạo báo cáo chi tiết và xuất file

Module này chịu trách nhiệm:
- Xuất kết quả ra file CSV, Excel
- Tạo báo cáo markdown
- Lưu kết quả visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EvaluationReporter:
    """
    Lớp để tạo và xuất báo cáo.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Khởi tạo EvaluationReporter.
        
        Args:
            output_dir (str): Thư mục lưu báo cáo
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_results_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Xuất kết quả ra file CSV.
        
        Args:
            df (pd.DataFrame): DataFrame cần xuất
            filename (str): Tên file
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Đã xuất: {filepath}")
    
    def export_results_excel(self, data_dict: Dict[str, pd.DataFrame],
                            filename: str) -> None:
        """
        Xuất kết quả ra file Excel với nhiều sheets.
        
        Args:
            data_dict (dict): {sheet_name: DataFrame}
            filename (str): Tên file
        """
        filepath = self.output_dir / filename

        engines = ['openpyxl', 'xlsxwriter']
        last_error = None

        for engine in engines:
            try:
                with pd.ExcelWriter(filepath, engine=engine) as writer:
                    for sheet_name, df in data_dict.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                print(f"✓ Đã xuất Excel ({engine}): {filepath}")
                return
            except ModuleNotFoundError as exc:
                last_error = exc

        # Fallback: nếu chưa cài engine Excel thì lưu từng sheet thành CSV để không làm dừng pipeline.
        fallback_dir = filepath.with_suffix('')
        fallback_dir.mkdir(parents=True, exist_ok=True)

        for sheet_name, df in data_dict.items():
            safe_name = str(sheet_name).replace(' ', '_')
            csv_path = fallback_dir / f"{safe_name}.csv"
            df.to_csv(csv_path, index=False)

        print("⚠️  Không tìm thấy openpyxl/xlsxwriter, đã fallback sang CSV.")
        print(f"   Thư mục fallback: {fallback_dir}")
        if last_error is not None:
            print(f"   Chi tiết: {last_error}")
    
    def create_markdown_report(self, title: str, sections: Dict[str, str],
                              filename: str = "report.md") -> str:
        """
        Tạo báo cáo markdown.
        
        Args:
            title (str): Tiêu đề báo cáo
            sections (dict): {section_name: content}
            filename (str): Tên file
            
        Returns:
            str: Nội dung báo cáo
        """
        report = f"# {title}\n\n"
        report += f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for section_name, content in sections.items():
            report += f"## {section_name}\n\n"
            report += f"{content}\n\n"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Đã tạo báo cáo: {filepath}")
        
        return report
    
    def create_model_comparison_report(self, 
                                      comparison_df: pd.DataFrame,
                                      cv_results_df: Optional[pd.DataFrame] = None,
                                      best_model: Optional[str] = None) -> str:
        """
        Tạo báo cáo so sánh models.
        
        Args:
            comparison_df (pd.DataFrame): Bảng kết quả test set
            cv_results_df (pd.DataFrame): Bảng kết quả cross-validation
            best_model (str): Tên model tốt nhất
            
        Returns:
            str: Nội dung báo cáo
        """
        print("\n" + "="*80)
        print("📄 TẠO BÁO CÁO SO SÁNH CÁC MODELS")
        print("="*80)
        
        report = "# Báo Cáo So Sánh Các Mô Hình Regression\n\n"
        report += f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Test Set Results
        report += "## Kết Quả Test Set\n\n"
        report += "| Model | MAE | RMSE | R² | MAPE |\n"
        report += "|-------|-----|------|-----|------|\n"
        
        for _, row in comparison_df.iterrows():
            report += f"| {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.4f} |\n"
        
        report += "\n"
        
        # Cross-validation Results
        if cv_results_df is not None and len(cv_results_df) > 0:
            report += "## Kết Quả Cross-Validation (5-Fold)\n\n"
            report += "| Model | R² Mean | R² Std | MAE Mean | MAE Std | RMSE Mean | RMSE Std |\n"
            report += "|-------|---------|--------|----------|---------|-----------|----------|\n"
            
            for _, row in cv_results_df.iterrows():
                report += f"| {row['Model']} | {row['R2_Mean']:.4f} | {row['R2_Std']:.4f} | "
                report += f"{row['MAE_Mean']:.4f} | {row['MAE_Std']:.4f} | "
                report += f"{row['RMSE_Mean']:.4f} | {row['RMSE_Std']:.4f} |\n"
            
            report += "\n"
        
        # Best Model
        if best_model:
            report += f"## ⭐ Mô Hình Được Khuyến Nghị\n\n"
            report += f"**{best_model}**\n\n"
        
        # Interpretation
        report += "## 💡 Giải Thích Kết Quả\n\n"
        report += "### Các Metrics Sử Dụng:\n"
        report += "- **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình\n"
        report += "- **RMSE (Root Mean Squared Error):** Căn bậc 2 của sai số bình phương trung bình\n"
        report += "- **R² (Hệ số xác định):** Tỷ lệ phương sai được giải thích (0-1, cao hơn tốt hơn)\n"
        report += "- **MAPE (Mean Absolute Percentage Error):** Sai số tuyệt đối phần trăm trung bình\n"
        report += "\n"
        
        # Save to file
        filepath = self.output_dir / "model_comparison_report.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Báo cáo đã lưu: {filepath}")
        
        return report
    
    def create_final_report(self, project_name: str,
                           sections: Dict[str, str]) -> str:
        """
        Tạo báo cáo cuối cùng dự án.
        
        Args:
            project_name (str): Tên dự án
            sections (dict): {section_name: content}
            
        Returns:
            str: Nội dung báo cáo
        """
        print("\n" + "="*80)
        print("📋 TẠO BÁO CÁO CUỐI CÙNG DỰ ÁN")
        print("="*80)
        
        report = f"# {project_name}\n\n"
        report += f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 📑 Mục Lục\n\n"
        for i, section_name in enumerate(sections.keys(), 1):
            report += f"{i}. {section_name}\n"
        report += "\n"
        
        # Nội dung từng section
        for section_name, content in sections.items():
            report += f"## {section_name}\n\n"
            report += f"{content}\n\n"
        
        # Kết luận
        report += "---\n\n"
        report += "_Báo cáo được tạo tự động bởi Data Mining Pipeline_\n"
        
        # Save to file
        filepath = self.output_dir / "final_report.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Báo cáo cuối cùng: {filepath}")
        
        return report
    
    def summary_stats_table(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Tạo bảng thống kê tóm tắt.
        
        Args:
            data_dict (dict): {series_name: Series}
            
        Returns:
            pd.DataFrame: Bảng thống kê
        """
        stats = []
        
        for name, series in data_dict.items():
            stats.append({
                'Variable': name,
                'Count': len(series),
                'Mean': series.mean(),
                'Std': series.std(),
                'Min': series.min(),
                '25%': series.quantile(0.25),
                'Median': series.median(),
                '75%': series.quantile(0.75),
                'Max': series.max()
            })
        
        return pd.DataFrame(stats)
    
    def print_summary(self, title: str, data: Dict) -> None:
        """
        In tóm tắt kết quả.
        
        Args:
            title (str): Tiêu đề
            data (dict): Dữ liệu cần in
        """
        print("\n" + "="*60)
        print(f"📊 {title}")
        print("="*60)
        
        for key, value in data.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")


def main():
    """Hàm chính để test module."""
    print("📚 Example: Report Module\n")
    
    # Tạo sample data
    comparison_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge', 'Random Forest', 'XGBoost'],
        'MAE': [5.2, 4.8, 2.3, 1.9],
        'RMSE': [6.5, 6.2, 3.1, 2.5],
        'R2': [0.75, 0.78, 0.92, 0.95],
        'MAPE': [0.15, 0.14, 0.08, 0.06]
    })
    
    cv_results = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge', 'Random Forest', 'XGBoost'],
        'R2_Mean': [0.73, 0.76, 0.90, 0.93],
        'R2_Std': [0.05, 0.05, 0.03, 0.02],
        'MAE_Mean': [5.5, 5.1, 2.5, 2.1],
        'MAE_Std': [0.8, 0.7, 0.4, 0.3],
        'RMSE_Mean': [6.8, 6.4, 3.3, 2.7],
        'RMSE_Std': [1.2, 1.0, 0.6, 0.4]
    })
    
    # Tạo reporter
    reporter = EvaluationReporter("./reports")
    
    # Xuất CSV
    reporter.export_results_csv(comparison_df, "model_comparison.csv")
    
    # Tạo báo cáo so sánh
    reporter.create_model_comparison_report(
        comparison_df, 
        cv_results,
        best_model="XGBoost"
    )
    
    # Tạo báo cáo cuối cùng
    sections = {
        "Giới thiệu": "Dự án dự báo năng suất cây trồng...",
        "Dữ liệu": "Dataset bao gồm 1000 mẫu...",
        "Kết quả": "XGBoost là model tốt nhất với R² = 0.95",
        "Đề xuất": "Nên sử dụng XGBoost cho dự báo thực tế"
    }
    
    reporter.create_final_report(
        "Crop Yield Prediction - Data Mining Project",
        sections
    )


if __name__ == "__main__":
    main()
