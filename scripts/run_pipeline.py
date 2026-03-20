"""
Pipeline Script - Chạy toàn bộ quy trình khai phá dữ liệu

Script này tích hợp tất cả các module và chạy pipeline hoàn chỉnh.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Thêm src vào path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import các modules
from data.loader import DataLoader
from data.cleaner import DataCleaner
from features.builder import FeatureBuilder
from mining.association import AssociationRuleMining
from mining.clustering import ClusteringAnalysis
from models.regression import RegressionModels
from evaluation.metrics import ModelMetrics, ResidualAnalysis
from evaluation.report import EvaluationReporter
from visualization.plots import PlotterLibrary


class DataMiningPipeline:
    """
    Lớp chính để chạy toàn bộ pipeline khai phá dữ liệu.
    """
    
    def __init__(self, config_file: str = "./configs/params.yaml"):
        """
        Khởi tạo pipeline.
        
        Args:
            config_file (str): Đường dẫn file cấu hình
        """
        # Tải cấu hình
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Khởi tạo các components
        self.loader = DataLoader(self.config['data_loading'].get('files', {}).get('raw', './data/raw'), use_cleaned=True)
        self.cleaner = DataCleaner()
        self.feature_builder = FeatureBuilder()
        self.mining_association = AssociationRuleMining()
        self.clustering = ClusteringAnalysis()
        self.regressor = RegressionModels()
        self.metrics = ModelMetrics()
        self.residuals = ResidualAnalysis()
        self.reporter = EvaluationReporter(self.config['paths']['reports'])
        self.plotter = PlotterLibrary()
        
        # Lưu dữ liệu
        self.df_raw = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = {}
        
        print("✓ Pipeline đã khởi tạo")
    
    def step_1_load_data(self):
        """Bước 1: Tải dữ liệu."""
        print("\n" + "="*80)
        print("BƯỚC 1: TẢI DỮ LIỆU")
        print("="*80)
        
        files = self.config['data_loading']['files']
        
        # Tải tất cả files
        self.df_raw, self.loader.datasets = self.loader.get_complete_dataset()
        
        print(f"\n✓ Dữ liệu chính:")
        print(f"   Shape: {self.df_raw.shape}")
        print(f"   Columns: {self.df_raw.columns.tolist()}")
        
        return self.df_raw
    
    def step_2_eda(self):
        """Bước 2: Phân tích khám phá (EDA)."""
        print("\n" + "="*80)
        print("BƯỚC 2: PHÂN TÍCH KHÁM PHÁ (EDA)")
        print("="*80)
        
        target = self.config['data_loading']['target_column']
        
        # Thông tin cơ bản
        self.loader.get_dataset_info(self.df_raw, "Raw Dataset")
        self.loader.check_missing_values(self.df_raw, "Raw Dataset")
        self.loader.get_statistical_summary(self.df_raw, "Raw Dataset")
        
        # Vẽ biểu đồ
        output_dir = Path(self.config['paths']['figures'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # self.plotter.plot_yield_distribution(self.df_raw, target, 
        #                                     str(output_dir / "01_yield_distribution.png"))
        # self.plotter.plot_correlation_heatmap(self.df_raw, 
        #                                       str(output_dir / "02_correlation_heatmap.png"))
        # self.plotter.plot_boxplot(self.df_raw, 
        #                           str(output_dir / "03_boxplot.png"))
        
        return self.df_raw
    
    def step_3_preprocessing(self):
        """Bước 3: Tiền xử lý dữ liệu."""
        print("\n" + "="*80)
        print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU")
        print("="*80)
        
        config = self.config['preprocessing']
        target = self.config['data_loading']['target_column']
        
        # Chạy pipeline xử lý
        self.df_processed = self.cleaner.complete_pipeline(
            self.df_raw,
            handle_missing=config['handle_missing']['fillna_numeric'],
            handle_outliers=True,
            encode=True,
            scale=True,
            target_column=target
        )
        
        # Lưu dữ liệu đã xử lý
        output_dir = Path(self.config['paths']['processed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.loader.save_processed_data(
            self.df_processed, 
            "processed_data.csv",
            str(output_dir)
        )
        
        return self.df_processed
    
    def step_4_feature_engineering(self):
        """Bước 4: Kỹ thuật tạo features."""
        print("\n" + "="*80)
        print("BƯỚC 4: KỸ THUẬT TẠO FEATURES")
        print("="*80)
        
        target = self.config['data_loading']['target_column']
        
        # Tách target
        X = self.df_processed.drop(columns=[target])
        y = self.df_processed[target]
        
        # Tạo polynomial features
        X_poly = self.feature_builder.create_polynomial_features(X, degree=2)
        
        # Tạo interaction features
        X_interaction = self.feature_builder.create_interaction_features(
            X_poly, 
            max_interactions=20
        )
        
        # Tính feature importance
        self.feature_builder.calculate_feature_importance_rf(X_interaction, y)
        
        # Chọn top features
        n_features = self.config['feature_engineering']['n_features']
        self.feature_builder.select_features(X_interaction, y, n_features=n_features)
        
        # Lấy X đã được chọn
        X_selected = self.feature_builder.get_selected_dataframe(X_interaction)
        
        # Vẽ biểu đồ feature importance
        output_dir = Path(self.config['paths']['figures'])
        importance_df = self.feature_builder.get_importance_dataframe()
        
        # self.plotter.plot_feature_importance(importance_df, top_n=15,
        #                                     output_path=str(output_dir / "04_feature_importance.png"))
        
        print(f"\n✓ Kỹ thuật features hoàn thành:")
        print(f"   Original shape: {X.shape}")
        print(f"   After engineering: {X_interaction.shape}")
        print(f"   After selection: {X_selected.shape}")
        
        return X_selected, y
    
    def step_5_mining(self, X: pd.DataFrame, y: pd.Series):
        """Bước 5: Khai phá dữ liệu."""
        print("\n" + "="*80)
        print("BƯỚC 5: KHAI PHÁ DỮ LIỆU (ASSOCIATION RULES & CLUSTERING)")
        print("="*80)
        
        # Association Rule Mining
        print("\n--- Association Rule Mining ---")
        config_mining = self.config['mining']
        
        # Rời rạc hóa
        df_disc = self.mining_association.discretize_features(
            X.head(1000),  # Lấy sample để không quá chậm
            n_bins=config_mining['discretization']['rainfall_bins']
        )
        
        # Tạo itemsets
        itemset_matrix = self.mining_association.create_itemsets(df_disc)
        
        # Tìm frequent itemsets
        self.mining_association.find_frequent_itemsets(
            itemset_matrix,
            min_support=config_mining['apriori']['min_support']
        )
        
        # Tạo rules
        self.mining_association.generate_rules(
            min_confidence=config_mining['apriori']['min_confidence'],
            min_lift=config_mining['apriori']['min_lift']
        )
        
        # Clustering
        print("\n--- Clustering Analysis ---")
        X_scaled = self.clustering.prepare_data(X.head(1000))  # Sample
        
        # Elbow method
        inertias = self.clustering.elbow_method(X_scaled, max_k=10)
        
        # KMeans
        self.clustering.kmeans_clustering(X_scaled, n_clusters=3)
        
        # DBSCAN
        self.clustering.dbscan_clustering(X_scaled, eps=0.5, min_samples=5)
    
    def step_6_split_data(self, X: pd.DataFrame, y: pd.Series):
        """Bước 6: Chia dữ liệu train/test."""
        print("\n" + "="*80)
        print("BƯỚC 6: CHIA DỮ LIỆU TRAIN/TEST")
        print("="*80)
        
        config = self.config['modeling']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        print(f"\n   Train set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        print(f"   y_train shape: {self.y_train.shape}")
        print(f"   y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def step_7_train_models(self):
        """Bước 7: Huấn luyện các mô hình."""
        print("\n" + "="*80)
        print("BƯỚC 7: HUẤN LUYỆN CÁC MÔ HÌNH")
        print("="*80)
        
        # Tạo models
        self.regressor.create_models()
        
        # Huấn luyện
        self.regressor.train_models(self.X_train, self.y_train)
        
        print(f"\n✓ Đã huấn luyện {len(self.regressor.trained_models)} models")
        
        return self.regressor.trained_models
    
    def step_8_evaluate_models(self):
        """Bước 8: Đánh giá mô hình."""
        print("\n" + "="*80)
        print("BƯỚC 8: ĐÁNH GIÁ CÁC MÔ HÌNH")
        print("="*80)
        
        # Đánh giá trên test set
        results_df = self.regressor.evaluate_models(self.X_test, self.y_test)
        
        print("\n📊 Kết quả Test Set:")
        print(results_df.to_string(index=False))
        
        # Cross-validation
        print("\n\n--- Cross-Validation ---")
        cv_results = self.regressor.cross_validation(
            self.X_train,
            self.y_train,
            cv_folds=self.config['modeling']['cv_folds']
        )
        
        print("\n📊 Kết quả Cross-Validation:")
        print(cv_results.to_string(index=False))
        
        # Lưu results
        output_dir = Path(self.config['paths']['tables'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / "model_comparison.csv", index=False)
        cv_results.to_csv(output_dir / "cv_results.csv", index=False)
        
        print(f"\n✓ Đã lưu kết quả vào {output_dir}")
        
        return results_df, cv_results
    
    def step_9_residual_analysis(self):
        """Bước 9: Phân tích Residuals."""
        print("\n" + "="*80)
        print("BƯỚC 9: PHÂN TÍCH RESIDUALS")
        print("="*80)
        
        output_dir = Path(self.config['paths']['figures'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.regressor.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            # Phân tích residuals
            self.residuals.analyze_residuals(
                self.y_test.values,
                y_pred,
                model_name
            )
            
            # Vẽ residual plots
            # self.plotter.plot_residuals(
            #     self.y_test.values, y_pred, model_name,
            #     str(output_dir / f"residuals_{model_name.replace(' ', '_')}.png")
            # )
    
    def step_10_visualization(self):
        """Bước 10: Trực quan hóa kết quả."""
        print("\n" + "="*80)
        print("BƯỚC 10: TRỰC QUAN HÓA KẾT QUẢ")
        print("="*80)
        
        output_dir = Path(self.config['paths']['figures'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.regressor.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            # Actual vs Predicted
            # self.plotter.plot_actual_vs_predicted(
            #     self.y_test.values, y_pred, model_name,
            #     str(output_dir / f"actual_vs_pred_{model_name.replace(' ', '_')}.png")
            # )
        
        print("✓ Trực quan hóa hoàn thành")
    
    def step_11_generate_report(self, results_df, cv_results):
        """Bước 11: Tạo báo cáo."""
        print("\n" + "="*80)
        print("BƯỚC 11: TẠO BÁO CÁO")
        print("="*80)
        
        best_model, best_r2 = self.regressor.get_best_model()
        
        # Báo cáo so sánh models
        self.reporter.create_model_comparison_report(
            results_df,
            cv_results,
            best_model
        )
        
        # Xuất CSV
        output_dir = Path(self.config['paths']['tables'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reporter.export_results_csv(results_df, "model_comparison.csv")
        
        print("✓ Báo cáo đã tạo")
    
    def run_full_pipeline(self):
        """Chạy toàn bộ pipeline."""
        print("\n" + "🚀 "*40)
        print("BẮT ĐẦU PIPELINE KHAI PHÁ DỮ LIỆU")
        print("🚀 "*40)
        
        try:
            # Step 1
            self.step_1_load_data()
            
            # Step 2
            self.step_2_eda()
            
            # Step 3
            self.step_3_preprocessing()
            
            # Step 4
            X_selected, y = self.step_4_feature_engineering()
            
            # Step 5
            self.step_5_mining(X_selected, y)
            
            # Step 6
            self.step_6_split_data(X_selected, y)
            
            # Step 7
            self.step_7_train_models()
            
            # Step 8
            results_df, cv_results = self.step_8_evaluate_models()
            
            # Step 9
            self.step_9_residual_analysis()
            
            # Step 10
            self.step_10_visualization()
            
            # Step 11
            self.step_11_generate_report(results_df, cv_results)
            
            print("\n" + "✅ "*40)
            print("PIPELINE HOÀN THÀNH THÀNH CÔNG!")
            print("✅ "*40)
            
        except Exception as e:
            print(f"\n❌ LỖI: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Hàm chính."""
    # Tìm config file
    config_file = "./configs/params.yaml"
    
    if not Path(config_file).exists():
        print(f"⚠️  File cấu hình không tìm thấy: {config_file}")
        config_file = "configs/params.yaml"
    
    # Khởi tạo và chạy pipeline
    pipeline = DataMiningPipeline(config_file)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
