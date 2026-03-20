"""
Clustering Analysis Module - Phân cụm dữ liệu

Module này thực hiện:
- KMeans clustering
- DBSCAN clustering
- Tính Silhouette score
- Phân tích đặc điểm mỗi cluster
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalysis:
    """
    Lớp để thực hiện phân tích clustering.
    
    Attributes:
        kmeans_model (KMeans): Model KMeans
        dbscan_model (DBSCAN): Model DBSCAN
        kmeans_labels (array): Labels từ KMeans
        dbscan_labels (array): Labels từ DBSCAN
    """
    
    def __init__(self):
        """Khởi tạo ClusteringAnalysis."""
        self.kmeans_model = None
        self.dbscan_model = None
        self.kmeans_labels = None
        self.dbscan_labels = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        
    def prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Chuẩn bị dữ liệu cho clustering (scaling).
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Dữ liệu đã scale
        """
        print("\n" + "="*60)
        print("📊 CHUẨN BỊ DỮ LIỆU CHO CLUSTERING")
        print("="*60)
        
        print(f"\n   Input shape: {X.shape}")
        
        # Xử lý missing values
        X_clean = X.fillna(X.mean())
        
        # Scaling
        self.X_scaled = self.scaler.fit_transform(X_clean)
        
        print(f"   ✓ Dữ liệu đã scale: {self.X_scaled.shape}")
        print(f"   ✓ Mean: {self.X_scaled.mean(axis=0)[:3].round(3)}")
        print(f"   ✓ Std: {self.X_scaled.std(axis=0)[:3].round(3)}")
        
        return self.X_scaled
    
    def elbow_method(self, X_scaled: np.ndarray,
                    max_k: int = 10,
                    n_init: int = 10) -> Dict[int, float]:
        """
        Tìm số cluster tối ưu sử dụng Elbow method.
        
        Args:
            X_scaled (np.ndarray): Dữ liệu đã scale
            max_k (int): Số cluster tối đa cần test
            n_init (int): Số lần chạy KMeans
            
        Returns:
            dict: Dictionary {k: inertia}
        """
        print("\n" + "="*60)
        print("📈 ELBOW METHOD - TÌM SỐ CLUSTER TỐI ƯU")
        print("="*60)
        
        print(f"\n   Test K từ 1 đến {max_k}...")
        
        inertias = {}
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
            kmeans.fit(X_scaled)
            inertias[k] = kmeans.inertia_
            print(f"   K={k:2d}: Inertia = {kmeans.inertia_:.2f}")
        
        print(f"\n   💡 Để chọn K tối ưu, hãy tìm điểm 'elbow' trong biểu đồ")
        
        return inertias
    
    def kmeans_clustering(self, X_scaled: np.ndarray,
                         n_clusters: int = 3,
                         n_init: int = 10) -> Tuple[np.ndarray, KMeans]:
        """
        Thực hiện KMeans clustering.
        
        Args:
            X_scaled (np.ndarray): Dữ liệu đã scale
            n_clusters (int): Số cluster
            n_init (int): Số lần chạy
            
        Returns:
            tuple: (labels, model)
        """
        print("\n" + "="*60)
        print("🎯 KMEANS CLUSTERING")
        print("="*60)
        
        print(f"\n   Số clusters: {n_clusters}")
        print(f"   Input shape: {X_scaled.shape}")
        print(f"   Đang huấn luyện...")
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=n_init
        )
        
        self.kmeans_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Tính silhouette score
        silhouette = silhouette_score(X_scaled, self.kmeans_labels)
        
        print(f"\n   ✓ Hoàn thành!")
        print(f"   • Inertia: {self.kmeans_model.inertia_:.2f}")
        print(f"   • Silhouette Score: {silhouette:.4f}")
        
        # Thống kê clusters
        print(f"\n   Phân bố Clusters:")
        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = count / len(self.kmeans_labels) * 100
            print(f"   • Cluster {cluster_id}: {count} mẫu ({percentage:.1f}%)")
        
        return self.kmeans_labels, self.kmeans_model
    
    def dbscan_clustering(self, X_scaled: np.ndarray,
                         eps: float = 0.5,
                         min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
        """
        Thực hiện DBSCAN clustering.
        
        Args:
            X_scaled (np.ndarray): Dữ liệu đã scale
            eps (float): Bán kính epsilon
            min_samples (int): Số mẫu tối thiểu
            
        Returns:
            tuple: (labels, model)
        """
        print("\n" + "="*60)
        print("🔍 DBSCAN CLUSTERING")
        print("="*60)
        
        print(f"\n   Eps: {eps}")
        print(f"   Min samples: {min_samples}")
        print(f"   Input shape: {X_scaled.shape}")
        print(f"   Đang huấn luyện...")
        
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = self.dbscan_model.fit_predict(X_scaled)
        
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)
        
        print(f"\n   ✓ Hoàn thành!")
        print(f"   • Số clusters: {n_clusters}")
        print(f"   • Noise points: {n_noise}")
        
        # Thống kê
        print(f"\n   Phân bố Clusters:")
        unique, counts = np.unique(self.dbscan_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"   • Noise: {count} mẫu")
            else:
                percentage = count / len(self.dbscan_labels) * 100
                print(f"   • Cluster {cluster_id}: {count} mẫu ({percentage:.1f}%)")
        
        return self.dbscan_labels, self.dbscan_model
    
    def analyze_clusters_kmeans(self, X: pd.DataFrame,
                               feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Phân tích đặc điểm các clusters từ KMeans.
        
        Args:
            X (pd.DataFrame): Feature matrix gốc
            feature_names (list): Tên các features
            
        Returns:
            pd.DataFrame: Thống kê mỗi cluster
        """
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH ĐẶC ĐIỂM CỦA TỪNG CLUSTER (KMeans)")
        print("="*60)
        
        if self.kmeans_labels is None:
            raise ValueError("Chưa chạy KMeans clustering!")
        
        X_with_labels = X.copy()
        X_with_labels['Cluster'] = self.kmeans_labels
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        print(f"\n   Phân tích {len(set(self.kmeans_labels))} clusters:")
        
        cluster_profiles = []
        
        for cluster_id in sorted(set(self.kmeans_labels)):
            cluster_data = X_with_labels[X_with_labels['Cluster'] == cluster_id]
            
            print(f"\n   Cluster {cluster_id} (n={len(cluster_data)}):")
            
            profile = {'Cluster': cluster_id, 'Size': len(cluster_data)}
            
            for feature in feature_names[:5]:  # In 5 features đầu
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                profile[f'{feature}_mean'] = mean_val
                print(f"      • {feature}: {mean_val:.2f} ± {std_val:.2f}")
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)
    
    def analyze_clusters_dbscan(self, X: pd.DataFrame,
                               feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Phân tích đặc điểm các clusters từ DBSCAN.
        
        Args:
            X (pd.DataFrame): Feature matrix gốc
            feature_names (list): Tên các features
            
        Returns:
            pd.DataFrame: Thống kê mỗi cluster
        """
        print("\n" + "="*60)
        print("📊 PHÂN TÍCH ĐẶC ĐIỂM CỦA TỪNG CLUSTER (DBSCAN)")
        print("="*60)
        
        if self.dbscan_labels is None:
            raise ValueError("Chưa chạy DBSCAN clustering!")
        
        X_with_labels = X.copy()
        X_with_labels['Cluster'] = self.dbscan_labels
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        unique_labels = set(self.dbscan_labels)
        print(f"\n   Phân tích {len(unique_labels)} clusters/groups:")
        
        cluster_profiles = []
        
        for cluster_id in sorted(unique_labels):
            cluster_data = X_with_labels[X_with_labels['Cluster'] == cluster_id]
            
            if cluster_id == -1:
                print(f"\n   Noise Points (n={len(cluster_data)}):")
            else:
                print(f"\n   Cluster {cluster_id} (n={len(cluster_data)}):")
            
            profile = {
                'Cluster': cluster_id if cluster_id != -1 else 'Noise',
                'Size': len(cluster_data)
            }
            
            for feature in feature_names[:5]:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                profile[f'{feature}_mean'] = mean_val
                print(f"      • {feature}: {mean_val:.2f} ± {std_val:.2f}")
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)
    
    def get_silhouette_scores(self, X_scaled: np.ndarray,
                             labels: np.ndarray) -> float:
        """
        Tính Silhouette score.
        
        Args:
            X_scaled (np.ndarray): Dữ liệu đã scale
            labels (np.ndarray): Cluster labels
            
        Returns:
            float: Silhouette score
        """
        score = silhouette_score(X_scaled, labels)
        return score
    
    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt clustering.
        
        Returns:
            dict: Thông tin tóm tắt
        """
        summary = {
            'kmeans_trained': self.kmeans_model is not None,
            'dbscan_trained': self.dbscan_model is not None
        }
        
        if self.kmeans_labels is not None:
            summary['kmeans_n_clusters'] = len(set(self.kmeans_labels))
            summary['kmeans_silhouette'] = silhouette_score(
                self.X_scaled, self.kmeans_labels
            ) if self.X_scaled is not None else None
        
        if self.dbscan_labels is not None:
            n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
            summary['dbscan_n_clusters'] = n_clusters
            summary['dbscan_n_noise'] = list(self.dbscan_labels).count(-1)
        
        return summary


def main():
    """Hàm chính để test module."""
    print("📚 Example: Clustering Analysis\n")
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 200
    
    # Tạo 3 clusters rõ ràng
    cluster1 = np.random.randn(n_samples//3, 2) + [0, 0]
    cluster2 = np.random.randn(n_samples//3, 2) + [5, 5]
    cluster3 = np.random.randn(n_samples//3, 2) + [2.5, -2.5]
    
    X = pd.DataFrame(
        np.vstack([cluster1, cluster2, cluster3]),
        columns=['Feature1', 'Feature2']
    )
    
    print("📊 Sample Data:")
    print(f"Shape: {X.shape}\n")
    
    # Khởi tạo clustering
    clustering = ClusteringAnalysis()
    
    # Chuẩn bị dữ liệu
    X_scaled = clustering.prepare_data(X)
    
    # Elbow method
    inertias = clustering.elbow_method(X_scaled, max_k=10)
    
    # KMeans
    clustering.kmeans_clustering(X_scaled, n_clusters=3)
    
    # DBSCAN
    clustering.dbscan_clustering(X_scaled, eps=0.5, min_samples=5)
    
    # Phân tích clusters
    print(clustering.analyze_clusters_kmeans(X))


if __name__ == "__main__":
    main()
