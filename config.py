"""
Конфигурационный файл для проекта сегментации клиентов банка
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Конфигурация данных"""
    data_file: str = 'DECENTRATHON_3.0.parquet'
    output_dir: str = 'results'
    
@dataclass
class FeatureConfig:
    """Конфигурация признаков"""
    
    clustering_features: List[str] = None
    
    
    categorical_features: List[str] = None
    
    
    time_features: List[str] = None
    
    def __post_init__(self):
        if self.clustering_features is None:
            self.clustering_features = [
                'total_transactions', 'total_amount', 'avg_amount', 'median_amount',
                'std_amount', 'amount_range', 'unique_merchants', 'unique_categories',
                'unique_cities', 'purchase_ratio', 'avg_merchants_per_transaction',
                'spending_consistency', 'preferred_hour', 'preferred_day'
            ]
        
        if self.categorical_features is None:
            self.categorical_features = [
                'preferred_pos_mode', 'preferred_wallet', 'activity_level', 'spending_level'
            ]
            
        if self.time_features is None:
            self.time_features = [
                'hour', 'day_of_week', 'month', 'preferred_hour', 'preferred_day'
            ]

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    
    max_clusters: int = 15
    random_state: int = 42
    n_init: int = 10
    
    
    outlier_contamination: float = 0.05
    
    
    pca_components: int = 2
    
@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""
    figure_size: tuple = (15, 10)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'husl'
    
    
    interactive_plot: str = 'cluster_analysis.html'
    static_plot: str = 'cluster_analysis_static.png'
    optimization_plot: str = 'cluster_optimization.png'

@dataclass
class OutputConfig:
    """Конфигурация выходных файлов"""
    segments_file: str = 'client_segments.csv'
    summary_file: str = 'cluster_summary_statistics.csv'
    detailed_report: str = 'segmentation_report.html'
    business_recommendations: str = 'business_recommendations.md'


class Config:
    """Главный класс конфигурации"""
    
    def __init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        
        
        os.makedirs(self.data.output_dir, exist_ok=True)
    
    def get_output_path(self, filename: str) -> str:
        """Получить полный путь к выходному файлу"""
        return os.path.join(self.data.output_dir, filename)


config = Config() 