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
                'spending_consistency', 'preferred_hour', 'preferred_day',
                # Новые признаки для лучшего разделения сегментов
                'high_value_transactions_ratio', 'travel_indicator', 'premium_merchant_ratio',
                'weekend_activity_ratio', 'evening_activity_ratio'
            ]
        
        if self.categorical_features is None:
            self.categorical_features = [
                'preferred_pos_mode', 'preferred_wallet', 'activity_level', 'spending_level',
                'client_tier', 'travel_pattern'
            ]
            
        if self.time_features is None:
            self.time_features = [
                'hour', 'day_of_week', 'month', 'preferred_hour', 'preferred_day'
            ]

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    
    max_clusters: int = 15
    target_clusters: int = 4  # Целевое количество кластеров
    random_state: int = 42
    n_init: int = 20  # Увеличиваем для лучшей стабильности
    
    
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

@dataclass
class SegmentConfig:
    """Конфигурация сегментов"""
    segment_definitions: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.segment_definitions is None:
            self.segment_definitions = {
                'regular': {
                    'name': '🔄 Обычные клиенты',
                    'emoji': '🔄',
                    'description': 'Стандартная активность и средние траты',
                    'priority': 'Средний',
                    'color': '#3498db'
                },
                'sleeping': {
                    'name': '😴 Спящие клиенты', 
                    'emoji': '😴',
                    'description': 'Низкая активность, требуют реактивации',
                    'priority': 'Высокий',
                    'color': '#95a5a6'
                },
                'vip': {
                    'name': '💎 VIP клиенты',
                    'emoji': '💎', 
                    'description': 'Высокие траты, премиум сегмент',
                    'priority': 'Очень высокий',
                    'color': '#f39c12'
                },
                'traveler': {
                    'name': '✈️ Путешественники',
                    'emoji': '✈️',
                    'description': 'Активные траты в разных городах',
                    'priority': 'Высокий', 
                    'color': '#e74c3c'
                }
            }

class Config:
    """Главный класс конфигурации"""
    
    def __init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        self.segments = SegmentConfig()
        
        
        os.makedirs(self.data.output_dir, exist_ok=True)
    
    def get_output_path(self, filename: str) -> str:
        """Получить полный путь к выходному файлу"""
        return os.path.join(self.data.output_dir, filename)


config = Config() 