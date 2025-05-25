"""
Модуль для обработки данных и создания поведенческих характеристик клиентов
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from config import config

class DataProcessor:
    """Класс для обработки данных и создания признаков"""
    
    def __init__(self, data_path: str):
        """Инициализация с загрузкой данных"""
        print("🔄 Загружаем данные...")
        self.df = pd.read_parquet(data_path)
        print(f"✅ Загружено {self.df.shape[0]:,} транзакций")
        
        self.client_features = None
        self.processed_features = None
        self.feature_descriptions = self._get_feature_descriptions()
        
    @property
    def data(self):
        """Свойство для доступа к данным (алиас для df)"""
        return self.df
    
    @data.setter
    def data(self, value):
        """Сеттер для данных"""
        self.df = value
        
    def _get_feature_descriptions(self) -> Dict[str, str]:
        """Описания поведенческих характеристик"""
        return {
            'total_transactions': 'Общее количество транзакций клиента',
            'total_amount': 'Общая сумма всех транзакций клиента (тенге)',
            'avg_amount': 'Средний размер транзакции (тенге)',
            'median_amount': 'Медианный размер транзакции (тенге)',
            'std_amount': 'Стандартное отклонение сумм транзакций',
            'min_amount': 'Минимальная сумма транзакции (тенге)',
            'max_amount': 'Максимальная сумма транзакции (тенге)',
            'amount_range': 'Диапазон сумм транзакций (max - min)',
            'preferred_hour': 'Предпочитаемое время совершения транзакций (час)',
            'preferred_day': 'Предпочитаемый день недели (0=понедельник)',
            'unique_merchants': 'Количество уникальных мерчантов',
            'unique_categories': 'Количество уникальных категорий MCC',
            'unique_cities': 'Количество уникальных городов',
            'purchase_count': 'Количество транзакций типа "Purchase"',
            'purchase_ratio': 'Доля покупок от общего числа транзакций',
            'avg_merchants_per_transaction': 'Среднее количество мерчантов на транзакцию',
            'spending_consistency': 'Консистентность трат (обратная к коэф. вариации)',
            'preferred_pos_mode': 'Предпочитаемый способ ввода карты',
            'preferred_wallet': 'Предпочитаемый тип электронного кошелька',
            'activity_level': 'Уровень активности (Низкая/Средняя/Высокая/Очень высокая)',
            'spending_level': 'Уровень трат (Низкий/Средний/Высокий/Премиум)'
        }
    
    def explore_data(self) -> Dict[str, Any]:
        """Первичный анализ данных"""
        print("\n" + "="*50)
        print("📊 АНАЛИЗ ДАННЫХ")
        print("="*50)
        
        
        data_info = {
            'shape': self.df.shape,
            'period': (self.df['transaction_timestamp'].min(), self.df['transaction_timestamp'].max()),
            'unique_clients': self.df['card_id'].nunique(),
            'unique_merchants': self.df['merchant_id'].nunique(),
            'total_volume': self.df['transaction_amount_kzt'].sum(),
            'avg_transaction': self.df['transaction_amount_kzt'].mean(),
            'median_transaction': self.df['transaction_amount_kzt'].median()
        }
        
        print(f"Размер данных: {data_info['shape']}")
        print(f"Период данных: {data_info['period'][0]} - {data_info['period'][1]}")
        print(f"Количество уникальных клиентов: {data_info['unique_clients']:,}")
        print(f"Количество уникальных мерчантов: {data_info['unique_merchants']:,}")
        print(f"Общий оборот: {data_info['total_volume']:,.2f} тенге")
        print(f"Средняя сумма транзакции: {data_info['avg_transaction']:.2f} тенге")
        print(f"Медианная сумма транзакции: {data_info['median_transaction']:.2f} тенге")
        
        
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("\n🔍 Пропущенные значения:")
            print(missing_data[missing_data > 0])
            data_info['missing_values'] = missing_data[missing_data > 0].to_dict()
        else:
            print("\n✅ Пропущенные значения отсутствуют")
            data_info['missing_values'] = {}
        
        return data_info
    
    def create_behavioral_features(self) -> pd.DataFrame:
        """
        Создание поведенческих характеристик клиентов
        
        Обоснование выбора метрик:
        1. Метрики активности - показывают интенсивность использования карты
        2. Метрики сумм - характеризуют покупательную способность
        3. Временные паттерны - выявляют предпочтения во времени
        4. Метрики разнообразия - показывают широту интересов клиента
        5. Технологические предпочтения - современность клиента
        6. Расчетные метрики - дополнительные инсайты о поведении
        """
        print("\n🔧 Создаем поведенческие характеристики клиентов...")
        
        
        self.df['hour'] = self.df['transaction_timestamp'].dt.hour
        self.df['day_of_week'] = self.df['transaction_timestamp'].dt.dayofweek
        self.df['month'] = self.df['transaction_timestamp'].dt.month
        
        
        print("   📈 Рассчитываем основные метрики активности...")
        client_agg = self.df.groupby('card_id').agg({
            
            'transaction_id': 'count',  
            'transaction_amount_kzt': ['sum', 'mean', 'median', 'std', 'min', 'max'],
            
            
            'hour': lambda x: x.mode().iloc[0] if not x.mode().empty else 12,
            'day_of_week': lambda x: x.mode().iloc[0] if not x.mode().empty else 0,
            
            
            'merchant_id': 'nunique',  
            'mcc_category': 'nunique',  
            'merchant_city': 'nunique',  
            
            
            'transaction_type': lambda x: (x == 'Purchase').sum(),
            
            
            'pos_entry_mode': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'wallet_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).round(2)
        
        
        client_agg.columns = [
            'total_transactions', 'total_amount', 'avg_amount', 'median_amount', 
            'std_amount', 'min_amount', 'max_amount', 'preferred_hour', 
            'preferred_day', 'unique_merchants', 'unique_categories', 
            'unique_cities', 'purchase_count', 'preferred_pos_mode', 'preferred_wallet'
        ]
        
        print("   🧮 Рассчитываем дополнительные метрики...")
        
        
        client_agg['amount_range'] = client_agg['max_amount'] - client_agg['min_amount']
        client_agg['purchase_ratio'] = client_agg['purchase_count'] / client_agg['total_transactions']
        client_agg['avg_merchants_per_transaction'] = client_agg['unique_merchants'] / client_agg['total_transactions']
        
        
        client_agg['spending_consistency'] = 1 / (1 + client_agg['std_amount'] / client_agg['avg_amount'].replace(0, 1))
        
        print("   🏷️ Создаем категориальные признаки...")
        
        
        
        activity_quantiles = client_agg['total_transactions'].quantile([0.25, 0.5, 0.75])
        client_agg['activity_level'] = pd.cut(
            client_agg['total_transactions'], 
            bins=[0, activity_quantiles[0.25], activity_quantiles[0.5], activity_quantiles[0.75], float('inf')], 
            labels=['Низкая', 'Средняя', 'Высокая', 'Очень высокая']
        )
        
        
        spending_quantiles = client_agg['total_amount'].quantile([0.25, 0.5, 0.75])
        client_agg['spending_level'] = pd.cut(
            client_agg['total_amount'], 
            bins=[0, spending_quantiles[0.25], spending_quantiles[0.5], spending_quantiles[0.75], float('inf')], 
            labels=['Низкий', 'Средний', 'Высокий', 'Премиум']
        )
        
        self.client_features = client_agg
        print(f"✅ Создано {len(self.client_features)} профилей клиентов с {len(client_agg.columns)} характеристиками")
        
        return client_agg
    
    def prepare_features_for_clustering(self) -> Tuple[np.ndarray, pd.Index]:
        """Подготовка признаков для кластеризации"""
        print("\n🎯 Подготавливаем признаки для кластеризации...")
        
        if self.client_features is None:
            raise ValueError("Сначала необходимо создать признаки с помощью create_behavioral_features()")
        
        
        numeric_features = config.features.clustering_features
        
        print(f"   📊 Используем {len(numeric_features)} числовых признаков:")
        for i, feature in enumerate(numeric_features, 1):
            description = self.feature_descriptions.get(feature, "Описание отсутствует")
            print(f"      {i:2d}. {feature}: {description}")
        
        
        features_df = self.client_features[numeric_features].fillna(0)
        
        print(f"   🔍 Обнаружение выбросов...")
        
        isolation_forest = IsolationForest(
            contamination=config.model.outlier_contamination, 
            random_state=config.model.random_state
        )
        outlier_mask = isolation_forest.fit_predict(features_df) == 1
        
        outlier_count = (~outlier_mask).sum()
        outlier_percentage = (~outlier_mask).mean() * 100
        print(f"   🚫 Обнаружено {outlier_count} выбросов ({outlier_percentage:.1f}%)")
        
        
        features_clean = features_df[outlier_mask]
        
        print(f"   ⚖️ Масштабирование признаков...")
        
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features_clean)
        
        self.processed_features = {
            'scaled_data': scaled_features,
            'feature_names': numeric_features,
            'scaler': scaler,
            'clean_indices': features_clean.index,
            'outlier_info': {
                'total_outliers': outlier_count,
                'outlier_percentage': outlier_percentage,
                'contamination_threshold': config.model.outlier_contamination
            }
        }
        
        print(f"✅ Подготовлено {scaled_features.shape[0]} клиентов для кластеризации")
        print(f"   📏 Размерность данных: {scaled_features.shape[1]} признаков")
        
        return scaled_features, features_clean.index
    
    def get_feature_importance_analysis(self) -> pd.DataFrame:
        """Анализ важности признаков"""
        if self.processed_features is None:
            raise ValueError("Сначала необходимо подготовить признаки для кластеризации")
        
        
        feature_stats = pd.DataFrame({
            'feature': self.processed_features['feature_names'],
            'description': [self.feature_descriptions.get(f, '') for f in self.processed_features['feature_names']]
        })
        
        
        clean_indices = self.processed_features['clean_indices']
        original_data = self.client_features.loc[clean_indices, self.processed_features['feature_names']]
        
        feature_stats['mean'] = original_data.mean().values
        feature_stats['std'] = original_data.std().values
        feature_stats['min'] = original_data.min().values
        feature_stats['max'] = original_data.max().values
        feature_stats['coefficient_of_variation'] = (feature_stats['std'] / feature_stats['mean']).round(3)
        
        return feature_stats.round(2)
    
    def save_feature_analysis(self, output_path: str = None):
        """Сохранение анализа признаков"""
        if output_path is None:
            output_path = config.get_output_path('feature_analysis.csv')
        
        feature_analysis = self.get_feature_importance_analysis()
        feature_analysis.to_csv(output_path, index=False, encoding='utf-8')
        print(f"📊 Анализ признаков сохранен: {output_path}")
        
        return feature_analysis 