#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создание файла с результатами сегментации в формате .parquet
Включает данные о клиентах, их сегментах и характеристиках
"""

import pandas as pd
import json
import os
from datetime import datetime

def create_segmentation_parquet():
    """Создает файл с результатами сегментации в формате .parquet"""
    
    print("🔄 Создание файла результатов сегментации в формате .parquet...")
    
    # Загружаем основные данные сегментации
    print("📊 Загрузка данных сегментации...")
    segments_df = pd.read_csv('results/client_segments.csv')
    
    # Загружаем профили сегментов для добавления описаний
    print("📋 Загрузка профилей сегментов...")
    with open('results/segment_profiles.json', 'r', encoding='utf-8') as f:
        segment_profiles = json.load(f)
    
    # Создаем словарь для маппинга кластеров к названиям сегментов
    cluster_to_segment = {}
    for segment_name, segment_data in segment_profiles.items():
        if 'Обычные' in segment_name:
            cluster_to_segment[0] = {
                'segment_name': 'Обычные клиенты',
                'segment_emoji': '🔄',
                'segment_description': 'Активные клиенты с высокой цифровой активностью',
                'business_strategy': 'Развитие и удержание'
            }
        elif 'Спящие' in segment_name:
            cluster_to_segment[1] = {
                'segment_name': 'Спящие клиенты', 
                'segment_emoji': '😴',
                'segment_description': 'Клиенты с низкой активностью и ограниченным использованием услуг',
                'business_strategy': 'Реактивация и вовлечение'
            }
    
    # Добавляем информацию о сегментах
    print("🏷️ Добавление информации о сегментах...")
    segments_df['segment_name'] = segments_df['cluster'].map(lambda x: cluster_to_segment.get(x, {}).get('segment_name', 'Неизвестный'))
    segments_df['segment_emoji'] = segments_df['cluster'].map(lambda x: cluster_to_segment.get(x, {}).get('segment_emoji', '❓'))
    segments_df['segment_description'] = segments_df['cluster'].map(lambda x: cluster_to_segment.get(x, {}).get('segment_description', 'Описание недоступно'))
    segments_df['business_strategy'] = segments_df['cluster'].map(lambda x: cluster_to_segment.get(x, {}).get('business_strategy', 'Стратегия не определена'))
    
    # Добавляем категории для числовых значений
    print("📈 Добавление категорий...")
    
    # Категории по размеру транзакций
    segments_df['transaction_volume_category'] = pd.cut(
        segments_df['total_transactions'], 
        bins=[0, 1000, 5000, 10000, float('inf')],
        labels=['Низкий', 'Средний', 'Высокий', 'Очень высокий']
    )
    
    # Категории по сумме оборота
    segments_df['revenue_category'] = pd.cut(
        segments_df['total_amount'], 
        bins=[0, 50000000, 100000000, 200000000, float('inf')],
        labels=['Низкий', 'Средний', 'Высокий', 'Премиум']
    )
    
    # Категории по среднему чеку
    segments_df['avg_ticket_category'] = pd.cut(
        segments_df['avg_amount'], 
        bins=[0, 15000, 25000, 40000, float('inf')],
        labels=['Эконом', 'Стандарт', 'Комфорт', 'Премиум']
    )
    
    # Добавляем временные метки
    segments_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
    segments_df['analysis_timestamp'] = datetime.now().isoformat()
    
    # Добавляем метаданные модели
    segments_df['model_type'] = 'K-means'
    segments_df['model_version'] = '1.0'
    segments_df['silhouette_score'] = 0.661
    segments_df['n_clusters'] = 2
    
    # Переупорядочиваем колонки для лучшей читаемости
    column_order = [
        'card_id', 'cluster', 'segment_name', 'segment_emoji', 'segment_description', 'business_strategy',
        'total_transactions', 'transaction_volume_category',
        'total_amount', 'revenue_category', 
        'avg_amount', 'avg_ticket_category',
        'median_amount', 'std_amount', 'min_amount', 'max_amount',
        'preferred_hour', 'preferred_day', 
        'unique_merchants', 'unique_categories', 'unique_cities',
        'purchase_count', 'preferred_pos_mode', 'preferred_wallet',
        'amount_range', 'purchase_ratio', 'avg_merchants_per_transaction',
        'spending_consistency', 'activity_level', 'spending_level',
        'analysis_date', 'analysis_timestamp',
        'model_type', 'model_version', 'silhouette_score', 'n_clusters'
    ]
    
    segments_df = segments_df[column_order]
    
    # Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)
    
    # Сохраняем в формате parquet
    output_path = 'results/customer_segmentation_results.parquet'
    print(f"💾 Сохранение в файл: {output_path}")
    
    segments_df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Создаем дополнительный файл с метаданными
    metadata = {
        'dataset_info': {
            'name': 'Customer Segmentation Results',
            'description': 'Результаты сегментации банковских клиентов с использованием машинного обучения',
            'total_customers': len(segments_df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'model_used': 'K-means clustering',
            'silhouette_score': 0.661,
            'n_clusters': 2
        },
        'segments_summary': {
            'segment_0': {
                'name': 'Обычные клиенты',
                'emoji': '🔄',
                'count': len(segments_df[segments_df['cluster'] == 0]),
                'percentage': round(len(segments_df[segments_df['cluster'] == 0]) / len(segments_df) * 100, 2)
            },
            'segment_1': {
                'name': 'Спящие клиенты',
                'emoji': '😴', 
                'count': len(segments_df[segments_df['cluster'] == 1]),
                'percentage': round(len(segments_df[segments_df['cluster'] == 1]) / len(segments_df) * 100, 2)
            }
        },
        'data_schema': {
            'card_id': 'Уникальный идентификатор клиента',
            'cluster': 'Номер кластера (0 или 1)',
            'segment_name': 'Название сегмента',
            'segment_emoji': 'Эмодзи сегмента',
            'segment_description': 'Описание сегмента',
            'business_strategy': 'Бизнес-стратегия для сегмента',
            'total_transactions': 'Общее количество транзакций',
            'total_amount': 'Общая сумма транзакций (тенге)',
            'avg_amount': 'Средняя сумма транзакции (тенге)',
            'preferred_hour': 'Предпочитаемый час для транзакций',
            'preferred_day': 'Предпочитаемый день недели',
            'unique_merchants': 'Количество уникальных мерчантов',
            'unique_categories': 'Количество уникальных категорий',
            'preferred_pos_mode': 'Предпочитаемый способ оплаты',
            'preferred_wallet': 'Предпочитаемый кошелек',
            'activity_level': 'Уровень активности',
            'spending_level': 'Уровень трат'
        },
        'file_info': {
            'format': 'Apache Parquet',
            'compression': 'Snappy',
            'engine': 'PyArrow',
            'file_size_mb': round(os.path.getsize(output_path) / (1024 * 1024), 2),
            'created_at': datetime.now().isoformat()
        }
    }
    
    # Сохраняем метаданные
    metadata_path = 'results/customer_segmentation_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Выводим статистику
    print("\n✅ Файл результатов сегментации успешно создан!")
    print(f"📁 Основной файл: {output_path}")
    print(f"📋 Метаданные: {metadata_path}")
    print(f"📊 Размер файла: {metadata['file_info']['file_size_mb']} MB")
    print(f"👥 Всего клиентов: {metadata['dataset_info']['total_customers']:,}")
    print("\n📈 Распределение по сегментам:")
    for segment_id, segment_info in metadata['segments_summary'].items():
        print(f"   {segment_info['emoji']} {segment_info['name']}: {segment_info['count']:,} ({segment_info['percentage']}%)")
    
    print(f"\n🔍 Структура данных:")
    print(f"   📋 Колонок: {len(segments_df.columns)}")
    print(f"   📊 Строк: {len(segments_df):,}")
    print(f"   💾 Формат: Apache Parquet (Snappy compression)")
    
    return output_path, metadata_path

if __name__ == "__main__":
    try:
        parquet_file, metadata_file = create_segmentation_parquet()
        print(f"\n🎉 Готово! Результаты сегментации сохранены в формате .parquet")
        print(f"📂 Файлы готовы для передачи и анализа")
    except Exception as e:
        print(f"❌ Ошибка при создании файла: {e}")
        print("💡 Убедитесь, что установлена библиотека pyarrow:")
        print("   pip install pyarrow") 