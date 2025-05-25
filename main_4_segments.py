#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный модуль для анализа сегментации клиентов с 4 сегментами
DECENTRATHON 3.0 | Mastercard Challenge | 2025

Цель: Создание 4 четких сегментов клиентов:
1. 🔄 Обычные клиенты - стандартная активность
2. 😴 Спящие клиенты - низкая активность, требуют реактивации  
3. 💎 VIP клиенты - высокие траты, премиум сегмент
4. ✈️ Путешественники - высокая мобильность, активность в выходные
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from clustering_models import ClusteringModels
from segment_analyzer import SegmentAnalyzer
from config import config

def main():
    """Основная функция анализа с 4 сегментами"""
    
    print("🚀 ЗАПУСК АНАЛИЗА СЕГМЕНТАЦИИ КЛИЕНТОВ (4 СЕГМЕНТА)")
    print("="*60)
    print("🎯 Цель: Создание 4 четких сегментов клиентов")
    print("   🔄 Обычные клиенты")
    print("   😴 Спящие клиенты") 
    print("   💎 VIP клиенты")
    print("   ✈️ Путешественники")
    print("="*60)
    
    try:
        # ЭТАП 1: Загрузка и исследование данных
        print("\n📊 ЭТАП 1: ЗАГРУЗКА И ИССЛЕДОВАНИЕ ДАННЫХ")
        print("-" * 50)
        
        processor = DataProcessor(config.data.data_file)
        data_info = processor.explore_data()
        
        print(f"✅ Загружено {data_info['total_records']:,} транзакций")
        print(f"   👥 Уникальных клиентов: {data_info['unique_clients']:,}")
        print(f"   📅 Период: {data_info['date_range']['start']} - {data_info['date_range']['end']}")
        
        # ЭТАП 2: Создание поведенческих признаков с новыми VIP и путешественническими метриками
        print("\n🛠️ ЭТАП 2: СОЗДАНИЕ ПОВЕДЕНЧЕСКИХ ПРИЗНАКОВ")
        print("-" * 50)
        
        client_features = processor.create_behavioral_features()
        
        # ЭТАП 3: Подготовка данных для кластеризации
        print("\n⚙️ ЭТАП 3: ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
        print("-" * 50)
        
        scaled_features, clean_indices = processor.prepare_features_for_clustering()
        
        # ЭТАП 4: Кластеризация клиентов с принудительным созданием 4 кластеров
        print("\n🤖 ЭТАП 4: КЛАСТЕРИЗАЦИЯ КЛИЕНТОВ (4 КЛАСТЕРА)")
        print("-" * 50)
        
        clustering = ClusteringModels(scaled_features, config.features.clustering_features)
        
        # Принудительно используем 4 кластера
        print(f"🎯 Создаем ровно {config.model.target_clusters} кластера для четкой сегментации")
        kmeans_result = clustering.fit_kmeans(n_clusters=config.model.target_clusters)
        
        print(f"\n✅ Кластеризация завершена:")
        print(f"   📊 Силуэтный коэффициент: {kmeans_result['metrics']['silhouette']:.3f}")
        print(f"   🎯 Количество кластеров: {kmeans_result['n_clusters']}")
        print(f"   📈 Инерция: {kmeans_result['inertia']:,.0f}")
        
        # ЭТАП 5: Анализ сегментов
        print("\n📈 ЭТАП 5: АНАЛИЗ СЕГМЕНТОВ")
        print("-" * 50)
        
        analyzer = SegmentAnalyzer(
            client_features=client_features,
            cluster_labels=kmeans_result['labels'],
            scaled_features=scaled_features,
            clean_indices=clean_indices
        )
        
        segment_analysis = analyzer.analyze_segments()
        
        # ЭТАП 6: Создание визуализаций
        print("\n🎨 ЭТАП 6: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("-" * 50)
        
        analyzer.create_comprehensive_visualizations()
        
        # ЭТАП 7: Генерация бизнес-рекомендаций
        print("\n💼 ЭТАП 7: ГЕНЕРАЦИЯ БИЗНЕС-РЕКОМЕНДАЦИЙ")
        print("-" * 50)
        
        business_recommendations = analyzer.generate_business_recommendations()
        
        # ЭТАП 8: Сохранение результатов
        print("\n💾 ЭТАП 8: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 50)
        
        analyzer.save_detailed_analysis()
        processor.save_feature_analysis()
        
        # Создание дополнительных аналитических файлов
        analyzer.analyze_segment_stability()
        analyzer.calculate_customer_lifetime_value()
        analyzer.analyze_customer_journey()
        analyzer.create_monitoring_dashboard_data()
        analyzer.design_ab_test_framework()
        
        # ЭТАП 9: Итоговое резюме
        print("\n📋 ЭТАП 9: ИТОГОВОЕ РЕЗЮМЕ")
        print("-" * 50)
        
        analyzer.print_executive_summary()
        
        # Детальная информация о сегментах
        print("\n🎯 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О СЕГМЕНТАХ:")
        print("="*60)
        
        for cluster_id, profile in segment_analysis.items():
            segment_type = profile['segment_type']
            print(f"\n{segment_type['emoji']} {segment_type['name']} (Кластер {cluster_id}):")
            print(f"   📊 Размер: {profile['size']:,} клиентов ({profile['percentage']:.1f}%)")
            print(f"   📝 Описание: {segment_type['description']}")
            print(f"   ⭐ Приоритет: {segment_type['priority']}")
            print(f"   💰 Ценность: {segment_type['value']}")
            
            # Ключевые характеристики
            chars = profile['characteristics']
            print(f"   📈 Средние транзакции: {chars['avg_transactions']:.0f}")
            print(f"   💵 Средний оборот: {chars['avg_total_amount']:,.0f} ₸")
            print(f"   🛒 Средний чек: {chars['avg_amount']:,.0f} ₸")
            print(f"   🏪 Уникальных мерчантов: {chars['avg_merchants']:.0f}")
        
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*60)
        print("📁 Результаты сохранены в папке 'results/'")
        print("🌐 Откройте 'results/segment_dashboard.html' для интерактивного просмотра")
        print("📊 Бизнес-рекомендации: 'results/business_recommendations.json'")
        print("📈 Профили сегментов: 'results/segment_profiles.json'")
        
        return {
            'segments': segment_analysis,
            'recommendations': business_recommendations,
            'model_metrics': kmeans_result['metrics']
        }
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ АНАЛИЗА: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🏦 СИСТЕМА СЕГМЕНТАЦИИ БАНКОВСКИХ КЛИЕНТОВ")
    print("🎯 Версия: 4 сегмента (Обычные, Спящие, VIP, Путешественники)")
    print("🏆 DECENTRATHON 3.0 | Mastercard Challenge | 2025")
    print()
    
    results = main()
    
    if results:
        print("\n✅ Система готова к использованию!")
        print("💡 Для демонстрации возможностей запустите: python demo.py")
    else:
        print("\n❌ Анализ завершился с ошибками") 