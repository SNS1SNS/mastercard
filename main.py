"""
Главный модуль для анализа сегментации банковских клиентов
Проект: Сегментация клиентов на основе поведенческих характеристик
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI

import pandas as pd
import numpy as np
from datetime import datetime
import os


from config import config
from data_processor import DataProcessor
from clustering_models import ClusteringModels
from segment_analyzer import SegmentAnalyzer

def print_project_header():
    """Печать заголовка проекта"""
    print("="*80)
    print("🏦 АНАЛИЗ СЕГМЕНТАЦИИ БАНКОВСКИХ КЛИЕНТОВ")
    print("="*80)
    print("📊 Проект: Сегментация клиентов на основе поведенческих характеристик")
    print("🎯 Цель: Выявление групп клиентов с похожим поведением для персонализации")
    print("🔬 Подход: Машинное обучение без учителя (кластеризация)")
    print("📅 Дата запуска:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)

def main():
    """Основная функция для выполнения полного анализа"""
    
    
    print_project_header()
    
    try:
        
        
        
        print("\n🔍 ЭТАП 1: ЗАГРУЗКА И ИССЛЕДОВАНИЕ ДАННЫХ")
        print("-" * 50)
        
        
        data_processor = DataProcessor(config.data.data_file)
        
        
        data_processor.explore_data()
        
        
        
        
        print("\n🛠️ ЭТАП 2: СОЗДАНИЕ ПОВЕДЕНЧЕСКИХ ПРИЗНАКОВ")
        print("-" * 50)
        
        
        client_features = data_processor.create_behavioral_features()
        
        print(f"✅ Создано признаков: {len(client_features.columns)}")
        print(f"✅ Количество клиентов: {len(client_features):,}")
        
        
        
        
        print("\n⚙️ ЭТАП 3: ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
        print("-" * 50)
        
        
        scaled_features, clean_indices = data_processor.prepare_features_for_clustering()
        
        print(f"✅ Данные очищены от выбросов: {len(clean_indices):,} клиентов")
        print(f"✅ Признаки масштабированы: {scaled_features.shape[1]} признаков")
        
        
        feature_importance = data_processor.get_feature_importance_analysis()
        print("\n📊 Топ-10 наиболее вариативных признаков:")
        top_features = feature_importance.nlargest(10, 'coefficient_of_variation')
        for _, row in top_features.iterrows():
            print(f"   • {row['feature']}: {row['coefficient_of_variation']:.3f}")
        
        
        data_processor.save_feature_analysis()
        
        
        
        
        print("\n🤖 ЭТАП 4: КЛАСТЕРИЗАЦИЯ КЛИЕНТОВ")
        print("-" * 50)
        
        
        clustering_models = ClusteringModels(scaled_features, client_features.columns.tolist())
        
        
        print("\n🔍 Поиск оптимального количества кластеров...")
        optimal_k = clustering_models.find_optimal_clusters_kmeans()
        
        
        print("\n🎯 Обучение моделей кластеризации...")
        
        
        clustering_models.fit_kmeans(n_clusters=optimal_k)
        
        
        clustering_models.fit_dbscan()
        
        
        clustering_models.fit_gaussian_mixture(n_components=optimal_k)
        
        
        print("\n📊 Сравнение моделей кластеризации:")
        comparison_results = clustering_models.compare_models()
        print(comparison_results)
        
        
        best_model_name = comparison_results.loc[comparison_results['Силуэтный коэффициент'].idxmax(), 'Модель']
        
        
        model_name_mapping = {
            'Kmeans': 'kmeans',
            'Dbscan': 'dbscan', 
            'Gaussian Mixture': 'gaussian_mixture',
            'Hierarchical': 'hierarchical'
        }
        best_model_key = model_name_mapping.get(best_model_name, best_model_name.lower())
        best_labels = clustering_models.results[best_model_key]['labels']
        
        print(f"\n🏆 Лучшая модель: {best_model_name}")
        
        
        clustering_models.explain_model_choice(best_model_key)
        
        
        
        
        print("\n📈 ЭТАП 5: АНАЛИЗ СЕГМЕНТОВ")
        print("-" * 50)
        
        
        segment_analyzer = SegmentAnalyzer(
            client_features=client_features,
            cluster_labels=best_labels,
            scaled_features=scaled_features,
            clean_indices=clean_indices
        )
        
        
        segment_profiles = segment_analyzer.analyze_segments()
        
        
        
        
        print("\n🎨 ЭТАП 6: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("-" * 50)
        
        
        segment_analyzer.create_comprehensive_visualizations()
        
        
        
        
        print("\n💼 ЭТАП 7: ГЕНЕРАЦИЯ БИЗНЕС-РЕКОМЕНДАЦИЙ")
        print("-" * 50)
        
        
        business_recommendations = segment_analyzer.generate_business_recommendations()
        
        
        
        
        print("\n💾 ЭТАП 8: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 50)
        
        
        segment_analyzer.save_detailed_analysis()
        
        
        
        
        print("\n📋 ЭТАП 9: ИТОГОВОЕ РЕЗЮМЕ")
        print("-" * 50)
        
        
        segment_analyzer.print_executive_summary()
        
        
        
        
        print("\n" + "="*80)
        print("🎉 АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("="*80)
        
        print(f"📁 Все результаты сохранены в папке: {config.data.output_dir}")
        print("\n📊 Созданные файлы:")
        
        
        output_files = [
            "client_segments.csv - Результаты сегментации клиентов",
            "segment_summary.csv - Сводная статистика по сегментам", 
            "segment_profiles.json - Детальные профили сегментов",
            "business_recommendations.json - Бизнес-рекомендации",
            "feature_analysis.csv - Анализ важности признаков",
            "segment_dashboard.html - Интерактивный дашборд",
            "segment_analysis_static.png - Статичные графики",
            "pca_visualization.png - PCA визуализация",
            "segment_heatmap.png - Тепловая карта характеристик",
            "cluster_optimization.png - График оптимизации кластеров"
        ]
        
        for file_desc in output_files:
            print(f"   • {file_desc}")
        
        print("\n💡 Рекомендации по использованию результатов:")
        print("   1. Изучите интерактивный дашборд для общего понимания сегментов")
        print("   2. Используйте бизнес-рекомендации для планирования маркетинговых кампаний")
        print("   3. Регулярно обновляйте сегментацию с новыми данными")
        print("   4. Мониторьте KPI для оценки эффективности стратегий")
        
        print("\n🚀 Следующие шаги:")
        print("   • Внедрение персонализированных предложений")
        print("   • A/B тестирование стратегий для каждого сегмента")
        print("   • Автоматизация процесса сегментации")
        print("   • Интеграция с CRM системой банка")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ АНАЛИЗА:")
        print(f"   {str(e)}")
        print("\n🔧 Рекомендации по устранению:")
        print("   1. Проверьте наличие файла данных")
        print("   2. Убедитесь в корректности путей")
        print("   3. Проверьте установку всех зависимостей")
        return False

def run_quick_analysis():
    """Быстрый анализ для демонстрации"""
    print("\n⚡ РЕЖИМ БЫСТРОГО АНАЛИЗА")
    print("-" * 30)
    
    try:
        
        data_processor = DataProcessor(config.data.data_file)
        
        
        client_features = data_processor.create_behavioral_features()
        
        
        sample_size = min(10000, len(client_features))
        sample_features = client_features.sample(n=sample_size, random_state=42)
        
        print(f"📊 Анализируем выборку: {sample_size:,} клиентов")
        
        
        scaled_features, clean_indices = data_processor.prepare_features_for_clustering()
        
        
        scaled_sample = scaled_features[:sample_size]
        clean_sample = clean_indices[:sample_size]
        
        
        clustering_models = ClusteringModels(scaled_sample, sample_features.columns.tolist())
        clustering_models.fit_kmeans(n_clusters=5)  
        
        
        segment_analyzer = SegmentAnalyzer(
            client_features=sample_features,
            cluster_labels=clustering_models.results['kmeans']['labels'],
            scaled_features=scaled_sample,
            clean_indices=clean_sample
        )
        
        
        segment_analyzer.analyze_segments()
        
        
        segment_analyzer.print_executive_summary()
        
        print("\n✅ Быстрый анализ завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка в быстром анализе: {str(e)}")

if __name__ == "__main__":
    import sys
    
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_analysis()
    else:
        main() 