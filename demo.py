"""
Демонстрационный скрипт для быстрого тестирования функциональности
сегментации клиентов банка
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import os

def test_imports():
    """Тестирование импорта всех модулей"""
    print("🔍 Тестирование импорта модулей...")
    
    try:
        from config import config
        print("   ✅ config.py - OK")
        
        from data_processor import DataProcessor
        print("   ✅ data_processor.py - OK")
        
        from clustering_models import ClusteringModels
        print("   ✅ clustering_models.py - OK")
        
        from segment_analyzer import SegmentAnalyzer
        print("   ✅ segment_analyzer.py - OK")
        
        print("✅ Все модули успешно импортированы!")
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_config():
    """Тестирование конфигурации"""
    print("\n🔧 Тестирование конфигурации...")
    
    try:
        from config import config
        
        print(f"   📁 Файл данных: {config.data.data_file}")
        print(f"   📁 Папка результатов: {config.data.output_dir}")
        print(f"   🎯 Признаки для кластеризации: {len(config.features.clustering_features) if config.features.clustering_features else 'Автоматически'}")
        print(f"   🤖 Алгоритм по умолчанию: K-Means")
        print(f"   📊 Стиль визуализации: {config.visualization.style}")
        
        
        if not os.path.exists(config.data.output_dir):
            os.makedirs(config.data.output_dir)
            print(f"   ✅ Создана папка результатов: {config.data.output_dir}")
        else:
            print(f"   ✅ Папка результатов существует: {config.data.output_dir}")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return False

def test_data_loading():
    """Тестирование загрузки данных"""
    print("\n📊 Тестирование загрузки данных...")
    
    try:
        from config import config
        from data_processor import DataProcessor
        
        
        if not os.path.exists(config.data.data_file):
            print(f"❌ Файл данных не найден: {config.data.data_file}")
            return False
        
        
        processor = DataProcessor(config.data.data_file)
        
        print(f"   ✅ Данные успешно загружены")
        print(f"   📈 Количество транзакций: {len(processor.data):,}")
        print(f"   👥 Количество уникальных клиентов: {processor.data['card_id'].nunique():,}")
        print(f"   🏪 Количество мерчантов: {processor.data['merchant_id'].nunique():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return False

def test_feature_creation():
    """Тестирование создания признаков"""
    print("\n🛠️ Тестирование создания признаков...")
    
    try:
        from config import config
        from data_processor import DataProcessor
        
        processor = DataProcessor(config.data.data_file)
        
        
        sample_data = processor.data.sample(n=min(50000, len(processor.data)), random_state=42)
        processor.data = sample_data
        
        
        features = processor.create_behavioral_features()
        
        print(f"   ✅ Признаки созданы успешно")
        print(f"   📊 Количество клиентов: {len(features):,}")
        print(f"   🔢 Количество признаков: {len(features.columns)}")
        print(f"   📋 Основные признаки: {list(features.columns[:5])}")
        
        
        missing_values = features.isnull().sum().sum()
        print(f"   🔍 Пропущенные значения: {missing_values}")
        
        return True, features
        
    except Exception as e:
        print(f"❌ Ошибка создания признаков: {e}")
        return False, None

def test_clustering():
    """Тестирование кластеризации"""
    print("\n🤖 Тестирование кластеризации...")
    
    try:
        from config import config
        from data_processor import DataProcessor
        from clustering_models import ClusteringModels
        
        processor = DataProcessor(config.data.data_file)
        
        
        sample_data = processor.data.sample(n=min(10000, len(processor.data)), random_state=42)
        processor.data = sample_data
        
        
        features = processor.create_behavioral_features()
        
        
        scaled_features, clean_indices = processor.prepare_features_for_clustering()
        
        print(f"   ✅ Данные подготовлены")
        print(f"   📊 Размер выборки: {scaled_features.shape}")
        print(f"   🧹 Очищено от выбросов: {len(clean_indices)} клиентов")
        
        
        clustering = ClusteringModels(scaled_features, features.columns.tolist())
        
        
        clustering.fit_kmeans(n_clusters=5)
        
        print(f"   ✅ K-Means кластеризация выполнена")
        print(f"   🎯 Количество кластеров: 5")
        
        
        labels = clustering.results['kmeans']['labels']
        unique_labels = np.unique(labels)
        
        print(f"   📈 Найдено кластеров: {len(unique_labels)}")
        print(f"   📊 Распределение по кластерам:")
        
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = (count / len(labels)) * 100
            print(f"      Кластер {label}: {count} клиентов ({percentage:.1f}%)")
        
        return True, clustering, features, scaled_features, clean_indices
        
    except Exception as e:
        print(f"❌ Ошибка кластеризации: {e}")
        return False, None, None, None, None

def test_analysis():
    """Тестирование анализа сегментов"""
    print("\n📈 Тестирование анализа сегментов...")
    
    try:
        
        success, clustering, features, scaled_features, clean_indices = test_clustering()
        
        if not success:
            return False
        
        from segment_analyzer import SegmentAnalyzer
        
        
        analyzer = SegmentAnalyzer(
            client_features=features,
            cluster_labels=clustering.results['kmeans']['labels'],
            scaled_features=scaled_features,
            clean_indices=clean_indices
        )
        
        
        segment_profiles = analyzer.analyze_segments()
        
        print(f"   ✅ Анализ сегментов выполнен")
        print(f"   🎯 Количество сегментов: {len(segment_profiles)}")
        
        
        for cluster_id, profile in segment_profiles.items():
            segment_type = profile['segment_type']
            print(f"      {segment_type['name']}: {profile['percentage']:.1f}% клиентов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка анализа сегментов: {e}")
        return False

def demo_advanced_analytics():
    """Демонстрация расширенной аналитики"""
    print("\n🔬 Демонстрация расширенной аналитики...")
    
    try:
        from config import config
        from data_processor import DataProcessor
        from clustering_models import ClusteringModels
        from segment_analyzer import SegmentAnalyzer
        
        
        processor = DataProcessor(config.data.data_file)
        sample_data = processor.data.sample(n=min(5000, len(processor.data)), random_state=42)
        processor.data = sample_data
        
        print(f"   📊 Загружена выборка: {len(sample_data)} записей")
        
        
        features = processor.create_behavioral_features()
        scaled_features, clean_indices = processor.prepare_features_for_clustering()
        
        
        clustering = ClusteringModels(scaled_features, features.columns.tolist())
        best_result = clustering.fit_kmeans(n_clusters=3)  
        
        
        analyzer = SegmentAnalyzer(
            client_features=features,
            cluster_labels=best_result['labels'],
            scaled_features=scaled_features,
            clean_indices=clean_indices
        )
        
        
        analyzer.analyze_segments()
        analyzer.generate_business_recommendations()
        
        print("\n🔬 Демонстрация новых возможностей:")
        
        
        print("\n📊 1. Анализ стабильности сегментов:")
        stability = analyzer.analyze_segment_stability()
        for cluster_id, metrics in stability.items():
            print(f"   Сегмент {cluster_id}: Стабильность {metrics['consistency_score']:.2f}, Риск оттока {metrics['churn_risk']:.1%}")
        
        
        print("\n💰 2. Customer Lifetime Value:")
        clv_data = analyzer.calculate_customer_lifetime_value()
        for cluster_id, clv in clv_data.items():
            print(f"   Сегмент {cluster_id}: CLV {clv['adjusted_clv']:,.0f} тенге, Уровень: {clv['value_tier']}")
        
        
        print("\n🛤️ 3. Анализ жизненного цикла:")
        journey = analyzer.analyze_customer_journey()
        for cluster_id, data in journey.items():
            print(f"   Сегмент {cluster_id}: {data['lifecycle_stage']}, Вовлеченность: {data['engagement_level']}")
        
        
        print("\n📊 4. Система мониторинга:")
        monitoring = analyzer.create_monitoring_dashboard_data()
        print(f"   Создано алертов: {len(monitoring['alerts'])}")
        for alert in monitoring['alerts'][:2]:  
            print(f"   - {alert['severity']}: {alert['message']}")
        
        
        print("\n🧪 5. План A/B тестов:")
        ab_tests = analyzer.design_ab_test_framework()
        for cluster_id, test_data in ab_tests.items():
            print(f"   Сегмент {cluster_id}: {len(test_data['test_scenarios'])} тестов, "
                  f"Длительность: {test_data['test_duration_weeks']} недель")
        
        
        print("\n📋 6. Автоматические отчеты:")
        report = analyzer.generate_executive_report()
        presentation = analyzer.create_presentation_slides()
        print(f"   Исполнительный отчет: {len(report)} символов")
        print(f"   Слайдов презентации: {len(presentation)} разделов")
        
        print("\n✅ Демонстрация расширенной аналитики завершена!")
        print("📁 Все результаты сохранены в папке 'results/'")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации расширенной аналитики: {e}")
        return False

def run_demo():
    """Запуск полной демонстрации"""
    print("="*60)
    print("🚀 ДЕМОНСТРАЦИЯ СИСТЕМЫ СЕГМЕНТАЦИИ КЛИЕНТОВ")
    print("="*60)
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    tests = [
        ("Импорт модулей", test_imports),
        ("Конфигурация", test_config),
        ("Загрузка данных", test_data_loading),
        ("Создание признаков", lambda: test_feature_creation()[0]),
        ("Кластеризация", lambda: test_clustering()[0]),
        ("Анализ сегментов", test_analysis),
        ("Расширенная аналитика", demo_advanced_analytics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - УСПЕШНО")
            else:
                print(f"❌ {test_name} - ОШИБКА")
                break  
                
        except Exception as e:
            print(f"❌ {test_name} - КРИТИЧЕСКАЯ ОШИБКА: {e}")
            results.append((test_name, False))
            break
    
    
    print("\n" + "="*60)
    print("📋 ИТОГОВЫЙ ОТЧЕТ ДЕМОНСТРАЦИИ")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"✅ Пройдено тестов: {passed}/{total}")
    print(f"📊 Процент успеха: {(passed/total)*100:.1f}%")
    
    print("\n📋 Детальные результаты:")
    for test_name, result in results:
        status = "✅ УСПЕШНО" if result else "❌ ОШИБКА"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("🚀 Система готова к использованию!")
        print("\n💡 Для запуска полного анализа используйте:")
        print("   python main.py")
        print("\n💡 Для быстрого анализа используйте:")
        print("   python main.py --quick")
    else:
        print("\n⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        print("🔧 Рекомендации по устранению:")
        print("   1. Проверьте установку всех зависимостей: pip install -r requirements.txt")
        print("   2. Убедитесь в наличии файла данных: DECENTRATHON_3.0.parquet")
        print("   3. Проверьте права доступа к файлам и папкам")
    
    print("="*60)

if __name__ == "__main__":
    run_demo() 