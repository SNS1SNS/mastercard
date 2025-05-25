"""
Демонстрационный скрипт системы сегментации банковских клиентов
Показывает основные возможности и результаты анализа
"""

import pandas as pd
import json
import os
from datetime import datetime

def print_header(title):
    """Красивый заголовок"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    """Заголовок секции"""
    print(f"\n📊 {title}")
    print("-" * 40)

def load_segmentation_results():
    """Загрузка результатов сегментации"""
    print_header("ЗАГРУЗКА РЕЗУЛЬТАТОВ СЕГМЕНТАЦИИ")
    
    # Проверяем наличие файлов
    files_to_check = [
        'results/client_segments.csv',
        'results/segment_profiles.json',
        'results/business_recommendations.json',
        'results/cluster_summary_statistics.csv'
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствуют файлы результатов:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Запустите сначала: python main_4_segments.py")
        return None, None, None, None
    
    # Загружаем данные
    print("✅ Загружаем результаты анализа...")
    
    # Основные результаты сегментации
    df_segments = pd.read_csv('results/client_segments.csv')
    print(f"   📊 Загружено {len(df_segments):,} клиентов")
    
    # Профили сегментов
    with open('results/segment_profiles.json', 'r', encoding='utf-8') as f:
        segment_profiles = json.load(f)
    print(f"   📋 Загружено {len(segment_profiles)} профилей сегментов")
    
    # Бизнес-рекомендации
    with open('results/business_recommendations.json', 'r', encoding='utf-8') as f:
        recommendations = json.load(f)
    print(f"   💼 Загружено рекомендаций для {len(recommendations)} сегментов")
    
    # Статистика кластеров
    df_stats = pd.read_csv('results/cluster_summary_statistics.csv')
    print(f"   📈 Загружена статистика по {len(df_stats)} кластерам")
    
    return df_segments, segment_profiles, recommendations, df_stats

def show_segment_overview(df_segments, segment_profiles):
    """Обзор сегментов"""
    print_header("ОБЗОР СЕГМЕНТОВ КЛИЕНТОВ")
    
    # Распределение по сегментам
    segment_counts = df_segments['segment_type'].value_counts()
    total_clients = len(df_segments)
    
    print_section("Распределение клиентов по сегментам")
    for segment, count in segment_counts.items():
        percentage = (count / total_clients) * 100
        print(f"   {segment}: {count:,} клиентов ({percentage:.1f}%)")
    
    print_section("Характеристики сегментов")
    for segment_name, profile in segment_profiles.items():
        print(f"\n🎯 {segment_name}")
        print(f"   📊 Размер: {profile['size']:,} клиентов ({profile['percentage']:.1f}%)")
        print(f"   💰 Средний оборот: {profile['avg_total_amount']:,.0f} тенге")
        print(f"   🔄 Средние транзакции: {profile['avg_total_transactions']:.0f}")
        print(f"   💳 Средний чек: {profile['avg_avg_amount']:,.0f} тенге")
        print(f"   🏪 Уникальных мерчантов: {profile['avg_unique_merchants']:.0f}")

def show_financial_analysis(df_segments):
    """Финансовый анализ"""
    print_header("ФИНАНСОВЫЙ АНАЛИЗ")
    
    # Группировка по сегментам
    financial_stats = df_segments.groupby('segment_type').agg({
        'total_amount': ['sum', 'mean', 'median'],
        'total_transactions': ['sum', 'mean'],
        'avg_amount': 'mean',
        'card_id': 'count'
    }).round(0)
    
    print_section("Финансовые показатели по сегментам")
    
    total_revenue = df_segments['total_amount'].sum()
    total_transactions = df_segments['total_transactions'].sum()
    
    print(f"💰 Общий оборот: {total_revenue:,.0f} тенге")
    print(f"🔄 Общие транзакции: {total_transactions:,.0f}")
    print(f"💳 Средний чек по банку: {total_revenue/total_transactions:,.0f} тенге")
    
    print("\n📊 Детализация по сегментам:")
    for segment in df_segments['segment_type'].unique():
        segment_data = df_segments[df_segments['segment_type'] == segment]
        
        segment_revenue = segment_data['total_amount'].sum()
        segment_transactions = segment_data['total_transactions'].sum()
        revenue_share = (segment_revenue / total_revenue) * 100
        
        print(f"\n   {segment}:")
        print(f"      💰 Оборот: {segment_revenue:,.0f} тенге ({revenue_share:.1f}%)")
        print(f"      🔄 Транзакции: {segment_transactions:,.0f}")
        print(f"      💳 Средний чек: {segment_revenue/segment_transactions:,.0f} тенге")
        print(f"      👥 Клиентов: {len(segment_data):,}")

def show_behavioral_patterns(df_segments):
    """Анализ поведенческих паттернов"""
    print_header("ПОВЕДЕНЧЕСКИЕ ПАТТЕРНЫ")
    
    print_section("Временные предпочтения")
    
    # Анализ по часам
    hour_analysis = df_segments.groupby('segment_type')['preferred_hour'].agg(['mean', 'mode']).round(1)
    print("⏰ Предпочитаемые часы для транзакций:")
    for segment in hour_analysis.index:
        mean_hour = hour_analysis.loc[segment, 'mean']
        print(f"   {segment}: {mean_hour:.1f} час")
    
    # Анализ по дням недели
    day_analysis = df_segments.groupby('segment_type')['preferred_day'].agg(['mean', 'mode']).round(1)
    days_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    
    print("\n📅 Предпочитаемые дни недели:")
    for segment in day_analysis.index:
        mean_day = int(day_analysis.loc[segment, 'mean'])
        day_name = days_names[mean_day] if mean_day < 7 else "Неопределен"
        print(f"   {segment}: {day_name}")
    
    print_section("Разнообразие активности")
    
    diversity_stats = df_segments.groupby('segment_type').agg({
        'unique_merchants': 'mean',
        'unique_categories': 'mean',
        'unique_cities': 'mean'
    }).round(1)
    
    for segment in diversity_stats.index:
        print(f"\n   {segment}:")
        print(f"      🏪 Мерчантов: {diversity_stats.loc[segment, 'unique_merchants']:.0f}")
        print(f"      🏷️ Категорий: {diversity_stats.loc[segment, 'unique_categories']:.0f}")
        print(f"      🌍 Городов: {diversity_stats.loc[segment, 'unique_cities']:.0f}")

def show_business_recommendations(recommendations):
    """Показ бизнес-рекомендаций"""
    print_header("БИЗНЕС-РЕКОМЕНДАЦИИ")
    
    for segment_name, rec_data in recommendations.items():
        print(f"\n🎯 {segment_name}")
        print(f"   📊 Приоритет: {rec_data['priority']}")
        print(f"   🎯 Стратегия: {rec_data['strategy']}")
        
        print("   💡 Рекомендации:")
        for i, recommendation in enumerate(rec_data['recommendations'], 1):
            print(f"      {i}. {recommendation}")
        
        if 'kpis' in rec_data:
            print("   📈 Ключевые метрики:")
            for kpi, value in rec_data['kpis'].items():
                print(f"      • {kpi}: {value}")

def show_model_quality():
    """Показ качества модели"""
    print_header("КАЧЕСТВО МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ")
    
    # Загружаем метаданные если есть
    metadata_file = 'results/customer_segmentation_metadata.json'
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print_section("Метрики качества кластеризации")
        
        if 'model_metrics' in metadata:
            metrics = metadata['model_metrics']
            print(f"   🎯 Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
            print(f"   📊 Calinski-Harabasz Index: {metrics.get('calinski_harabasz_score', 'N/A')}")
            print(f"   📈 Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A')}")
        
        if 'dataset_info' in metadata:
            dataset = metadata['dataset_info']
            print_section("Информация о датасете")
            print(f"   👥 Клиентов: {dataset.get('total_clients', 'N/A'):,}")
            print(f"   🔄 Транзакций: {dataset.get('total_transactions', 'N/A'):,}")
            print(f"   📅 Период: {dataset.get('date_range', 'N/A')}")
    else:
        print("⚠️ Файл метаданных не найден")

def show_files_created():
    """Показ созданных файлов"""
    print_header("СОЗДАННЫЕ ФАЙЛЫ И РЕЗУЛЬТАТЫ")
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("❌ Папка results не найдена")
        return
    
    files = os.listdir(results_dir)
    
    # Группируем файлы по типам
    file_groups = {
        'Данные сегментации': [f for f in files if f.endswith('.csv') and 'segment' in f],
        'Профили и рекомендации': [f for f in files if f.endswith('.json')],
        'Визуализации': [f for f in files if f.endswith(('.png', '.html'))],
        'Отчеты и презентации': [f for f in files if f.endswith(('.pdf', '.pptx', '.md'))],
        'Словари данных': [f for f in files if 'dictionary' in f]
    }
    
    for group_name, group_files in file_groups.items():
        if group_files:
            print_section(group_name)
            for file in sorted(group_files):
                file_path = os.path.join(results_dir, file)
                file_size = os.path.getsize(file_path)
                
                # Форматируем размер файла
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024*1024:
                    size_str = f"{file_size/1024:.1f} KB"
                else:
                    size_str = f"{file_size/(1024*1024):.1f} MB"
                
                print(f"   📄 {file} ({size_str})")

def show_next_steps():
    """Следующие шаги"""
    print_header("СЛЕДУЮЩИЕ ШАГИ")
    
    print("🚀 Рекомендуемые действия:")
    print("   1. 📊 Изучите интерактивный дашборд: results/segment_dashboard.html")
    print("   2. 📋 Ознакомьтесь с презентацией: results/presentation_segmentation.pdf")
    print("   3. 📖 Изучите словарь данных: results/data_dictionary.xlsx")
    print("   4. 💼 Внедрите бизнес-рекомендации из results/business_recommendations.json")
    print("   5. 📈 Настройте мониторинг по данным из results/monitoring_data.json")
    print("   6. 🧪 Запустите A/B тесты по планам из results/ab_test_plan.json")
    
    print("\n🔧 Техническая интеграция:")
    print("   • Используйте results/client_segments.csv для CRM системы")
    print("   • Интегрируйте рекомендации в маркетинговые кампании")
    print("   • Настройте автоматическое обновление сегментации")
    
    print("\n📞 Поддержка:")
    print("   • Документация: README.md")
    print("   • Техническая поддержка: команда Data Science")

def main():
    """Основная функция демонстрации"""
    print("🏦 ДЕМОНСТРАЦИЯ СИСТЕМЫ СЕГМЕНТАЦИИ БАНКОВСКИХ КЛИЕНТОВ")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Загружаем данные
    df_segments, segment_profiles, recommendations, df_stats = load_segmentation_results()
    
    if df_segments is None:
        return
    
    # Показываем результаты
    show_segment_overview(df_segments, segment_profiles)
    show_financial_analysis(df_segments)
    show_behavioral_patterns(df_segments)
    show_business_recommendations(recommendations)
    show_model_quality()
    show_files_created()
    show_next_steps()
    
    print("\n" + "="*70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("🎯 Система готова к использованию!")
    print("="*70)

if __name__ == "__main__":
    main() 