#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрация работы с файлом результатов сегментации в формате .parquet
Показывает основные способы анализа данных
"""

import pandas as pd
import json

def demo_parquet_analysis():
    """Демонстрирует анализ данных из .parquet файла"""
    
    print("🔍 ДЕМОНСТРАЦИЯ РАБОТЫ С РЕЗУЛЬТАТАМИ СЕГМЕНТАЦИИ (.parquet)")
    print("=" * 60)
    
    # Загрузка данных
    print("\n📊 1. ЗАГРУЗКА ДАННЫХ")
    df = pd.read_parquet('results/customer_segmentation_results.parquet')
    print(f"✅ Загружено {len(df):,} записей с {len(df.columns)} колонками")
    
    # Загрузка метаданных
    with open('results/customer_segmentation_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📅 Дата анализа: {metadata['dataset_info']['analysis_date']}")
    print(f"🤖 Модель: {metadata['dataset_info']['model_used']}")
    print(f"📈 Качество: {metadata['dataset_info']['silhouette_score']}")
    
    # Основная информация
    print("\n📋 2. СТРУКТУРА ДАННЫХ")
    print(f"Размер файла: {metadata['file_info']['file_size_mb']} MB")
    print(f"Формат: {metadata['file_info']['format']}")
    print(f"Сжатие: {metadata['file_info']['compression']}")
    
    # Просмотр первых записей
    print("\n👀 3. ПЕРВЫЕ 3 ЗАПИСИ")
    display_columns = ['card_id', 'segment_name', 'total_transactions', 'total_amount', 'avg_amount']
    print(df[display_columns].head(3).to_string(index=False))
    
    # Анализ по сегментам
    print("\n📊 4. АНАЛИЗ ПО СЕГМЕНТАМ")
    segment_analysis = df.groupby('segment_name').agg({
        'card_id': 'count',
        'total_transactions': ['mean', 'median'],
        'total_amount': ['mean', 'median'],
        'avg_amount': ['mean', 'median'],
        'unique_merchants': 'mean',
        'unique_categories': 'mean'
    }).round(2)
    
    print(segment_analysis)
    
    # Распределение по категориям
    print("\n📈 5. РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ")
    
    print("\n🔸 По объему транзакций:")
    transaction_dist = df['transaction_volume_category'].value_counts()
    for category, count in transaction_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")
    
    print("\n🔸 По доходности:")
    revenue_dist = df['revenue_category'].value_counts()
    for category, count in revenue_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")
    
    print("\n🔸 По среднему чеку:")
    ticket_dist = df['avg_ticket_category'].value_counts()
    for category, count in ticket_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")
    
    # Временные предпочтения
    print("\n🕐 6. ВРЕМЕННЫЕ ПРЕДПОЧТЕНИЯ")
    
    print("\n🔸 Популярные часы для транзакций:")
    hour_dist = df['preferred_hour'].value_counts().head(5)
    for hour, count in hour_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {hour:02d}:00 - {count:,} клиентов ({percentage:.1f}%)")
    
    print("\n🔸 Популярные дни недели:")
    day_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    day_dist = df['preferred_day'].value_counts()
    for day, count in day_dist.items():
        percentage = (count / len(df)) * 100
        day_name = day_names[day] if day < len(day_names) else f"День {day}"
        print(f"   {day_name}: {count:,} клиентов ({percentage:.1f}%)")
    
    # Способы оплаты
    print("\n💳 7. ПРЕДПОЧИТАЕМЫЕ СПОСОБЫ ОПЛАТЫ")
    
    print("\n🔸 POS режимы:")
    pos_dist = df['preferred_pos_mode'].value_counts().head(5)
    for pos_mode, count in pos_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {pos_mode}: {count:,} ({percentage:.1f}%)")
    
    print("\n🔸 Кошельки:")
    wallet_dist = df['preferred_wallet'].value_counts().head(5)
    for wallet, count in wallet_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {wallet}: {count:,} ({percentage:.1f}%)")
    
    # Топ клиенты по сегментам
    print("\n🏆 8. ТОП-3 КЛИЕНТА ПО СЕГМЕНТАМ")
    
    for segment in df['segment_name'].unique():
        segment_data = df[df['segment_name'] == segment]
        top_clients = segment_data.nlargest(3, 'total_amount')[['card_id', 'total_transactions', 'total_amount', 'avg_amount']]
        
        print(f"\n🔸 {segment}:")
        for _, client in top_clients.iterrows():
            print(f"   ID: {client['card_id']} | "
                  f"Транзакций: {client['total_transactions']:,} | "
                  f"Оборот: {client['total_amount']:,.0f}₸ | "
                  f"Средний чек: {client['avg_amount']:,.0f}₸")
    
    # Сводка
    print("\n📋 9. КРАТКАЯ СВОДКА")
    print(f"✅ Всего проанализировано: {len(df):,} клиентов")
    print(f"✅ Выявлено сегментов: {df['segment_name'].nunique()}")
    print(f"✅ Общий оборот: {df['total_amount'].sum():,.0f} тенге")
    print(f"✅ Средний оборот на клиента: {df['total_amount'].mean():,.0f} тенге")
    print(f"✅ Средний чек: {df['avg_amount'].mean():,.0f} тенге")
    
    print("\n🎯 ГОТОВО! Данные успешно проанализированы из .parquet файла")
    print("📂 Файл готов для использования в любых аналитических системах")

if __name__ == "__main__":
    try:
        demo_parquet_analysis()
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        print("💡 Убедитесь, что файл customer_segmentation_results.parquet существует") 