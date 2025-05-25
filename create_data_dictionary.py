"""
Скрипт для создания Data Dictionary - описания всех полей и метрик
"""

import pandas as pd
import json
from datetime import datetime

def create_data_dictionary():
    """Создание словаря данных"""
    
    # Исходные поля датасета
    original_fields = {
        'card_id': {
            'type': 'string',
            'description': 'Уникальный идентификатор карты клиента',
            'example': 'CARD_12345',
            'source': 'Исходные данные'
        },
        'transaction_dttm': {
            'type': 'datetime',
            'description': 'Дата и время совершения транзакции',
            'example': '2023-01-01 12:30:45',
            'source': 'Исходные данные'
        },
        'amount_rub': {
            'type': 'float',
            'description': 'Сумма транзакции в рублях',
            'example': '1500.50',
            'source': 'Исходные данные'
        },
        'amount_kzt': {
            'type': 'float',
            'description': 'Сумма транзакции в тенге',
            'example': '7500.25',
            'source': 'Исходные данные'
        },
        'currency': {
            'type': 'string',
            'description': 'Валюта транзакции',
            'example': 'KZT, RUB, USD',
            'source': 'Исходные данные'
        },
        'operation_type': {
            'type': 'string',
            'description': 'Тип операции (покупка, снятие наличных и т.д.)',
            'example': 'PURCHASE, CASH_WITHDRAWAL',
            'source': 'Исходные данные'
        },
        'merchant_id': {
            'type': 'string',
            'description': 'Идентификатор мерчанта',
            'example': 'MERCHANT_789',
            'source': 'Исходные данные'
        },
        'merchant_mcc': {
            'type': 'string',
            'description': 'MCC код мерчанта (категория)',
            'example': '5411 (продуктовые магазины)',
            'source': 'Исходные данные'
        },
        'merchant_city': {
            'type': 'string',
            'description': 'Город мерчанта',
            'example': 'Алматы, Астана',
            'source': 'Исходные данные'
        },
        'original_amount': {
            'type': 'float',
            'description': 'Оригинальная сумма транзакции',
            'example': '100.00',
            'source': 'Исходные данные'
        },
        'pos_entry_mode': {
            'type': 'string',
            'description': 'Способ ввода данных карты',
            'example': 'CHIP, CONTACTLESS, MAGNETIC_STRIPE',
            'source': 'Исходные данные'
        },
        'wallet_type': {
            'type': 'string',
            'description': 'Тип электронного кошелька',
            'example': 'APPLE_PAY, GOOGLE_PAY, SAMSUNG_PAY',
            'source': 'Исходные данные'
        }
    }
    
    # Созданные поведенческие признаки
    behavioral_features = {
        'total_transactions': {
            'type': 'integer',
            'description': 'Общее количество транзакций клиента за период',
            'formula': 'COUNT(transactions)',
            'source': 'Вычисляемый признак'
        },
        'total_amount': {
            'type': 'float',
            'description': 'Общая сумма всех транзакций клиента (тенге)',
            'formula': 'SUM(amount_kzt)',
            'source': 'Вычисляемый признак'
        },
        'avg_amount': {
            'type': 'float',
            'description': 'Средний размер транзакции (тенге)',
            'formula': 'total_amount / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'median_amount': {
            'type': 'float',
            'description': 'Медианный размер транзакции (тенге)',
            'formula': 'MEDIAN(amount_kzt)',
            'source': 'Вычисляемый признак'
        },
        'std_amount': {
            'type': 'float',
            'description': 'Стандартное отклонение сумм транзакций',
            'formula': 'STD(amount_kzt)',
            'source': 'Вычисляемый признак'
        },
        'amount_range': {
            'type': 'float',
            'description': 'Диапазон сумм транзакций (max - min)',
            'formula': 'MAX(amount_kzt) - MIN(amount_kzt)',
            'source': 'Вычисляемый признак'
        },
        'unique_merchants': {
            'type': 'integer',
            'description': 'Количество уникальных мерчантов',
            'formula': 'COUNT(DISTINCT merchant_id)',
            'source': 'Вычисляемый признак'
        },
        'unique_categories': {
            'type': 'integer',
            'description': 'Количество уникальных категорий MCC',
            'formula': 'COUNT(DISTINCT merchant_mcc)',
            'source': 'Вычисляемый признак'
        },
        'unique_cities': {
            'type': 'integer',
            'description': 'Количество уникальных городов',
            'formula': 'COUNT(DISTINCT merchant_city)',
            'source': 'Вычисляемый признак'
        },
        'purchase_ratio': {
            'type': 'float',
            'description': 'Доля покупок от общего числа транзакций',
            'formula': 'COUNT(operation_type=PURCHASE) / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'avg_merchants_per_transaction': {
            'type': 'float',
            'description': 'Среднее количество мерчантов на транзакцию',
            'formula': 'unique_merchants / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'spending_consistency': {
            'type': 'float',
            'description': 'Консистентность трат (обратная к коэф. вариации)',
            'formula': '1 / (std_amount / avg_amount)',
            'source': 'Вычисляемый признак'
        },
        'preferred_hour': {
            'type': 'integer',
            'description': 'Предпочитаемое время совершения транзакций (час)',
            'formula': 'MODE(HOUR(transaction_dttm))',
            'source': 'Вычисляемый признак'
        },
        'preferred_day': {
            'type': 'integer',
            'description': 'Предпочитаемый день недели (0=понедельник)',
            'formula': 'MODE(DAYOFWEEK(transaction_dttm))',
            'source': 'Вычисляемый признак'
        },
        'high_value_transactions_ratio': {
            'type': 'float',
            'description': 'Доля высокоценных транзакций от общего числа',
            'formula': 'COUNT(amount_kzt > PERCENTILE_70) / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'travel_indicator': {
            'type': 'float',
            'description': 'Индикатор путешественника',
            'formula': 'unique_cities / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'premium_merchant_ratio': {
            'type': 'float',
            'description': 'Доля премиум мерчантов от общего числа',
            'formula': 'COUNT(premium_merchants) / unique_merchants',
            'source': 'Вычисляемый признак'
        },
        'weekend_activity_ratio': {
            'type': 'float',
            'description': 'Доля транзакций в выходные',
            'formula': 'COUNT(WEEKEND_transactions) / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'evening_activity_ratio': {
            'type': 'float',
            'description': 'Доля транзакций в вечернее время',
            'formula': 'COUNT(HOUR >= 18) / total_transactions',
            'source': 'Вычисляемый признак'
        },
        'preferred_pos_mode': {
            'type': 'string',
            'description': 'Предпочитаемый способ ввода данных карты',
            'formula': 'MODE(pos_entry_mode)',
            'source': 'Вычисляемый признак'
        },
        'preferred_wallet': {
            'type': 'string',
            'description': 'Предпочитаемый тип электронного кошелька',
            'formula': 'MODE(wallet_type)',
            'source': 'Вычисляемый признак'
        }
    }
    
    # Результаты сегментации
    segmentation_results = {
        'cluster': {
            'type': 'integer',
            'description': 'Номер кластера/сегмента (0, 1, 2, 3)',
            'values': {
                0: 'Активные клиенты (83.4%)',
                1: 'Спящие клиенты (1.1%)',
                2: 'Путешественники (4.8%)',
                3: 'VIP клиенты (10.8%)'
            },
            'source': 'Результат кластеризации K-means'
        },
        'segment_type': {
            'type': 'string',
            'description': 'Название типа сегмента',
            'values': [
                '⚡ Активные клиенты',
                '😴 Спящие клиенты', 
                '✈️ Путешественники',
                '💎 VIP клиенты'
            ],
            'source': 'Классификация на основе характеристик'
        }
    }
    
    # Метрики качества
    quality_metrics = {
        'silhouette_score': {
            'type': 'float',
            'description': 'Силуэтный коэффициент качества кластеризации',
            'range': '[-1, 1]',
            'interpretation': 'Чем ближе к 1, тем лучше качество',
            'current_value': 0.483
        },
        'calinski_harabasz_index': {
            'type': 'float',
            'description': 'Индекс Калински-Харабаша',
            'interpretation': 'Чем выше, тем лучше разделение кластеров',
            'current_value': 765.38
        },
        'davies_bouldin_index': {
            'type': 'float',
            'description': 'Индекс Дэвиса-Болдина',
            'interpretation': 'Чем ниже, тем лучше качество кластеризации',
            'current_value': 0.916
        }
    }
    
    # Бизнес-метрики
    business_metrics = {
        'customer_lifetime_value': {
            'type': 'float',
            'description': 'Пожизненная ценность клиента (тенге)',
            'formula': 'avg_monthly_revenue * estimated_lifetime_months * (1 - churn_risk)',
            'source': 'Расчетная метрика'
        },
        'churn_risk': {
            'type': 'float',
            'description': 'Риск оттока клиента',
            'range': '[0, 1]',
            'interpretation': '0 = низкий риск, 1 = высокий риск',
            'source': 'Расчетная метрика'
        },
        'stability_index': {
            'type': 'float',
            'description': 'Индекс стабильности сегмента',
            'formula': '(cv_transactions + cv_amount + cv_merchants) / 3',
            'interpretation': 'Чем ниже, тем стабильнее сегмент',
            'source': 'Расчетная метрика'
        },
        'growth_potential': {
            'type': 'string',
            'description': 'Потенциал роста сегмента',
            'values': ['Низкий', 'Средний', 'Высокий', 'Очень высокий'],
            'source': 'Экспертная оценка'
        }
    }
    
    # Создаем полный словарь
    data_dictionary = {
        'metadata': {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0',
            'description': 'Словарь данных для системы сегментации банковских клиентов',
            'total_fields': len(original_fields) + len(behavioral_features) + len(segmentation_results) + len(quality_metrics) + len(business_metrics)
        },
        'original_fields': original_fields,
        'behavioral_features': behavioral_features,
        'segmentation_results': segmentation_results,
        'quality_metrics': quality_metrics,
        'business_metrics': business_metrics
    }
    
    return data_dictionary

def create_csv_dictionary():
    """Создание CSV версии словаря данных"""
    
    data_dict = create_data_dictionary()
    
    # Подготавливаем данные для CSV
    csv_data = []
    
    # Исходные поля
    for field, info in data_dict['original_fields'].items():
        csv_data.append({
            'field_name': field,
            'category': 'Исходные данные',
            'data_type': info['type'],
            'description': info['description'],
            'example': info.get('example', ''),
            'formula': '',
            'source': info['source']
        })
    
    # Поведенческие признаки
    for field, info in data_dict['behavioral_features'].items():
        csv_data.append({
            'field_name': field,
            'category': 'Поведенческие признаки',
            'data_type': info['type'],
            'description': info['description'],
            'example': '',
            'formula': info.get('formula', ''),
            'source': info['source']
        })
    
    # Результаты сегментации
    for field, info in data_dict['segmentation_results'].items():
        csv_data.append({
            'field_name': field,
            'category': 'Результаты сегментации',
            'data_type': info['type'],
            'description': info['description'],
            'example': str(info.get('values', '')),
            'formula': '',
            'source': info['source']
        })
    
    # Метрики качества
    for field, info in data_dict['quality_metrics'].items():
        csv_data.append({
            'field_name': field,
            'category': 'Метрики качества',
            'data_type': info['type'],
            'description': info['description'],
            'example': str(info.get('current_value', '')),
            'formula': '',
            'source': 'Автоматический расчет'
        })
    
    # Бизнес-метрики
    for field, info in data_dict['business_metrics'].items():
        csv_data.append({
            'field_name': field,
            'category': 'Бизнес-метрики',
            'data_type': info['type'],
            'description': info['description'],
            'example': '',
            'formula': info.get('formula', ''),
            'source': info['source']
        })
    
    return pd.DataFrame(csv_data)

def create_excel_dictionary():
    """Создание Excel версии с несколькими листами"""
    
    data_dict = create_data_dictionary()
    
    # Создаем Excel файл с несколькими листами
    with pd.ExcelWriter('results/data_dictionary.xlsx', engine='openpyxl') as writer:
        
        # Лист 1: Общая информация
        metadata_df = pd.DataFrame([
            ['Дата создания', data_dict['metadata']['created_date']],
            ['Версия', data_dict['metadata']['version']],
            ['Описание', data_dict['metadata']['description']],
            ['Общее количество полей', data_dict['metadata']['total_fields']],
            ['Исходных полей', len(data_dict['original_fields'])],
            ['Поведенческих признаков', len(data_dict['behavioral_features'])],
            ['Результатов сегментации', len(data_dict['segmentation_results'])],
            ['Метрик качества', len(data_dict['quality_metrics'])],
            ['Бизнес-метрик', len(data_dict['business_metrics'])]
        ], columns=['Параметр', 'Значение'])
        
        metadata_df.to_excel(writer, sheet_name='Общая информация', index=False)
        
        # Лист 2: Исходные поля
        original_df = pd.DataFrame([
            {
                'Поле': field,
                'Тип данных': info['type'],
                'Описание': info['description'],
                'Пример': info.get('example', ''),
                'Источник': info['source']
            }
            for field, info in data_dict['original_fields'].items()
        ])
        original_df.to_excel(writer, sheet_name='Исходные поля', index=False)
        
        # Лист 3: Поведенческие признаки
        behavioral_df = pd.DataFrame([
            {
                'Признак': field,
                'Тип данных': info['type'],
                'Описание': info['description'],
                'Формула': info.get('formula', ''),
                'Источник': info['source']
            }
            for field, info in data_dict['behavioral_features'].items()
        ])
        behavioral_df.to_excel(writer, sheet_name='Поведенческие признаки', index=False)
        
        # Лист 4: Результаты сегментации
        segmentation_df = pd.DataFrame([
            {
                'Поле': field,
                'Тип данных': info['type'],
                'Описание': info['description'],
                'Возможные значения': str(info.get('values', '')),
                'Источник': info['source']
            }
            for field, info in data_dict['segmentation_results'].items()
        ])
        segmentation_df.to_excel(writer, sheet_name='Результаты сегментации', index=False)
        
        # Лист 5: Метрики
        metrics_data = []
        
        # Добавляем метрики качества
        for field, info in data_dict['quality_metrics'].items():
            metrics_data.append({
                'Метрика': field,
                'Категория': 'Качество кластеризации',
                'Тип данных': info['type'],
                'Описание': info['description'],
                'Текущее значение': info.get('current_value', ''),
                'Интерпретация': info.get('interpretation', '')
            })
        
        # Добавляем бизнес-метрики
        for field, info in data_dict['business_metrics'].items():
            metrics_data.append({
                'Метрика': field,
                'Категория': 'Бизнес-метрики',
                'Тип данных': info['type'],
                'Описание': info['description'],
                'Текущее значение': '',
                'Интерпретация': info.get('interpretation', '')
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Метрики', index=False)

def main():
    """Основная функция создания словаря данных"""
    print("📚 Создание Data Dictionary...")
    
    # Создаем JSON версию
    print("   📄 Создание JSON версии...")
    data_dict = create_data_dictionary()
    
    with open('results/data_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    # Создаем CSV версию
    print("   📊 Создание CSV версии...")
    csv_df = create_csv_dictionary()
    csv_df.to_csv('results/data_dictionary.csv', index=False, encoding='utf-8')
    
    # Создаем Excel версию
    print("   📈 Создание Excel версии...")
    create_excel_dictionary()
    
    print("✅ Data Dictionary создан в трех форматах:")
    print("   📄 JSON: results/data_dictionary.json")
    print("   📊 CSV: results/data_dictionary.csv")
    print("   📈 Excel: results/data_dictionary.xlsx")
    
    # Выводим статистику
    print(f"\n📊 Статистика:")
    print(f"   • Исходных полей: {len(data_dict['original_fields'])}")
    print(f"   • Поведенческих признаков: {len(data_dict['behavioral_features'])}")
    print(f"   • Результатов сегментации: {len(data_dict['segmentation_results'])}")
    print(f"   • Метрик качества: {len(data_dict['quality_metrics'])}")
    print(f"   • Бизнес-метрик: {len(data_dict['business_metrics'])}")
    print(f"   • Всего полей: {data_dict['metadata']['total_fields']}")

if __name__ == "__main__":
    main() 