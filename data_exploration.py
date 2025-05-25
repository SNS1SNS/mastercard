import pandas as pd
import numpy as np


print("Загружаем данные...")
df = pd.read_parquet('DECENTRATHON_3.0.parquet')


print(f"Размер данных: {df.shape}")
print(f"Количество строк: {df.shape[0]:,}")
print(f"Количество столбцов: {df.shape[1]}")

print("\nСтолбцы:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nТипы данных:")
print(df.dtypes)

print("\nПервые 5 строк:")
print(df.head())

print("\nОписательная статистика:")
print(df.describe())

print("\nПропущенные значения:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nУникальные значения в ключевых столбцах:")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count:,} уникальных значений") 