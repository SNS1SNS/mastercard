# 🚀 Инструкции по установке и запуску

## 📋 Требования к системе

- **Python**: версия 3.8 или выше
- **Операционная система**: Windows, macOS, Linux
- **Память**: минимум 8 ГБ RAM (рекомендуется 16 ГБ)
- **Свободное место**: минимум 2 ГБ

## 🔧 Установка

### 1. Клонирование проекта

```bash
git clone https://github.com/SNS1SNS/mastercard
cd model
```

### 2. Создание виртуального окружения

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Проверка установки

Запустите демонстрационный скрипт для проверки:

```bash
python demo.py
```

## 📊 Подготовка данных

Убедитесь, что файл данных `DECENTRATHON_3.0.parquet` находится в корневой папке проекта.

### Структура данных

Файл должен содержать следующие колонки:
- `card_id` - идентификатор карты клиента
- `merchant_id` - идентификатор мерчанта
- `amount` - сумма транзакции
- `transaction_date` - дата и время транзакции
- `mcc` - код категории мерчанта (опционально)

## 🚀 Запуск анализа

### Полный анализ

```bash
python main.py
```

### Быстрый анализ (на выборке данных)

```bash
python main.py --quick
```

### Демонстрационный режим

```bash
python demo.py
```

## 📁 Результаты

После выполнения анализа в папке `results/` будут созданы следующие файлы:

### 📊 Основные результаты
- `client_segments.csv` - сегментация клиентов
- `segment_summary.csv` - сводка по сегментам
- `segment_profiles.json` - профили сегментов
- `business_recommendations.json` - бизнес-рекомендации

### 📈 Визуализации
- `segment_dashboard.html` - интерактивная панель
- `segment_analysis_static.png` - статические графики
- `pca_visualization.png` - PCA визуализация
- `segment_heatmap.png` - тепловая карта
- `cluster_optimization.png` - оптимизация кластеров

### 🔍 Аналитика
- `feature_analysis.csv` - анализ признаков

## ⚠️ Устранение проблем

### Ошибка импорта модулей

```bash
pip install --upgrade -r requirements.txt
```

### Ошибка памяти

Используйте быстрый режим:
```bash
python main.py --quick
```

### Файл данных не найден

Убедитесь, что файл `DECENTRATHON_3.0.parquet` находится в корневой папке проекта.

### Ошибки визуализации

Установите дополнительные зависимости:
```bash
pip install kaleido plotly-orca
```

## 🔧 Настройка конфигурации

Отредактируйте файл `config.py` для изменения параметров:

```python
# Изменение количества кластеров
config.model.n_clusters = 6

# Изменение алгоритма кластеризации
config.model.algorithm = 'gaussian_mixture'

# Изменение признаков для кластеризации
config.features.clustering_features = [
    'total_transactions',
    'avg_amount',
    'total_amount'
]
```

## 📞 Поддержка

При возникновении проблем:

1. Проверьте версию Python: `python --version`
2. Проверьте установленные пакеты: `pip list`
3. Запустите демонстрацию: `python demo.py`
4. Проверьте логи в консоли

## 🎯 Быстрый старт

Для быстрого запуска выполните следующие команды:

```bash
# 1. Активация виртуального окружения
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Проверка системы
python demo.py

# 3. Запуск анализа
python main.py --quick

# 4. Просмотр результатов
# Откройте файл results/segment_dashboard.html в браузере
```

## 📚 Дополнительная информация

- Подробное описание проекта: `README.md`
- Структура кода: см. комментарии в файлах
- Методология: см. раздел "Методология" в `README.md`

---

**Удачного анализа! 🎉** 