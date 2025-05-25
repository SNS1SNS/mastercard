"""
Проект сегментации клиентов банка на основе поведенческих характеристик
Автор: AI Assistant
Дата: 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Безопасная настройка стиля
plt.style.use('default')
try:
    sns.set_palette("husl")
except:
    pass

class BankClientSegmentation:
    """Класс для сегментации клиентов банка"""
    
    def __init__(self, data_path):
        """Инициализация с загрузкой данных"""
        print("🔄 Загружаем данные...")
        self.df = pd.read_parquet(data_path)
        print(f"✅ Загружено {self.df.shape[0]:,} транзакций")
        
        
        self.client_features = None
        self.scaled_features = None
        self.clusters = None
        self.scaler = None
        
    def explore_data(self):
        """Первичный анализ данных"""
        print("\n" + "="*50)
        print("📊 АНАЛИЗ ДАННЫХ")
        print("="*50)
        
        print(f"Размер данных: {self.df.shape}")
        print(f"Период данных: {self.df['transaction_timestamp'].min()} - {self.df['transaction_timestamp'].max()}")
        print(f"Количество уникальных клиентов: {self.df['card_id'].nunique():,}")
        print(f"Количество уникальных мерчантов: {self.df['merchant_id'].nunique():,}")
        
        
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("\n🔍 Пропущенные значения:")
            print(missing_data[missing_data > 0])
        
        
        print(f"\n💰 Статистика по суммам транзакций:")
        print(f"Средняя сумма: {self.df['transaction_amount_kzt'].mean():.2f} тенге")
        print(f"Медианная сумма: {self.df['transaction_amount_kzt'].median():.2f} тенге")
        print(f"Общий оборот: {self.df['transaction_amount_kzt'].sum():,.2f} тенге")
        
    def create_client_features(self):
        """Создание поведенческих характеристик клиентов"""
        print("\n🔧 Создаем поведенческие характеристики клиентов...")
        
        
        self.df['hour'] = self.df['transaction_timestamp'].dt.hour
        self.df['day_of_week'] = self.df['transaction_timestamp'].dt.dayofweek
        self.df['month'] = self.df['transaction_timestamp'].dt.month
        
        
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
        
        
        client_agg['amount_range'] = client_agg['max_amount'] - client_agg['min_amount']
        client_agg['purchase_ratio'] = client_agg['purchase_count'] / client_agg['total_transactions']
        client_agg['avg_merchants_per_transaction'] = client_agg['unique_merchants'] / client_agg['total_transactions']
        client_agg['spending_consistency'] = 1 / (1 + client_agg['std_amount'] / client_agg['avg_amount'])
        
        
        client_agg['activity_level'] = pd.cut(
            client_agg['total_transactions'], 
            bins=[0, 10, 50, 200, float('inf')], 
            labels=['Низкая', 'Средняя', 'Высокая', 'Очень высокая']
        )
        
        
        client_agg['spending_level'] = pd.cut(
            client_agg['total_amount'], 
            bins=[0, 100000, 500000, 2000000, float('inf')], 
            labels=['Низкий', 'Средний', 'Высокий', 'Премиум']
        )
        
        self.client_features = client_agg
        print(f"✅ Создано {len(self.client_features)} профилей клиентов")
        
        return client_agg
    
    def prepare_features_for_clustering(self):
        """Подготовка признаков для кластеризации"""
        print("\n🎯 Подготавливаем признаки для кластеризации...")
        
        
        numeric_features = [
            'total_transactions', 'total_amount', 'avg_amount', 'median_amount',
            'std_amount', 'amount_range', 'unique_merchants', 'unique_categories',
            'unique_cities', 'purchase_ratio', 'avg_merchants_per_transaction',
            'spending_consistency', 'preferred_hour', 'preferred_day'
        ]
        
        
        features_df = self.client_features[numeric_features].fillna(0)
        
        
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_mask = isolation_forest.fit_predict(features_df) == 1
        
        print(f"🔍 Обнаружено {(~outlier_mask).sum()} выбросов ({(~outlier_mask).mean()*100:.1f}%)")
        
        
        features_clean = features_df[outlier_mask]
        
        
        self.scaler = RobustScaler()
        self.scaled_features = self.scaler.fit_transform(features_clean)
        
        print(f"✅ Подготовлено {self.scaled_features.shape[0]} клиентов для кластеризации")
        
        return self.scaled_features, features_clean.index
    
    def find_optimal_clusters(self, max_clusters=15):
        """Поиск оптимального количества кластеров"""
        print("\n🔍 Поиск оптимального количества кластеров...")
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.scaled_features, cluster_labels))
        
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        
        axes[0].plot(K_range, inertias, 'bo-')
        axes[0].set_title('Метод локтя')
        axes[0].set_xlabel('Количество кластеров')
        axes[0].set_ylabel('Инерция')
        axes[0].grid(True)
        
        
        axes[1].plot(K_range, silhouette_scores, 'ro-')
        axes[1].set_title('Силуэтный анализ')
        axes[1].set_xlabel('Количество кластеров')
        axes[1].set_ylabel('Силуэтный коэффициент')
        axes[1].grid(True)
        
        
        axes[2].plot(K_range, calinski_scores, 'go-')
        axes[2].set_title('Индекс Калински-Харабаша')
        axes[2].set_xlabel('Количество кластеров')
        axes[2].set_ylabel('Индекс CH')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()  # Освобождаем память
        print("✅ График оптимизации кластеров сохранен")
        
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"🎯 Рекомендуемое количество кластеров: {optimal_k}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=None):
        """Выполнение кластеризации"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"\n🎯 Выполняем кластеризацию с {n_clusters} кластерами...")
        
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.scaled_features)
        
        
        silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
        calinski_score = calinski_harabasz_score(self.scaled_features, cluster_labels)
        
        print(f"📊 Качество кластеризации:")
        print(f"   Силуэтный коэффициент: {silhouette_avg:.3f}")
        print(f"   Индекс Калински-Харабаша: {calinski_score:.2f}")
        
        self.clusters = cluster_labels
        self.kmeans_model = kmeans
        
        return cluster_labels
    
    def analyze_clusters(self):
        """Анализ полученных кластеров"""
        print("\n📈 АНАЛИЗ КЛАСТЕРОВ")
        print("="*50)
        
        
        valid_indices = self.client_features.index[self.client_features.index.isin(
            self.client_features.index[~self.client_features.index.duplicated()]
        )]
        
        
        cluster_df = self.client_features.loc[valid_indices[:len(self.clusters)]].copy()
        cluster_df['cluster'] = self.clusters
        
        
        cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
        print("Размеры кластеров:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(cluster_df)) * 100
            print(f"  Кластер {cluster_id}: {size:,} клиентов ({percentage:.1f}%)")
        
        
        print("\n📊 Характеристики кластеров:")
        
        key_metrics = ['total_transactions', 'total_amount', 'avg_amount', 
                      'unique_merchants', 'unique_categories', 'purchase_ratio']
        
        cluster_summary = cluster_df.groupby('cluster')[key_metrics].agg(['mean', 'median']).round(2)
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            print(f"\n🎯 КЛАСТЕР {cluster_id}:")
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            
            print(f"   Размер: {len(cluster_data):,} клиентов")
            print(f"   Среднее количество транзакций: {cluster_data['total_transactions'].mean():.1f}")
            print(f"   Средняя общая сумма: {cluster_data['total_amount'].mean():,.0f} тенге")
            print(f"   Средний чек: {cluster_data['avg_amount'].mean():,.0f} тенге")
            print(f"   Среднее количество мерчантов: {cluster_data['unique_merchants'].mean():.1f}")
            print(f"   Доля покупок: {cluster_data['purchase_ratio'].mean():.2f}")
            
            
            if cluster_data['total_transactions'].mean() > cluster_df['total_transactions'].mean():
                if cluster_data['avg_amount'].mean() > cluster_df['avg_amount'].mean():
                    cluster_type = "🌟 Премиум клиенты (высокая активность + высокий чек)"
                else:
                    cluster_type = "⚡ Активные клиенты (высокая активность + средний чек)"
            else:
                if cluster_data['avg_amount'].mean() > cluster_df['avg_amount'].mean():
                    cluster_type = "💎 VIP клиенты (низкая активность + высокий чек)"
                else:
                    cluster_type = "😴 Пассивные клиенты (низкая активность + низкий чек)"
            
            print(f"   Тип: {cluster_type}")
        
        return cluster_df
    
    def visualize_clusters(self, cluster_df):
        """Визуализация кластеров"""
        print("\n🎨 Создаем визуализации...")
        
        
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(self.scaled_features)
        
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Кластеры в пространстве PCA', 'Распределение по активности',
                          'Распределение по суммам', 'Соотношение активности и сумм'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        
        colors = px.colors.qualitative.Set1[:len(cluster_df['cluster'].unique())]
        for i, cluster_id in enumerate(sorted(cluster_df['cluster'].unique())):
            mask = self.clusters == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=pca_features[mask, 0],
                    y=pca_features[mask, 1],
                    mode='markers',
                    name=f'Кластер {cluster_id}',
                    marker=dict(color=colors[i], size=5, opacity=0.6),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        
        cluster_transactions = cluster_df.groupby('cluster')['total_transactions'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Кластер {i}' for i in cluster_transactions.index],
                y=cluster_transactions.values,
                name='Среднее количество транзакций',
                marker_color=colors[:len(cluster_transactions)],
                showlegend=False
            ),
            row=1, col=2
        )
        
        
        cluster_amounts = cluster_df.groupby('cluster')['total_amount'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Кластер {i}' for i in cluster_amounts.index],
                y=cluster_amounts.values,
                name='Средняя общая сумма',
                marker_color=colors[:len(cluster_amounts)],
                showlegend=False
            ),
            row=2, col=1
        )
        
        
        cluster_avg_amounts = cluster_df.groupby('cluster')['avg_amount'].mean()
        fig.add_trace(
            go.Scatter(
                x=cluster_transactions.values,
                y=cluster_avg_amounts.values,
                mode='markers+text',
                text=[f'Кластер {i}' for i in cluster_transactions.index],
                textposition="top center",
                marker=dict(
                    size=cluster_df.groupby('cluster').size().values / 10,
                    color=colors[:len(cluster_transactions)],
                    opacity=0.7
                ),
                name='Кластеры',
                showlegend=False
            ),
            row=2, col=2
        )
        
        
        fig.update_layout(
            title_text="Анализ кластеров клиентов банка",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_yaxes(title_text="Количество транзакций", row=1, col=2)
        fig.update_yaxes(title_text="Общая сумма (тенге)", row=2, col=1)
        fig.update_xaxes(title_text="Количество транзакций", row=2, col=2)
        fig.update_yaxes(title_text="Средний чек (тенге)", row=2, col=2)
        
        fig.write_html("cluster_analysis.html")
        print("✅ Интерактивная визуализация сохранена в cluster_analysis.html")
        
        
        plt.figure(figsize=(15, 10))
        
        
        plt.subplot(2, 3, 1)
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                            c=self.clusters, cmap='tab10', alpha=0.6)
        plt.title('Кластеры в пространстве PCA')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
        plt.colorbar(scatter)
        
        
        plt.subplot(2, 3, 2)
        cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
        plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
        plt.title('Размеры кластеров')
        plt.xlabel('Кластер')
        plt.ylabel('Количество клиентов')
        plt.xticks(range(len(cluster_sizes)), [f'Кластер {i}' for i in cluster_sizes.index])
        
        
        plt.subplot(2, 3, 3)
        cluster_transactions = cluster_df.groupby('cluster')['total_transactions'].mean()
        plt.bar(range(len(cluster_transactions)), cluster_transactions.values)
        plt.title('Среднее количество транзакций')
        plt.xlabel('Кластер')
        plt.ylabel('Количество транзакций')
        plt.xticks(range(len(cluster_transactions)), [f'Кластер {i}' for i in cluster_transactions.index])
        
        
        plt.subplot(2, 3, 4)
        cluster_amounts = cluster_df.groupby('cluster')['total_amount'].mean()
        plt.bar(range(len(cluster_amounts)), cluster_amounts.values)
        plt.title('Средняя общая сумма')
        plt.xlabel('Кластер')
        plt.ylabel('Сумма (тенге)')
        plt.xticks(range(len(cluster_amounts)), [f'Кластер {i}' for i in cluster_amounts.index])
        
        
        plt.subplot(2, 3, 5)
        cluster_avg_amounts = cluster_df.groupby('cluster')['avg_amount'].mean()
        plt.bar(range(len(cluster_avg_amounts)), cluster_avg_amounts.values)
        plt.title('Средний чек')
        plt.xlabel('Кластер')
        plt.ylabel('Средний чек (тенге)')
        plt.xticks(range(len(cluster_avg_amounts)), [f'Кластер {i}' for i in cluster_avg_amounts.index])
        
        
        plt.subplot(2, 3, 6)
        key_metrics = ['total_transactions', 'total_amount', 'avg_amount', 'unique_merchants']
        cluster_heatmap_data = cluster_df.groupby('cluster')[key_metrics].mean()
        
        # Нормализация данных для тепловой карты
        try:
            cluster_heatmap_normalized = (cluster_heatmap_data - cluster_heatmap_data.min()) / (cluster_heatmap_data.max() - cluster_heatmap_data.min())
            
            # Заменяем NaN на 0 если есть
            cluster_heatmap_normalized = cluster_heatmap_normalized.fillna(0)
            
            sns.heatmap(cluster_heatmap_normalized.T, annot=True, cmap='YlOrRd', 
                       xticklabels=[f'Кластер {i}' for i in cluster_heatmap_data.index],
                       yticklabels=['Транзакции', 'Общая сумма', 'Средний чек', 'Мерчанты'],
                       fmt='.2f')
            plt.title('Нормализованные характеристики кластеров')
        except Exception as e:
            print(f"⚠️ Ошибка при создании тепловой карты: {e}")
            plt.text(0.5, 0.5, 'Ошибка создания\nтепловой карты', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Характеристики кластеров (ошибка)')
        
        plt.tight_layout()
        plt.savefig('cluster_static_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Освобождаем память
        print("✅ Статические графики созданы и сохранены")
    
    def generate_business_recommendations(self, cluster_df):
        """Генерация бизнес-рекомендаций"""
        print("\n💼 БИЗНЕС-РЕКОМЕНДАЦИИ")
        print("="*50)
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            size_pct = (len(cluster_data) / len(cluster_df)) * 100
            
            print(f"\n🎯 КЛАСТЕР {cluster_id} ({size_pct:.1f}% клиентов):")
            
            avg_transactions = cluster_data['total_transactions'].mean()
            avg_amount = cluster_data['avg_amount'].mean()
            avg_total = cluster_data['total_amount'].mean()
            avg_merchants = cluster_data['unique_merchants'].mean()
            
            
            if avg_transactions > cluster_df['total_transactions'].mean():
                if avg_amount > cluster_df['avg_amount'].mean():
                    print("   📈 ТИП: Премиум клиенты")
                    print("   🎯 СТРАТЕГИЯ:")
                    print("     • Программы лояльности с эксклюзивными привилегиями")
                    print("     • Персональный менеджер")
                    print("     • Премиум продукты (Private Banking)")
                    print("     • Инвестиционные продукты")
                    print("     • Кэшбэк программы с повышенными ставками")
                else:
                    print("   ⚡ ТИП: Активные клиенты")
                    print("   🎯 СТРАТЕГИЯ:")
                    print("     • Программы лояльности с бонусами за активность")
                    print("     • Кредитные продукты с льготными условиями")
                    print("     • Мобильные приложения с расширенным функционалом")
                    print("     • Кэшбэк в популярных категориях")
                    print("     • Уведомления о специальных предложениях")
            else:
                if avg_amount > cluster_df['avg_amount'].mean():
                    print("   💎 ТИП: VIP клиенты")
                    print("   🎯 СТРАТЕГИЯ:")
                    print("     • Стимулирование частоты использования карт")
                    print("     • Специальные предложения для увеличения активности")
                    print("     • Консьерж-сервисы")
                    print("     • Эксклюзивные мероприятия")
                    print("     • Персонализированные предложения")
                else:
                    print("   😴 ТИП: Пассивные клиенты")
                    print("   🎯 СТРАТЕГИЯ:")
                    print("     • Реактивационные кампании")
                    print("     • Образовательные программы по финансовой грамотности")
                    print("     • Простые и понятные продукты")
                    print("     • Стимулирующие акции для первых транзакций")
                    print("     • Упрощение процессов использования карт")
            
            print(f"   📊 КЛЮЧЕВЫЕ МЕТРИКИ:")
            print(f"     • Среднее количество транзакций: {avg_transactions:.1f}")
            print(f"     • Средний чек: {avg_amount:,.0f} тенге")
            print(f"     • Средний оборот: {avg_total:,.0f} тенге")
            print(f"     • Среднее количество мерчантов: {avg_merchants:.1f}")
    
    def save_results(self, cluster_df):
        """Сохранение результатов"""
        print("\n💾 Сохраняем результаты...")
        
        
        cluster_df.to_csv('client_segments.csv', index=True)
        
        
        summary_stats = cluster_df.groupby('cluster').agg({
            'total_transactions': ['count', 'mean', 'median', 'std'],
            'total_amount': ['mean', 'median', 'std'],
            'avg_amount': ['mean', 'median', 'std'],
            'unique_merchants': ['mean', 'median'],
            'unique_categories': ['mean', 'median'],
            'purchase_ratio': ['mean', 'median']
        }).round(2)
        
        summary_stats.to_csv('cluster_summary_statistics.csv')
        
        print("✅ Результаты сохранены:")
        print("   • client_segments.csv - детальная информация по клиентам")
        print("   • cluster_summary_statistics.csv - сводная статистика по кластерам")
        print("   • cluster_analysis.html - интерактивная визуализация")
        print("   • cluster_static_analysis.png - статичные графики")
    
    def run_full_analysis(self, n_clusters=None):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА СЕГМЕНТАЦИИ КЛИЕНТОВ")
        print("="*60)
        
        
        self.explore_data()
        
        
        self.create_client_features()
        
        
        self.prepare_features_for_clustering()
        
        
        self.perform_clustering(n_clusters)
        
        
        cluster_df = self.analyze_clusters()
        
        
        self.visualize_clusters(cluster_df)
        
        
        self.generate_business_recommendations(cluster_df)
        
        
        self.save_results(cluster_df)
        
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
        print("="*60)
        
        return cluster_df


def main():
    """Основная функция"""
    
    segmentation = BankClientSegmentation('DECENTRATHON_3.0.parquet')
    
    
    results = segmentation.run_full_analysis()
    
    return segmentation, results

if __name__ == "__main__":
    segmentation, results = main() 