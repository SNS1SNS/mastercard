"""
Модуль для анализа сегментов клиентов и генерации бизнес-инсайтов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import warnings
import json
warnings.filterwarnings('ignore')

from config import config

class SegmentAnalyzer:
    """Класс для анализа клиентских сегментов"""
    
    def __init__(self, client_features: pd.DataFrame, cluster_labels: np.ndarray, 
                 scaled_features: np.ndarray, clean_indices: pd.Index):
        """Инициализация анализатора сегментов"""
        self.client_features = client_features
        self.cluster_labels = cluster_labels
        self.scaled_features = scaled_features
        self.clean_indices = clean_indices
        
        
        self.cluster_df = self._create_cluster_dataframe()
        self.segment_profiles = {}
        self.business_recommendations = {}
        
    def _create_cluster_dataframe(self) -> pd.DataFrame:
        """Создание DataFrame с результатами кластеризации"""
        
        cluster_df = self.client_features.loc[self.clean_indices].copy()
        cluster_df['cluster'] = self.cluster_labels
        
        return cluster_df
    
    def analyze_segments(self) -> Dict[int, Dict[str, Any]]:
        """Детальный анализ каждого сегмента"""
        print("\n📈 АНАЛИЗ СЕГМЕНТОВ КЛИЕНТОВ")
        print("="*50)
        
        segment_analysis = {}
        
        
        total_clients = len(self.cluster_df)
        cluster_sizes = self.cluster_df['cluster'].value_counts().sort_index()
        
        print(f"Общее количество клиентов: {total_clients:,}")
        print(f"Количество сегментов: {len(cluster_sizes)}")
        
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            size = len(cluster_data)
            percentage = (size / total_clients) * 100
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id}:")
            print(f"   Размер: {size:,} клиентов ({percentage:.1f}%)")
            
            
            characteristics = self._calculate_segment_characteristics(cluster_data)
            
            
            segment_type = self._classify_segment(cluster_data, characteristics)
            
            
            time_patterns = self._analyze_time_patterns(cluster_data)
            
            
            tech_preferences = self._analyze_tech_preferences(cluster_data)
            
            segment_analysis[cluster_id] = {
                'size': size,
                'percentage': percentage,
                'characteristics': characteristics,
                'segment_type': segment_type,
                'time_patterns': time_patterns,
                'tech_preferences': tech_preferences
            }
            
            
            print(f"   Тип сегмента: {segment_type['name']}")
            print(f"   Среднее количество транзакций: {characteristics['avg_transactions']:.1f}")
            print(f"   Средняя общая сумма: {characteristics['avg_total_amount']:,.0f} тенге")
            print(f"   Средний чек: {characteristics['avg_amount']:,.0f} тенге")
            print(f"   Среднее количество мерчантов: {characteristics['avg_merchants']:.1f}")
            print(f"   Доля покупок: {characteristics['purchase_ratio']:.2f}")
        
        self.segment_profiles = segment_analysis
        return segment_analysis
    
    def _calculate_segment_characteristics(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Расчет характеристик сегмента"""
        return {
            'avg_transactions': cluster_data['total_transactions'].mean(),
            'median_transactions': cluster_data['total_transactions'].median(),
            'avg_total_amount': cluster_data['total_amount'].mean(),
            'median_total_amount': cluster_data['total_amount'].median(),
            'avg_amount': cluster_data['avg_amount'].mean(),
            'median_amount': cluster_data['median_amount'].mean(),
            'avg_merchants': cluster_data['unique_merchants'].mean(),
            'avg_categories': cluster_data['unique_categories'].mean(),
            'avg_cities': cluster_data['unique_cities'].mean(),
            'purchase_ratio': cluster_data['purchase_ratio'].mean(),
            'spending_consistency': cluster_data['spending_consistency'].mean(),
            'amount_range': cluster_data['amount_range'].mean()
        }
    
    def _classify_segment(self, cluster_data: pd.DataFrame, characteristics: Dict[str, float]) -> Dict[str, str]:
        """Классификация типа сегмента"""
        
        overall_avg_transactions = self.cluster_df['total_transactions'].mean()
        overall_avg_amount = self.cluster_df['avg_amount'].mean()
        overall_avg_total = self.cluster_df['total_amount'].mean()
        
        avg_transactions = characteristics['avg_transactions']
        avg_amount = characteristics['avg_amount']
        avg_total = characteristics['avg_total_amount']
        
        
        if avg_transactions > overall_avg_transactions * 1.2:  
            if avg_amount > overall_avg_amount * 1.2:  
                return {
                    'name': '🌟 Премиум клиенты',
                    'description': 'Высокая активность + высокий средний чек',
                    'priority': 'Высокий',
                    'value': 'Очень высокая'
                }
            else:  
                return {
                    'name': '⚡ Активные клиенты',
                    'description': 'Высокая активность + средний чек',
                    'priority': 'Высокий',
                    'value': 'Высокая'
                }
        else:  
            if avg_amount > overall_avg_amount * 1.2:  
                return {
                    'name': '💎 VIP клиенты',
                    'description': 'Низкая активность + высокий чек',
                    'priority': 'Средний',
                    'value': 'Высокая'
                }
            elif avg_transactions < overall_avg_transactions * 0.5:  
                return {
                    'name': '😴 Спящие клиенты',
                    'description': 'Очень низкая активность + низкий чек',
                    'priority': 'Низкий',
                    'value': 'Низкая'
                }
            else:  
                return {
                    'name': '🔄 Обычные клиенты',
                    'description': 'Средняя активность + средний чек',
                    'priority': 'Средний',
                    'value': 'Средняя'
                }
    
    def _analyze_time_patterns(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ временных паттернов сегмента"""
        return {
            'preferred_hour': cluster_data['preferred_hour'].mode().iloc[0] if not cluster_data['preferred_hour'].mode().empty else 12,
            'preferred_day': cluster_data['preferred_day'].mode().iloc[0] if not cluster_data['preferred_day'].mode().empty else 0,
            'hour_distribution': cluster_data['preferred_hour'].value_counts().to_dict(),
            'day_distribution': cluster_data['preferred_day'].value_counts().to_dict()
        }
    
    def _analyze_tech_preferences(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ технологических предпочтений"""
        return {
            'preferred_pos_mode': cluster_data['preferred_pos_mode'].mode().iloc[0] if not cluster_data['preferred_pos_mode'].mode().empty else 'Unknown',
            'preferred_wallet': cluster_data['preferred_wallet'].mode().iloc[0] if not cluster_data['preferred_wallet'].mode().empty else 'Unknown',
            'pos_distribution': cluster_data['preferred_pos_mode'].value_counts().to_dict(),
            'wallet_distribution': cluster_data['preferred_wallet'].value_counts().to_dict()
        }
    
    def generate_segment_names(self) -> Dict[int, str]:
        """Генерация осмысленных названий сегментов"""
        segment_names = {}
        
        for cluster_id, profile in self.segment_profiles.items():
            characteristics = profile['characteristics']
            segment_type = profile['segment_type']
            
            
            base_name = segment_type['name']
            
            
            if characteristics['avg_merchants'] > self.cluster_df['unique_merchants'].mean() * 1.5:
                base_name += " (Разнообразные покупки)"
            elif characteristics['spending_consistency'] > 0.8:
                base_name += " (Стабильные траты)"
            elif characteristics['avg_cities'] > self.cluster_df['unique_cities'].mean() * 1.5:
                base_name += " (Путешественники)"
            
            segment_names[cluster_id] = base_name
        
        return segment_names
    
    def create_comprehensive_visualizations(self):
        """Создание комплексных визуализаций"""
        print("\n🎨 Создание визуализаций сегментов...")
        
        
        self._create_interactive_dashboard()
        
        
        self._create_static_visualizations()
        
        
        self._create_pca_visualization()
        
        
        self._create_heatmap()
        
        print("✅ Все визуализации созданы")
    
    def _create_interactive_dashboard(self):
        """Создание интерактивного дашборда"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Распределение клиентов по сегментам',
                'Средние характеристики сегментов',
                'Активность vs Средний чек',
                'Разнообразие поведения',
                'Временные предпочтения',
                'Технологические предпочтения'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        
        cluster_sizes = self.cluster_df['cluster'].value_counts().sort_index()
        segment_names = self.generate_segment_names()
        
        fig.add_trace(
            go.Pie(
                labels=[segment_names.get(i, f'Сегмент {i}') for i in cluster_sizes.index],
                values=cluster_sizes.values,
                name="Размеры сегментов"
            ),
            row=1, col=1
        )
        
        
        avg_transactions = self.cluster_df.groupby('cluster')['total_transactions'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Сегмент {i}' for i in avg_transactions.index],
                y=avg_transactions.values,
                name='Среднее количество транзакций'
            ),
            row=1, col=2
        )
        
        
        avg_amounts = self.cluster_df.groupby('cluster')['avg_amount'].mean()
        cluster_sizes_for_scatter = self.cluster_df.groupby('cluster').size()
        
        fig.add_trace(
            go.Scatter(
                x=avg_transactions.values,
                y=avg_amounts.values,
                mode='markers+text',
                text=[f'Сегмент {i}' for i in avg_transactions.index],
                textposition="top center",
                marker=dict(
                    size=cluster_sizes_for_scatter.values / 50,
                    opacity=0.7
                ),
                name='Сегменты'
            ),
            row=2, col=1
        )
        
        
        avg_merchants = self.cluster_df.groupby('cluster')['unique_merchants'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Сегмент {i}' for i in avg_merchants.index],
                y=avg_merchants.values,
                name='Среднее количество мерчантов'
            ),
            row=2, col=2
        )
        
        
        preferred_hours = self.cluster_df.groupby('cluster')['preferred_hour'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Сегмент {i}' for i in preferred_hours.index],
                y=preferred_hours.values,
                name='Предпочитаемый час'
            ),
            row=3, col=1
        )
        
        
        
        purchase_ratios = self.cluster_df.groupby('cluster')['purchase_ratio'].mean()
        fig.add_trace(
            go.Bar(
                x=[f'Сегмент {i}' for i in purchase_ratios.index],
                y=purchase_ratios.values,
                name='Доля покупок'
            ),
            row=3, col=2
        )
        
        
        fig.update_layout(
            title_text="Комплексный анализ клиентских сегментов",
            title_x=0.5,
            height=1200,
            showlegend=False
        )
        
        
        output_path = config.get_output_path('segment_dashboard.html')
        fig.write_html(output_path)
        print(f"📊 Интерактивный дашборд сохранен: {output_path}")
    
    def _create_static_visualizations(self):
        """Создание статичных визуализаций"""
        plt.style.use(config.visualization.style)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Анализ клиентских сегментов', fontsize=16, fontweight='bold')
        
        
        cluster_sizes = self.cluster_df['cluster'].value_counts().sort_index()
        axes[0, 0].pie(cluster_sizes.values, labels=[f'Сегмент {i}' for i in cluster_sizes.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Распределение клиентов по сегментам')
        
        
        avg_transactions = self.cluster_df.groupby('cluster')['total_transactions'].mean()
        axes[0, 1].bar(range(len(avg_transactions)), avg_transactions.values)
        axes[0, 1].set_title('Среднее количество транзакций')
        axes[0, 1].set_xlabel('Сегмент')
        axes[0, 1].set_ylabel('Количество транзакций')
        axes[0, 1].set_xticks(range(len(avg_transactions)))
        axes[0, 1].set_xticklabels([f'Сегмент {i}' for i in avg_transactions.index])
        
        
        avg_amounts = self.cluster_df.groupby('cluster')['total_amount'].mean()
        axes[0, 2].bar(range(len(avg_amounts)), avg_amounts.values)
        axes[0, 2].set_title('Средняя общая сумма')
        axes[0, 2].set_xlabel('Сегмент')
        axes[0, 2].set_ylabel('Сумма (тенге)')
        axes[0, 2].set_xticks(range(len(avg_amounts)))
        axes[0, 2].set_xticklabels([f'Сегмент {i}' for i in avg_amounts.index])
        
        
        avg_check = self.cluster_df.groupby('cluster')['avg_amount'].mean()
        axes[1, 0].bar(range(len(avg_check)), avg_check.values)
        axes[1, 0].set_title('Средний чек')
        axes[1, 0].set_xlabel('Сегмент')
        axes[1, 0].set_ylabel('Средний чек (тенге)')
        axes[1, 0].set_xticks(range(len(avg_check)))
        axes[1, 0].set_xticklabels([f'Сегмент {i}' for i in avg_check.index])
        
        
        avg_merchants = self.cluster_df.groupby('cluster')['unique_merchants'].mean()
        axes[1, 1].bar(range(len(avg_merchants)), avg_merchants.values)
        axes[1, 1].set_title('Среднее количество мерчантов')
        axes[1, 1].set_xlabel('Сегмент')
        axes[1, 1].set_ylabel('Количество мерчантов')
        axes[1, 1].set_xticks(range(len(avg_merchants)))
        axes[1, 1].set_xticklabels([f'Сегмент {i}' for i in avg_merchants.index])
        
        
        axes[1, 2].scatter(avg_transactions.values, avg_check.values, 
                          s=cluster_sizes.values/10, alpha=0.7)
        for i, (x, y) in enumerate(zip(avg_transactions.values, avg_check.values)):
            axes[1, 2].annotate(f'Сегмент {avg_transactions.index[i]}', (x, y), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 2].set_title('Активность vs Средний чек')
        axes[1, 2].set_xlabel('Среднее количество транзакций')
        axes[1, 2].set_ylabel('Средний чек (тенге)')
        
        
        preferred_hours = self.cluster_df.groupby('cluster')['preferred_hour'].mean()
        axes[2, 0].bar(range(len(preferred_hours)), preferred_hours.values)
        axes[2, 0].set_title('Предпочитаемое время (час)')
        axes[2, 0].set_xlabel('Сегмент')
        axes[2, 0].set_ylabel('Час')
        axes[2, 0].set_xticks(range(len(preferred_hours)))
        axes[2, 0].set_xticklabels([f'Сегмент {i}' for i in preferred_hours.index])
        
        
        purchase_ratios = self.cluster_df.groupby('cluster')['purchase_ratio'].mean()
        axes[2, 1].bar(range(len(purchase_ratios)), purchase_ratios.values)
        axes[2, 1].set_title('Доля покупок')
        axes[2, 1].set_xlabel('Сегмент')
        axes[2, 1].set_ylabel('Доля покупок')
        axes[2, 1].set_xticks(range(len(purchase_ratios)))
        axes[2, 1].set_xticklabels([f'Сегмент {i}' for i in purchase_ratios.index])
        
        
        spending_consistency = self.cluster_df.groupby('cluster')['spending_consistency'].mean()
        axes[2, 2].bar(range(len(spending_consistency)), spending_consistency.values)
        axes[2, 2].set_title('Консистентность трат')
        axes[2, 2].set_xlabel('Сегмент')
        axes[2, 2].set_ylabel('Консистентность')
        axes[2, 2].set_xticks(range(len(spending_consistency)))
        axes[2, 2].set_xticklabels([f'Сегмент {i}' for i in spending_consistency.index])
        
        plt.tight_layout()
        output_path = config.get_output_path('segment_analysis_static.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ Статические графики созданы и сохранены")
    
    def _create_pca_visualization(self):
        """Создание PCA визуализации"""
        
        pca = PCA(n_components=2, random_state=config.model.random_state)
        pca_features = pca.fit_transform(self.scaled_features)
        
        plt.figure(figsize=(12, 8))
        
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(self.cluster_labels))))
        
        for i, cluster_id in enumerate(sorted(np.unique(self.cluster_labels))):
            mask = self.cluster_labels == cluster_id
            plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                       c=[colors[i]], label=f'Сегмент {cluster_id}', 
                       alpha=0.6, s=50)
        
        plt.title('Визуализация сегментов в пространстве PCA')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = config.get_output_path('pca_visualization.png')
        plt.savefig(output_path, dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"📊 PCA визуализация сохранена: {output_path}")
        print(f"   Объясненная дисперсия: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    
    def _create_heatmap(self):
        """Создание тепловой карты характеристик сегментов"""
        
        key_metrics = [
            'total_transactions', 'total_amount', 'avg_amount', 
            'unique_merchants', 'unique_categories', 'purchase_ratio',
            'spending_consistency'
        ]
        
        try:
            # Группировка данных по кластерам
            heatmap_data = self.cluster_df.groupby('cluster')[key_metrics].mean()
            
            # Проверка на наличие данных
            if heatmap_data.empty:
                print("⚠️ Нет данных для создания тепловой карты")
                return
            
            # Нормализация данных с обработкой NaN
            heatmap_normalized = heatmap_data.copy()
            for col in heatmap_data.columns:
                col_min = heatmap_data[col].min()
                col_max = heatmap_data[col].max()
                
                # Проверяем, что есть разброс в данных
                if col_max != col_min and not pd.isna(col_min) and not pd.isna(col_max):
                    heatmap_normalized[col] = (heatmap_data[col] - col_min) / (col_max - col_min)
                else:
                    # Если нет разброса или есть NaN, заполняем нулями
                    heatmap_normalized[col] = 0.5
            
            # Заменяем оставшиеся NaN на 0
            heatmap_normalized = heatmap_normalized.fillna(0.5)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_normalized.T, 
                       annot=True, 
                       cmap='YlOrRd', 
                       xticklabels=[f'Сегмент {i}' for i in heatmap_data.index],
                       yticklabels=[
                           'Транзакции', 'Общая сумма', 'Средний чек',
                           'Мерчанты', 'Категории', 'Доля покупок',
                           'Консистентность'
                       ],
                       fmt='.2f',
                       cbar_kws={'label': 'Нормализованное значение'})
            
            plt.title('Тепловая карта характеристик сегментов\n(нормализованные значения)')
            plt.tight_layout()
            
            output_path = config.get_output_path('segment_heatmap.png')
            plt.savefig(output_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Тепловая карта сохранена: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Ошибка при создании тепловой карты: {e}")
            # Создаем простую заглушку
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f'Ошибка создания\nтепловой карты:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title('Тепловая карта характеристик сегментов')
            plt.axis('off')
            
            output_path = config.get_output_path('segment_heatmap.png')
            plt.savefig(output_path, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Заглушка тепловой карты сохранена: {output_path}")
    
    def generate_business_recommendations(self) -> Dict[int, Dict[str, Any]]:
        """Генерация бизнес-рекомендаций для каждого сегмента"""
        print("\n💼 ГЕНЕРАЦИЯ БИЗНЕС-РЕКОМЕНДАЦИЙ")
        print("="*50)
        
        recommendations = {}
        
        for cluster_id, profile in self.segment_profiles.items():
            segment_type = profile['segment_type']
            characteristics = profile['characteristics']
            size_pct = profile['percentage']
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id}: {segment_type['name']} ({size_pct:.1f}% клиентов)")
            
            
            if '🌟 Премиум' in segment_type['name']:
                rec = self._generate_premium_recommendations(characteristics)
            elif '⚡ Активные' in segment_type['name']:
                rec = self._generate_active_recommendations(characteristics)
            elif '💎 VIP' in segment_type['name']:
                rec = self._generate_vip_recommendations(characteristics)
            elif '😴 Спящие' in segment_type['name']:
                rec = self._generate_sleeping_recommendations(characteristics)
            else:  
                rec = self._generate_regular_recommendations(characteristics)
            
            recommendations[cluster_id] = {
                'segment_info': profile,
                'strategies': rec['strategies'],
                'products': rec['products'],
                'channels': rec['channels'],
                'kpis': rec['kpis'],
                'priority': segment_type['priority'],
                'expected_impact': rec['expected_impact']
            }
            
            
            print(f"   📈 СТРАТЕГИИ:")
            for strategy in rec['strategies']:
                print(f"     • {strategy}")
            
            print(f"   🎁 ПРОДУКТЫ:")
            for product in rec['products']:
                print(f"     • {product}")
            
            print(f"   📱 КАНАЛЫ:")
            for channel in rec['channels']:
                print(f"     • {channel}")
        
        self.business_recommendations = recommendations
        return recommendations
    
    def _generate_premium_recommendations(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Рекомендации для премиум клиентов"""
        return {
            'strategies': [
                'Программы лояльности с эксклюзивными привилегиями',
                'Персональный менеджер и консьерж-сервис',
                'Приоритетное обслуживание во всех каналах',
                'Эксклюзивные мероприятия и предложения',
                'Повышенные лимиты и специальные условия'
            ],
            'products': [
                'Private Banking услуги',
                'Инвестиционные продукты и портфели',
                'Премиум кредитные карты с высоким кэшбэком',
                'VIP страхование и защита',
                'Консультации по управлению капиталом'
            ],
            'channels': [
                'Персональные встречи с менеджером',
                'Приоритетная линия поддержки',
                'Мобильное приложение с расширенным функционалом',
                'Эксклюзивные digital-сервисы'
            ],
            'kpis': [
                'Увеличение среднего чека на 15-20%',
                'Рост количества продуктов на клиента',
                'Повышение NPS до 80+',
                'Снижение оттока до 2-3%'
            ],
            'expected_impact': 'Высокий - основной источник прибыли банка'
        }
    
    def _generate_active_recommendations(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Рекомендации для активных клиентов"""
        return {
            'strategies': [
                'Программы лояльности с бонусами за активность',
                'Геймификация банковских услуг',
                'Кэшбэк в популярных категориях',
                'Специальные предложения для частых пользователей',
                'Развитие цифровых сервисов'
            ],
            'products': [
                'Кредитные карты с льготными условиями',
                'Накопительные счета с повышенной ставкой',
                'Потребительские кредиты на выгодных условиях',
                'Страхование покупок и путешествий',
                'Инвестиционные продукты для начинающих'
            ],
            'channels': [
                'Мобильное приложение как основной канал',
                'Push-уведомления о специальных предложениях',
                'Чат-бот для быстрых операций',
                'Социальные сети для коммуникации'
            ],
            'kpis': [
                'Увеличение частоты использования на 25%',
                'Рост cross-sell на 30%',
                'Повышение engagement в digital-каналах',
                'NPS 70+'
            ],
            'expected_impact': 'Высокий - потенциал роста в премиум сегмент'
        }
    
    def _generate_vip_recommendations(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Рекомендации для VIP клиентов"""
        return {
            'strategies': [
                'Стимулирование частоты использования карт',
                'Персонализированные предложения',
                'Консьерж-сервисы и lifestyle-услуги',
                'Эксклюзивные события и привилегии',
                'Программы для увеличения активности'
            ],
            'products': [
                'Премиум карты с особыми привилегиями',
                'Инвестиционные и накопительные продукты',
                'Страхование премиум-класса',
                'Услуги private banking',
                'Кредитные продукты под залог активов'
            ],
            'channels': [
                'Персональные консультации',
                'Приоритетная поддержка',
                'Эксклюзивные мероприятия',
                'Персонализированные digital-сервисы'
            ],
            'kpis': [
                'Увеличение частоты транзакций на 40%',
                'Рост активности в 2 раза',
                'Повышение удовлетворенности',
                'Снижение риска оттока'
            ],
            'expected_impact': 'Средний - высокая ценность при правильной активации'
        }
    
    def _generate_sleeping_recommendations(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Рекомендации для спящих клиентов"""
        return {
            'strategies': [
                'Реактивационные кампании',
                'Образовательные программы по финансовой грамотности',
                'Простые и понятные продукты',
                'Стимулирующие акции для первых транзакций',
                'Упрощение процессов использования'
            ],
            'products': [
                'Базовые дебетовые карты без комиссий',
                'Простые накопительные счета',
                'Микрокредиты и рассрочки',
                'Базовое страхование',
                'Образовательные финансовые продукты'
            ],
            'channels': [
                'SMS и email-рассылки',
                'Простое мобильное приложение',
                'Обучающие материалы и вебинары',
                'Контакт-центр для поддержки'
            ],
            'kpis': [
                'Реактивация 15-20% клиентов',
                'Увеличение активности в 3-5 раз',
                'Снижение стоимости обслуживания',
                'Повышение финансовой грамотности'
            ],
            'expected_impact': 'Низкий - долгосрочная перспектива развития'
        }
    
    def _generate_regular_recommendations(self, characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Рекомендации для обычных клиентов"""
        return {
            'strategies': [
                'Стандартные программы лояльности',
                'Cross-sell дополнительных продуктов',
                'Развитие digital-привычек',
                'Постепенное повышение активности',
                'Качественное базовое обслуживание'
            ],
            'products': [
                'Стандартные кредитные карты',
                'Депозиты и накопительные счета',
                'Потребительские кредиты',
                'Базовое страхование',
                'Платежные сервисы'
            ],
            'channels': [
                'Мобильное приложение',
                'Интернет-банк',
                'Отделения банка',
                'Контакт-центр'
            ],
            'kpis': [
                'Увеличение количества продуктов на 1-2',
                'Рост активности на 20%',
                'Повышение NPS до 60',
                'Стабильное удержание клиентов'
            ],
            'expected_impact': 'Средний - стабильная база банка'
        }
    
    def save_detailed_analysis(self):
        """Сохранение детального анализа"""
        print("\n💾 Сохранение результатов анализа...")
        
        
        output_path = config.get_output_path(config.output.segments_file)
        self.cluster_df.to_csv(output_path, index=True, encoding='utf-8')
        print(f"   📊 Результаты сегментации: {output_path}")
        
        
        summary_stats = self.cluster_df.groupby('cluster').agg({
            'total_transactions': ['count', 'mean', 'median', 'std'],
            'total_amount': ['mean', 'median', 'std'],
            'avg_amount': ['mean', 'median', 'std'],
            'unique_merchants': ['mean', 'median'],
            'unique_categories': ['mean', 'median'],
            'purchase_ratio': ['mean', 'median'],
            'spending_consistency': ['mean', 'median']
        }).round(2)
        
        summary_path = config.get_output_path(config.output.summary_file)
        summary_stats.to_csv(summary_path, encoding='utf-8')
        print(f"   📈 Сводная статистика: {summary_path}")
        
        
        profiles_path = config.get_output_path('segment_profiles.json')
        with open(profiles_path, 'w', encoding='utf-8') as f:
            
            profiles_json = {}
            for k, v in self.segment_profiles.items():
                profiles_json[str(k)] = self._convert_for_json(v)
            json.dump(profiles_json, f, ensure_ascii=False, indent=2)
        print(f"   👥 Профили сегментов: {profiles_path}")
        
        
        recommendations_path = config.get_output_path('business_recommendations.json')
        with open(recommendations_path, 'w', encoding='utf-8') as f:
            recommendations_json = {}
            for k, v in self.business_recommendations.items():
                recommendations_json[str(k)] = self._convert_for_json(v)
            json.dump(recommendations_json, f, ensure_ascii=False, indent=2)
        print(f"   💼 Бизнес-рекомендации: {recommendations_path}")
        
        
        print(f"\n🔬 Выполнение расширенной аналитики...")
        
        
        stability_metrics = self.analyze_segment_stability()
        stability_path = config.get_output_path('segment_stability.json')
        with open(stability_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(stability_metrics), f, ensure_ascii=False, indent=2)
        print(f"   📊 Анализ стабильности: {stability_path}")
        
        
        clv_metrics = self.calculate_customer_lifetime_value()
        clv_path = config.get_output_path('customer_lifetime_value.json')
        with open(clv_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(clv_metrics), f, ensure_ascii=False, indent=2)
        print(f"   💰 Customer Lifetime Value: {clv_path}")
        
        
        journey_analysis = self.analyze_customer_journey()
        journey_path = config.get_output_path('customer_journey.json')
        with open(journey_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(journey_analysis), f, ensure_ascii=False, indent=2)
        print(f"   🛤️ Анализ пути клиента: {journey_path}")
        
        
        monitoring_data = self.create_monitoring_dashboard_data()
        print(f"   📊 Данные мониторинга созданы")
        
        
        ab_tests = self.design_ab_test_framework()
        print(f"   🧪 План A/B тестов создан")
        
        
        print(f"\n📋 Создание отчетов и презентационных материалов...")
        
        
        executive_report = self.generate_executive_report()
        print(f"   📋 Исполнительный отчет создан")
        
        
        presentation_data = self.create_presentation_slides()
        print(f"   🎨 Данные для презентации подготовлены")
        
        print("✅ Все результаты сохранены")
    
    def _convert_for_json(self, obj):
        """Конвертация объектов для JSON сериализации"""
        if isinstance(obj, dict):
            return {str(k): self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, 'item'):  
            return obj.item()
        else:
            return obj
    
    def print_executive_summary(self):
        """Печать краткого резюме для руководства"""
        print("\n" + "="*60)
        print("📋 КРАТКОЕ РЕЗЮМЕ ДЛЯ РУКОВОДСТВА")
        print("="*60)
        
        total_clients = len(self.cluster_df)
        n_segments = len(self.segment_profiles)
        
        print(f"🎯 Общие результаты:")
        print(f"   • Проанализировано клиентов: {total_clients:,}")
        print(f"   • Выявлено сегментов: {n_segments}")
        
        print(f"\n📊 Структура клиентской базы:")
        
        
        segments_by_size = sorted(self.segment_profiles.items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
        
        for cluster_id, profile in segments_by_size:
            segment_type = profile['segment_type']
            print(f"   • {segment_type['name']}: {profile['percentage']:.1f}% "
                  f"({profile['size']:,} клиентов) - Приоритет: {segment_type['priority']}")
        
        print(f"\n💡 Ключевые инсайты:")
        
        
        most_valuable = max(segments_by_size, 
                          key=lambda x: x[1]['characteristics']['avg_total_amount'])
        print(f"   • Самый ценный сегмент: {most_valuable[1]['segment_type']['name']} "
              f"(средний оборот: {most_valuable[1]['characteristics']['avg_total_amount']:,.0f} тенге)")
        
        
        most_active = max(segments_by_size, 
                         key=lambda x: x[1]['characteristics']['avg_transactions'])
        print(f"   • Самый активный сегмент: {most_active[1]['segment_type']['name']} "
              f"(среднее количество транзакций: {most_active[1]['characteristics']['avg_transactions']:.1f})")
        
        
        growth_potential = [s for s in segments_by_size 
                          if 'Спящие' in s[1]['segment_type']['name'] or 
                             'VIP' in s[1]['segment_type']['name']]
        if growth_potential:
            print(f"   • Наибольший потенциал роста: {growth_potential[0][1]['segment_type']['name']} "
                  f"({growth_potential[0][1]['percentage']:.1f}% клиентов)")
        
        print(f"\n🎯 Рекомендации по приоритизации:")
        print(f"   1. Высокий приоритет: Премиум и Активные клиенты (основной фокус)")
        print(f"   2. Средний приоритет: VIP и Обычные клиенты (развитие и удержание)")
        print(f"   3. Низкий приоритет: Спящие клиенты (долгосрочные программы)")
        
        print("="*60)
    
    def analyze_segment_stability(self) -> Dict[int, Dict[str, float]]:
        """Анализ стабильности сегментов во времени"""
        print("\n📊 АНАЛИЗ СТАБИЛЬНОСТИ СЕГМЕНТОВ")
        print("="*50)
        
        stability_metrics = {}
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            
            
            cv_transactions = cluster_data['total_transactions'].std() / cluster_data['total_transactions'].mean()
            cv_amount = cluster_data['avg_amount'].std() / cluster_data['avg_amount'].mean()
            cv_merchants = cluster_data['unique_merchants'].std() / cluster_data['unique_merchants'].mean()
            
            
            stability_index = (cv_transactions + cv_amount + cv_merchants) / 3
            
            
            low_activity_threshold = self.cluster_df['total_transactions'].quantile(0.25)
            churn_risk = len(cluster_data[cluster_data['total_transactions'] < low_activity_threshold]) / len(cluster_data)
            
            stability_metrics[cluster_id] = {
                'stability_index': stability_index,
                'cv_transactions': cv_transactions,
                'cv_amount': cv_amount,
                'cv_merchants': cv_merchants,
                'churn_risk': churn_risk,
                'consistency_score': 1 - stability_index  
            }
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id}:")
            print(f"   Индекс стабильности: {stability_index:.3f} ({'Стабильный' if stability_index < 0.5 else 'Нестабильный'})")
            print(f"   Риск оттока: {churn_risk:.1%}")
            print(f"   Оценка консистентности: {1 - stability_index:.3f}")
        
        return stability_metrics
    
    def calculate_customer_lifetime_value(self) -> Dict[int, Dict[str, float]]:
        """Расчет Customer Lifetime Value для каждого сегмента"""
        print("\n💰 РАСЧЕТ CUSTOMER LIFETIME VALUE")
        print("="*50)
        
        clv_metrics = {}
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            
            
            avg_monthly_revenue = cluster_data['total_amount'].mean() / 12  
            
            
            avg_transaction_frequency = cluster_data['total_transactions'].mean()
            
            
            
            if avg_transaction_frequency > 0:
                estimated_lifetime_months = min(60, max(12, avg_transaction_frequency * 2))  
            else:
                estimated_lifetime_months = 12
            
            
            clv = avg_monthly_revenue * estimated_lifetime_months
            
            
            stability_data = self.analyze_segment_stability()
            churn_risk = stability_data.get(cluster_id, {}).get('churn_risk', 0.5)
            adjusted_clv = clv * (1 - churn_risk)
            
            clv_metrics[cluster_id] = {
                'avg_monthly_revenue': avg_monthly_revenue,
                'estimated_lifetime_months': estimated_lifetime_months,
                'basic_clv': clv,
                'adjusted_clv': adjusted_clv,
                'churn_risk': churn_risk,
                'value_tier': self._classify_clv_tier(adjusted_clv)
            }
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id}:")
            print(f"   Средний месячный доход: {avg_monthly_revenue:,.0f} тенге")
            print(f"   Оценочное время жизни: {estimated_lifetime_months:.1f} месяцев")
            print(f"   CLV (базовый): {clv:,.0f} тенге")
            print(f"   CLV (скорректированный): {adjusted_clv:,.0f} тенге")
            print(f"   Уровень ценности: {self._classify_clv_tier(adjusted_clv)}")
        
        return clv_metrics
    
    def _classify_clv_tier(self, clv: float) -> str:
        """Классификация уровня CLV"""
        if clv >= 500000:  
            return "🌟 Высокая ценность"
        elif clv >= 200000:  
            return "💎 Средняя ценность"
        elif clv >= 50000:   
            return "⭐ Базовая ценность"
        else:
            return "📉 Низкая ценность"
    
    def analyze_customer_journey(self) -> Dict[int, Dict[str, Any]]:
        """Анализ пути клиента и этапов жизненного цикла"""
        print("\n🛤️ АНАЛИЗ ПУТИ КЛИЕНТА")
        print("="*50)
        
        journey_analysis = {}
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            
            
            avg_transactions = cluster_data['total_transactions'].mean()
            avg_merchants = cluster_data['unique_merchants'].mean()
            spending_consistency = cluster_data['spending_consistency'].mean()
            
            
            if avg_transactions < 5 and avg_merchants < 10:
                lifecycle_stage = "🌱 Новички"
                stage_description = "Только начинают пользоваться услугами"
            elif avg_transactions < 20 and spending_consistency < 0.5:
                lifecycle_stage = "🔍 Исследователи"
                stage_description = "Изучают различные возможности"
            elif spending_consistency >= 0.7 and avg_merchants >= 20:
                lifecycle_stage = "💪 Зрелые клиенты"
                stage_description = "Стабильное использование услуг"
            elif avg_transactions >= 50:
                lifecycle_stage = "👑 Чемпионы"
                stage_description = "Максимальное использование услуг"
            else:
                lifecycle_stage = "⚖️ Стабильные"
                stage_description = "Регулярное использование услуг"
            
            
            preferred_time = cluster_data['preferred_hour'].mode().iloc[0] if not cluster_data['preferred_hour'].mode().empty else 12
            time_pattern = self._classify_time_pattern(preferred_time)
            
            
            tech_diversity = cluster_data[['preferred_pos_mode', 'preferred_wallet']].nunique().sum()
            tech_maturity = "Высокая" if tech_diversity > 3 else "Средняя" if tech_diversity > 1 else "Низкая"
            
            journey_analysis[cluster_id] = {
                'lifecycle_stage': lifecycle_stage,
                'stage_description': stage_description,
                'time_pattern': time_pattern,
                'tech_maturity': tech_maturity,
                'engagement_level': self._calculate_engagement_level(cluster_data),
                'next_best_action': self._suggest_next_action(lifecycle_stage, cluster_data)
            }
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id}:")
            print(f"   Этап жизненного цикла: {lifecycle_stage}")
            print(f"   Описание: {stage_description}")
            print(f"   Временной паттерн: {time_pattern}")
            print(f"   Технологическая зрелость: {tech_maturity}")
        
        return journey_analysis
    
    def _classify_time_pattern(self, hour: int) -> str:
        """Классификация временного паттерна"""
        if 6 <= hour <= 9:
            return "🌅 Утренние"
        elif 10 <= hour <= 14:
            return "☀️ Дневные"
        elif 15 <= hour <= 18:
            return "🌆 Вечерние"
        elif 19 <= hour <= 22:
            return "🌙 Ночные"
        else:
            return "🦉 Поздние"
    
    def _calculate_engagement_level(self, cluster_data: pd.DataFrame) -> str:
        """Расчет уровня вовлеченности"""
        
        transactions_score = min(1.0, cluster_data['total_transactions'].mean() / 100)
        merchants_score = min(1.0, cluster_data['unique_merchants'].mean() / 50)
        consistency_score = cluster_data['spending_consistency'].mean()
        
        engagement_score = (transactions_score + merchants_score + consistency_score) / 3
        
        if engagement_score >= 0.8:
            return "🔥 Очень высокий"
        elif engagement_score >= 0.6:
            return "⚡ Высокий"
        elif engagement_score >= 0.4:
            return "📈 Средний"
        else:
            return "📉 Низкий"
    
    def _suggest_next_action(self, lifecycle_stage: str, cluster_data: pd.DataFrame) -> List[str]:
        """Предложение следующих действий"""
        actions = []
        
        if "Новички" in lifecycle_stage:
            actions = [
                "Онбординг программа",
                "Обучающие материалы",
                "Приветственные бонусы",
                "Простые продукты"
            ]
        elif "Исследователи" in lifecycle_stage:
            actions = [
                "Персонализированные рекомендации",
                "Демо новых функций",
                "Кэшбэк программы",
                "Расширение лимитов"
            ]
        elif "Зрелые" in lifecycle_stage:
            actions = [
                "Программы лояльности",
                "Премиум продукты",
                "Инвестиционные предложения",
                "Персональный менеджер"
            ]
        elif "Чемпионы" in lifecycle_stage:
            actions = [
                "VIP статус",
                "Эксклюзивные предложения",
                "Реферальные программы",
                "Консьерж сервисы"
            ]
        else:
            actions = [
                "Поддержание активности",
                "Кросс-продажи",
                "Сезонные акции",
                "Улучшение сервиса"
            ]
        
        return actions
    
    def create_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Создание данных для мониторинга сегментов"""
        print("\n📊 ПОДГОТОВКА ДАННЫХ ДЛЯ МОНИТОРИНГА")
        print("="*50)
        
        monitoring_data = {
            'timestamp': pd.Timestamp.now(),
            'segments_overview': {},
            'alerts': [],
            'kpi_trends': {},
            'recommendations': {}
        }
        
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            
            segment_kpis = {
                'size': len(cluster_data),
                'avg_transactions': cluster_data['total_transactions'].mean(),
                'avg_revenue': cluster_data['total_amount'].mean(),
                'avg_frequency': cluster_data['total_transactions'].mean() / 12,  
                'retention_proxy': cluster_data['spending_consistency'].mean(),
                'growth_potential': self._calculate_growth_potential(cluster_data)
            }
            
            monitoring_data['segments_overview'][cluster_id] = segment_kpis
            
            
            alerts = self._generate_segment_alerts(cluster_id, segment_kpis, cluster_data)
            monitoring_data['alerts'].extend(alerts)
        
        
        monitoring_path = config.get_output_path('monitoring_data.json')
        with open(monitoring_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(monitoring_data), f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📊 Данные мониторинга сохранены: {monitoring_path}")
        print(f"🚨 Обнаружено алертов: {len(monitoring_data['alerts'])}")
        
        return monitoring_data
    
    def _calculate_growth_potential(self, cluster_data: pd.DataFrame) -> str:
        """Расчет потенциала роста сегмента"""
        avg_transactions = cluster_data['total_transactions'].mean()
        avg_merchants = cluster_data['unique_merchants'].mean()
        spending_consistency = cluster_data['spending_consistency'].mean()
        
        
        transaction_score = min(1.0, avg_transactions / 100)
        merchant_score = min(1.0, avg_merchants / 50)
        consistency_score = spending_consistency
        
        growth_score = (transaction_score + merchant_score + consistency_score) / 3
        
        if growth_score >= 0.8:
            return "Низкий (уже развитый)"
        elif growth_score >= 0.6:
            return "Средний"
        elif growth_score >= 0.4:
            return "Высокий"
        else:
            return "Очень высокий"
    
    def _generate_segment_alerts(self, cluster_id: int, kpis: Dict[str, float], cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Генерация алертов для сегмента"""
        alerts = []
        
        
        if kpis['avg_transactions'] < 5:
            alerts.append({
                'segment_id': cluster_id,
                'type': 'LOW_ACTIVITY',
                'severity': 'HIGH',
                'message': f'Сегмент {cluster_id}: Очень низкая активность ({kpis["avg_transactions"]:.1f} транзакций)',
                'recommendation': 'Запустить реактивационную кампанию'
            })
        
        
        if kpis['retention_proxy'] < 0.3:
            alerts.append({
                'segment_id': cluster_id,
                'type': 'CHURN_RISK',
                'severity': 'HIGH',
                'message': f'Сегмент {cluster_id}: Высокий риск оттока (консистентность {kpis["retention_proxy"]:.2f})',
                'recommendation': 'Персонализированные предложения для удержания'
            })
        
        
        if kpis['size'] > len(self.cluster_df) * 0.3 and kpis['avg_revenue'] < self.cluster_df['total_amount'].median():
            alerts.append({
                'segment_id': cluster_id,
                'type': 'LARGE_LOW_VALUE',
                'severity': 'MEDIUM',
                'message': f'Сегмент {cluster_id}: Большой размер ({kpis["size"]} клиентов) но низкий доход',
                'recommendation': 'Стратегия увеличения среднего чека'
            })
        
        
        if kpis['growth_potential'] == "Очень высокий":
            alerts.append({
                'segment_id': cluster_id,
                'type': 'GROWTH_OPPORTUNITY',
                'severity': 'LOW',
                'message': f'Сегмент {cluster_id}: Высокий потенциал роста',
                'recommendation': 'Инвестировать в развитие сегмента'
            })
        
        return alerts
    
    def design_ab_test_framework(self) -> Dict[str, Any]:
        """Дизайн A/B тестов для сегментов"""
        print("\n🧪 ДИЗАЙН A/B ТЕСТОВ")
        print("="*50)
        
        ab_tests = {}
        
        for cluster_id in sorted(self.cluster_df['cluster'].unique()):
            cluster_data = self.cluster_df[self.cluster_df['cluster'] == cluster_id]
            segment_type = self.segment_profiles[cluster_id]['segment_type']['name']
            
            
            if '😴 Спящие' in segment_type:
                test_scenarios = self._design_reactivation_tests(cluster_data)
            elif '⚡ Активные' in segment_type:
                test_scenarios = self._design_engagement_tests(cluster_data)
            elif '🌟 Премиум' in segment_type or '💎 VIP' in segment_type:
                test_scenarios = self._design_premium_tests(cluster_data)
            else:
                test_scenarios = self._design_general_tests(cluster_data)
            
            ab_tests[cluster_id] = {
                'segment_type': segment_type,
                'segment_size': len(cluster_data),
                'test_scenarios': test_scenarios,
                'sample_size_recommendation': self._calculate_sample_size(len(cluster_data)),
                'test_duration_weeks': self._recommend_test_duration(cluster_data),
                'success_metrics': self._define_success_metrics(segment_type)
            }
            
            print(f"\n🎯 СЕГМЕНТ {cluster_id} ({segment_type}):")
            print(f"   Размер выборки: {ab_tests[cluster_id]['sample_size_recommendation']}")
            print(f"   Длительность теста: {ab_tests[cluster_id]['test_duration_weeks']} недель")
            print(f"   Количество сценариев: {len(test_scenarios)}")
        
        
        ab_test_path = config.get_output_path('ab_test_plan.json')
        with open(ab_test_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(ab_tests), f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 План A/B тестов сохранен: {ab_test_path}")
        
        return ab_tests
    
    def _design_reactivation_tests(self, cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Дизайн тестов для реактивации спящих клиентов"""
        return [
            {
                'test_name': 'Персональные предложения vs Общие акции',
                'hypothesis': 'Персонализированные предложения увеличат активность на 25%',
                'control_group': 'Стандартные email-рассылки',
                'test_group': 'Персонализированные предложения на основе истории',
                'primary_metric': 'Количество транзакций в месяц',
                'secondary_metrics': ['Сумма транзакций', 'Количество активных дней']
            },
            {
                'test_name': 'Образовательный контент vs Скидки',
                'hypothesis': 'Образовательный подход повысит долгосрочную активность',
                'control_group': 'Скидки и промокоды',
                'test_group': 'Обучающие материалы + небольшие бонусы',
                'primary_metric': 'Retention rate через 3 месяца',
                'secondary_metrics': ['Engagement с контентом', 'NPS']
            }
        ]
    
    def _design_engagement_tests(self, cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Дизайн тестов для повышения вовлеченности активных клиентов"""
        return [
            {
                'test_name': 'Геймификация vs Кэшбэк',
                'hypothesis': 'Игровые элементы увеличат частоту использования на 15%',
                'control_group': 'Стандартная кэшбэк программа',
                'test_group': 'Система достижений и уровней',
                'primary_metric': 'Частота транзакций',
                'secondary_metrics': ['Разнообразие мерчантов', 'Время в приложении']
            },
            {
                'test_name': 'Социальные функции vs Индивидуальные награды',
                'hypothesis': 'Социальные элементы повысят лояльность',
                'control_group': 'Индивидуальные бонусы',
                'test_group': 'Реферальная программа + социальные челленджи',
                'primary_metric': 'Customer Lifetime Value',
                'secondary_metrics': ['Количество рефералов', 'Социальная активность']
            }
        ]
    
    def _design_premium_tests(self, cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Дизайн тестов для премиум клиентов"""
        return [
            {
                'test_name': 'Консьерж-сервис vs Автоматизированная поддержка',
                'hypothesis': 'Персональный сервис увеличит удовлетворенность на 20%',
                'control_group': 'Стандартная поддержка',
                'test_group': 'Персональный менеджер',
                'primary_metric': 'Net Promoter Score (NPS)',
                'secondary_metrics': ['Customer Satisfaction', 'Время решения вопросов']
            },
            {
                'test_name': 'Эксклюзивные продукты vs Улучшенные условия',
                'hypothesis': 'Эксклюзивность повысит лояльность больше чем льготы',
                'control_group': 'Улучшенные тарифы',
                'test_group': 'Доступ к эксклюзивным продуктам',
                'primary_metric': 'Retention rate',
                'secondary_metrics': ['Cross-sell success', 'Wallet share']
            }
        ]
    
    def _design_general_tests(self, cluster_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Дизайн общих тестов"""
        return [
            {
                'test_name': 'Мобильные уведомления vs Email',
                'hypothesis': 'Push-уведомления эффективнее email для вовлечения',
                'control_group': 'Email-рассылки',
                'test_group': 'Push-уведомления',
                'primary_metric': 'Click-through rate',
                'secondary_metrics': ['Conversion rate', 'App engagement']
            },
            {
                'test_name': 'Временные акции vs Постоянные программы',
                'hypothesis': 'Ограниченные по времени акции создают больше активности',
                'control_group': 'Постоянная программа лояльности',
                'test_group': 'Еженедельные временные акции',
                'primary_metric': 'Количество транзакций',
                'secondary_metrics': ['Средний чек', 'Частота использования']
            }
        ]
    
    def _calculate_sample_size(self, segment_size: int) -> Dict[str, int]:
        """Расчет размера выборки для A/B теста"""
        
        min_sample = max(100, int(segment_size * 0.1))  
        recommended_sample = max(500, int(segment_size * 0.2))  
        max_sample = min(segment_size // 2, int(segment_size * 0.5))  
        
        return {
            'minimum': min_sample,
            'recommended': recommended_sample,
            'maximum': max_sample
        }
    
    def _recommend_test_duration(self, cluster_data: pd.DataFrame) -> int:
        """Рекомендация длительности теста в неделях"""
        avg_frequency = cluster_data['total_transactions'].mean() / 12  
        
        if avg_frequency >= 10:  
            return 2  
        elif avg_frequency >= 5:  
            return 4  
        elif avg_frequency >= 2:  
            return 6  
        else:  
            return 8  
    
    def _define_success_metrics(self, segment_type: str) -> List[str]:
        """Определение метрик успеха для типа сегмента"""
        base_metrics = ['Конверсия', 'Активность', 'Удовлетворенность']
        
        if '😴 Спящие' in segment_type:
            return base_metrics + ['Реактивация', 'Время до первой транзакции']
        elif '⚡ Активные' in segment_type:
            return base_metrics + ['Частота использования', 'Разнообразие активности']
        elif '🌟 Премиум' in segment_type or '💎 VIP' in segment_type:
            return base_metrics + ['NPS', 'Wallet share', 'Retention']
        else:
            return base_metrics + ['Cross-sell', 'Средний чек'] 
    
    def generate_executive_report(self) -> str:
        """Генерация исполнительного отчета"""
        print("\n📋 СОЗДАНИЕ ИСПОЛНИТЕЛЬНОГО ОТЧЕТА")
        print("="*50)
        
        report_lines = []
        
        
        report_lines.extend([
            "# ОТЧЕТ ПО СЕГМЕНТАЦИИ БАНКОВСКИХ КЛИЕНТОВ",
            f"**Дата создания:** {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}",
            f"**Общее количество клиентов:** {len(self.cluster_df):,}",
            f"**Количество сегментов:** {len(self.segment_profiles)}",
            "",
            "## 📊 КРАТКОЕ РЕЗЮМЕ",
            ""
        ])
        
        
        segments_by_size = sorted(self.segment_profiles.items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
        
        for cluster_id, profile in segments_by_size:
            segment_type = profile['segment_type']
            characteristics = profile['characteristics']
            
            report_lines.extend([
                f"### {segment_type['name']} (Сегмент {cluster_id})",
                f"- **Размер:** {profile['size']:,} клиентов ({profile['percentage']:.1f}%)",
                f"- **Приоритет:** {segment_type['priority']}",
                f"- **Средние транзакции:** {characteristics['avg_transactions']:.1f}",
                f"- **Средний доход:** {characteristics['avg_total_amount']:,.0f} тенге",
                f"- **Средний чек:** {characteristics['avg_amount']:,.0f} тенге",
                ""
            ])
        
        
        clv_metrics = self.calculate_customer_lifetime_value()
        report_lines.extend([
            "## 💰 АНАЛИЗ ЦЕННОСТИ КЛИЕНТОВ (CLV)",
            ""
        ])
        
        for cluster_id in sorted(clv_metrics.keys()):
            clv_data = clv_metrics[cluster_id]
            segment_name = self.segment_profiles[cluster_id]['segment_type']['name']
            
            report_lines.extend([
                f"### {segment_name}",
                f"- **CLV (скорректированный):** {clv_data['adjusted_clv']:,.0f} тенге",
                f"- **Месячный доход:** {clv_data['avg_monthly_revenue']:,.0f} тенге",
                f"- **Время жизни:** {clv_data['estimated_lifetime_months']:.1f} месяцев",
                f"- **Уровень ценности:** {clv_data['value_tier']}",
                ""
            ])
        
        
        report_lines.extend([
            "## 🎯 КЛЮЧЕВЫЕ РЕКОМЕНДАЦИИ",
            ""
        ])
        
        
        high_priority_segments = [s for s in segments_by_size 
                                if s[1]['segment_type']['priority'] == 'Высокий']
        
        if high_priority_segments:
            report_lines.extend([
                "### Приоритетные действия:",
                ""
            ])
            
            for i, (cluster_id, profile) in enumerate(high_priority_segments[:3], 1):
                segment_name = profile['segment_type']['name']
                if cluster_id in self.business_recommendations:
                    strategies = self.business_recommendations[cluster_id]['strategies'][:2]
                    report_lines.append(f"{i}. **{segment_name}:**")
                    for strategy in strategies:
                        report_lines.append(f"   - {strategy}")
                    report_lines.append("")
        
        
        monitoring_data = self.create_monitoring_dashboard_data()
        high_severity_alerts = [alert for alert in monitoring_data['alerts'] 
                              if alert['severity'] == 'HIGH']
        
        if high_severity_alerts:
            report_lines.extend([
                "## 🚨 КРИТИЧЕСКИЕ АЛЕРТЫ",
                ""
            ])
            
            for alert in high_severity_alerts:
                report_lines.extend([
                    f"- **{alert['message']}**",
                    f"  *Рекомендация:* {alert['recommendation']}",
                    ""
                ])
        
        
        growth_opportunities = []
        for cluster_id, profile in self.segment_profiles.items():
            if cluster_id in clv_metrics:
                clv_data = clv_metrics[cluster_id]
                if "Высокий" in monitoring_data['segments_overview'][cluster_id]['growth_potential']:
                    growth_opportunities.append((cluster_id, profile, clv_data))
        
        if growth_opportunities:
            report_lines.extend([
                "## 📈 ВОЗМОЖНОСТИ РОСТА",
                ""
            ])
            
            for cluster_id, profile, clv_data in growth_opportunities:
                segment_name = profile['segment_type']['name']
                potential_revenue = clv_data['adjusted_clv'] * profile['size']
                
                report_lines.extend([
                    f"### {segment_name}",
                    f"- **Потенциальный доход:** {potential_revenue:,.0f} тенге",
                    f"- **Размер сегмента:** {profile['size']:,} клиентов",
                    f"- **Текущий CLV:** {clv_data['adjusted_clv']:,.0f} тенге",
                    ""
                ])
        
        
        total_clv = sum(clv_metrics[cid]['adjusted_clv'] * self.segment_profiles[cid]['size'] 
                       for cid in clv_metrics.keys())
        
        report_lines.extend([
            "## 📋 ЗАКЛЮЧЕНИЕ",
            "",
            f"- **Общая ценность клиентской базы:** {total_clv:,.0f} тенге",
            f"- **Средний CLV:** {total_clv / len(self.cluster_df):,.0f} тенге на клиента",
            f"- **Количество критических алертов:** {len(high_severity_alerts)}",
            f"- **Сегментов с высоким потенциалом роста:** {len(growth_opportunities)}",
            "",
            "---",
            "*Отчет создан автоматически системой сегментации клиентов*"
        ])
        
        
        report_content = "\n".join(report_lines)
        report_path = config.get_output_path('executive_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 Исполнительный отчет сохранен: {report_path}")
        
        return report_content
    
    def create_presentation_slides(self) -> Dict[str, Any]:
        """Создание данных для презентации"""
        print("\n🎨 ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕЗЕНТАЦИИ")
        print("="*50)
        
        slides_data = {
            'title_slide': {
                'title': 'Сегментация банковских клиентов',
                'subtitle': f'Анализ {len(self.cluster_df):,} клиентов',
                'date': pd.Timestamp.now().strftime('%d.%m.%Y'),
                'segments_count': len(self.segment_profiles)
            },
            'overview_slide': {
                'total_clients': len(self.cluster_df),
                'segments': []
            },
            'clv_slide': {
                'title': 'Customer Lifetime Value по сегментам',
                'data': []
            },
            'recommendations_slide': {
                'title': 'Ключевые рекомендации',
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            },
            'alerts_slide': {
                'title': 'Критические алерты',
                'alerts': []
            }
        }
        
        
        segments_by_size = sorted(self.segment_profiles.items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
        
        for cluster_id, profile in segments_by_size:
            slides_data['overview_slide']['segments'].append({
                'name': profile['segment_type']['name'],
                'size': profile['size'],
                'percentage': profile['percentage'],
                'priority': profile['segment_type']['priority'],
                'avg_transactions': profile['characteristics']['avg_transactions'],
                'avg_revenue': profile['characteristics']['avg_total_amount']
            })
        
        
        clv_metrics = self.calculate_customer_lifetime_value()
        for cluster_id in sorted(clv_metrics.keys()):
            clv_data = clv_metrics[cluster_id]
            segment_name = self.segment_profiles[cluster_id]['segment_type']['name']
            
            slides_data['clv_slide']['data'].append({
                'segment_name': segment_name,
                'clv': clv_data['adjusted_clv'],
                'monthly_revenue': clv_data['avg_monthly_revenue'],
                'lifetime_months': clv_data['estimated_lifetime_months'],
                'value_tier': clv_data['value_tier']
            })
        
        
        for cluster_id, recommendations in self.business_recommendations.items():
            segment_name = self.segment_profiles[cluster_id]['segment_type']['name']
            priority = recommendations['priority']
            
            rec_data = {
                'segment_name': segment_name,
                'strategies': recommendations['strategies'][:3],  # Топ-3
                'expected_impact': recommendations['expected_impact']
            }
            
            if priority == 'Высокий':
                slides_data['recommendations_slide']['high_priority'].append(rec_data)
            elif priority == 'Средний':
                slides_data['recommendations_slide']['medium_priority'].append(rec_data)
            else:
                slides_data['recommendations_slide']['low_priority'].append(rec_data)
        
        
        monitoring_data = self.create_monitoring_dashboard_data()
        for alert in monitoring_data['alerts']:
            if alert['severity'] in ['HIGH', 'MEDIUM']:
                slides_data['alerts_slide']['alerts'].append({
                    'message': alert['message'],
                    'severity': alert['severity'],
                    'recommendation': alert['recommendation']
                })
        
        
        presentation_path = config.get_output_path('presentation_data.json')
        with open(presentation_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(slides_data), f, ensure_ascii=False, indent=2)
        
        print(f"🎨 Данные презентации сохранены: {presentation_path}")
        
        return slides_data