"""
Модуль для кластеризации клиентов с различными алгоритмами
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import config

class ClusteringModels:
    """Класс для различных алгоритмов кластеризации"""
    
    def __init__(self, scaled_features: np.ndarray, feature_names: List[str]):
        """Инициализация с подготовленными данными"""
        self.scaled_features = scaled_features
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.optimal_clusters = {}
        
    def get_algorithm_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Описания алгоритмов кластеризации"""
        return {
            'kmeans': {
                'name': 'K-Means',
                'description': 'Разделяет данные на k кластеров, минимизируя внутрикластерную дисперсию',
                'pros': [
                    'Простота интерпретации и реализации',
                    'Быстрая работа на больших данных',
                    'Четкие центроиды кластеров',
                    'Детерминированные результаты',
                    'Хорошо работает с глобулярными кластерами'
                ],
                'cons': [
                    'Требует предварительного задания количества кластеров',
                    'Чувствителен к выбросам',
                    'Предполагает сферическую форму кластеров',
                    'Чувствителен к масштабу признаков'
                ],
                'best_for': 'Бизнес-сегментация с четкими границами между группами клиентов'
            },
            'dbscan': {
                'name': 'DBSCAN',
                'description': 'Находит кластеры произвольной формы на основе плотности точек',
                'pros': [
                    'Автоматически определяет количество кластеров',
                    'Находит кластеры произвольной формы',
                    'Устойчив к выбросам',
                    'Выделяет аномальные точки'
                ],
                'cons': [
                    'Сложность подбора параметров eps и min_samples',
                    'Плохо работает с кластерами разной плотности',
                    'Чувствителен к размерности данных',
                    'Может создавать слишком много мелких кластеров'
                ],
                'best_for': 'Выявление аномальных клиентов и нестандартных паттернов поведения'
            },
            'gaussian_mixture': {
                'name': 'Gaussian Mixture Model',
                'description': 'Моделирует данные как смесь гауссовых распределений',
                'pros': [
                    'Мягкая кластеризация (вероятности принадлежности)',
                    'Гибкость в форме кластеров',
                    'Статистическая интерпретация',
                    'Может моделировать перекрывающиеся кластеры'
                ],
                'cons': [
                    'Требует предварительного задания количества компонент',
                    'Вычислительно сложнее K-means',
                    'Может переобучаться на малых выборках',
                    'Чувствителен к инициализации'
                ],
                'best_for': 'Когда клиенты могут принадлежать к нескольким сегментам одновременно'
            },
            'hierarchical': {
                'name': 'Agglomerative Clustering',
                'description': 'Иерархическая кластеризация снизу вверх',
                'pros': [
                    'Создает иерархию кластеров',
                    'Не требует предварительного задания количества кластеров',
                    'Детерминированные результаты',
                    'Хорошая визуализация через дендрограмму'
                ],
                'cons': [
                    'Высокая вычислительная сложность O(n³)',
                    'Чувствителен к выбросам',
                    'Сложность выбора критерия связи',
                    'Не подходит для очень больших данных'
                ],
                'best_for': 'Исследовательский анализ для понимания структуры данных'
            }
        }
    
    def find_optimal_clusters_kmeans(self, max_clusters: int = None) -> int:
        """Поиск оптимального количества кластеров для K-means"""
        if max_clusters is None:
            max_clusters = config.model.max_clusters
            
        print(f"\n🔍 Поиск оптимального количества кластеров для K-means (до {max_clusters})...")
        
        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            
            kmeans = KMeans(
                n_clusters=k, 
                random_state=config.model.random_state, 
                n_init=config.model.n_init
            )
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(self.scaled_features, cluster_labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.scaled_features, cluster_labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(self.scaled_features, cluster_labels))
        
        
        self._plot_optimization_metrics(K_range, metrics, 'K-means')
        
        
        
        optimal_k = K_range[np.argmax(metrics['silhouette'])]
        
        
        elbow_k = self._find_elbow_point(metrics['inertia'])
        
        print(f"📊 Результаты оптимизации:")
        print(f"   Силуэтный анализ рекомендует: {optimal_k} кластеров")
        print(f"   Метод локтя рекомендует: {elbow_k} кластеров")
        print(f"   Максимальный индекс Калински-Харабаша: {K_range[np.argmax(metrics['calinski_harabasz'])]} кластеров")
        print(f"   Минимальный индекс Дэвиса-Болдина: {K_range[np.argmin(metrics['davies_bouldin'])]} кластеров")
        
        self.optimal_clusters['kmeans'] = optimal_k
        return optimal_k
    
    def _find_elbow_point(self, inertias: List[float]) -> int:
        """Поиск точки локтя методом максимальной кривизны"""
        
        x = np.arange(len(inertias))
        y = np.array(inertias)
        
        
        if len(y) > 2:
            second_derivative = np.diff(y, 2)
            elbow_idx = np.argmax(second_derivative) + 2  
            return elbow_idx + 2  
        else:
            return 3  
    
    def _plot_optimization_metrics(self, K_range: range, metrics: Dict, algorithm: str):
        """Визуализация метрик оптимизации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Оптимизация количества кластеров - {algorithm}', fontsize=16)
        
        
        axes[0, 0].plot(K_range, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Метод локтя (Inertia)')
        axes[0, 0].set_xlabel('Количество кластеров')
        axes[0, 0].set_ylabel('Инерция')
        axes[0, 0].grid(True, alpha=0.3)
        
        
        axes[0, 1].plot(K_range, metrics['silhouette'], 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Силуэтный анализ')
        axes[0, 1].set_xlabel('Количество кластеров')
        axes[0, 1].set_ylabel('Силуэтный коэффициент')
        axes[0, 1].grid(True, alpha=0.3)
        
        
        axes[1, 0].plot(K_range, metrics['calinski_harabasz'], 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Индекс Калински-Харабаша')
        axes[1, 0].set_xlabel('Количество кластеров')
        axes[1, 0].set_ylabel('Индекс CH')
        axes[1, 0].grid(True, alpha=0.3)
        
        
        axes[1, 1].plot(K_range, metrics['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Индекс Дэвиса-Болдина')
        axes[1, 1].set_xlabel('Количество кластеров')
        axes[1, 1].set_ylabel('Индекс DB (меньше = лучше)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config.get_output_path(f'{algorithm.lower()}_optimization.png'), 
                   dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ График оптимизации {algorithm} сохранен")
    
    def fit_kmeans(self, n_clusters: int = None) -> Dict[str, Any]:
        """Обучение K-means модели"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters_kmeans()
        
        print(f"\n🎯 Обучение K-means с {n_clusters} кластерами...")
        
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=config.model.random_state,
            n_init=config.model.n_init
        )
        cluster_labels = kmeans.fit_predict(self.scaled_features)
        
        
        metrics = self._calculate_clustering_metrics(cluster_labels)
        
        
        self.models['kmeans'] = kmeans
        self.results['kmeans'] = {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'metrics': metrics,
            'centroids': kmeans.cluster_centers_
        }
        
        print(f"✅ K-means обучен успешно")
        self._print_metrics(metrics, 'K-means')
        
        return self.results['kmeans']
    
    def fit_dbscan(self, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
        """Обучение DBSCAN модели"""
        if eps is None:
            eps = self._estimate_eps()
        if min_samples is None:
            min_samples = max(2, int(np.log(len(self.scaled_features))))
        
        print(f"\n🎯 Обучение DBSCAN (eps={eps:.3f}, min_samples={min_samples})...")
        
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.scaled_features)
        
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"   Найдено кластеров: {n_clusters}")
        print(f"   Выбросов (шум): {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        if n_clusters < 2:
            print("⚠️ DBSCAN нашел менее 2 кластеров. Попробуйте изменить параметры.")
            return None
        
        
        if n_noise < len(cluster_labels):
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 0 and len(set(cluster_labels[valid_mask])) > 1:
                metrics = self._calculate_clustering_metrics(
                    cluster_labels[valid_mask], 
                    self.scaled_features[valid_mask]
                )
            else:
                metrics = {'silhouette': 0, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
        else:
            metrics = {'silhouette': 0, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
        
        
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'metrics': metrics,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print(f"✅ DBSCAN обучен успешно")
        self._print_metrics(metrics, 'DBSCAN')
        
        return self.results['dbscan']
    
    def _estimate_eps(self) -> float:
        """Оценка параметра eps для DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        
        
        k = max(2, int(np.log(len(self.scaled_features))))
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.scaled_features)
        distances, indices = neighbors_fit.kneighbors(self.scaled_features)
        
        
        distances = np.sort(distances[:, k-1], axis=0)
        
        
        
        eps = np.percentile(distances, 90)
        
        return eps
    
    def fit_gaussian_mixture(self, n_components: int = None) -> Dict[str, Any]:
        """Обучение Gaussian Mixture Model"""
        if n_components is None:
            n_components = self.optimal_clusters.get('kmeans', 4)
        
        print(f"\n🎯 Обучение Gaussian Mixture Model с {n_components} компонентами...")
        
        
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=config.model.random_state,
            covariance_type='full'
        )
        gmm.fit(self.scaled_features)
        cluster_labels = gmm.predict(self.scaled_features)
        
        
        metrics = self._calculate_clustering_metrics(cluster_labels)
        
        
        metrics['aic'] = gmm.aic(self.scaled_features)
        metrics['bic'] = gmm.bic(self.scaled_features)
        metrics['log_likelihood'] = gmm.score(self.scaled_features)
        
        
        self.models['gaussian_mixture'] = gmm
        self.results['gaussian_mixture'] = {
            'labels': cluster_labels,
            'n_clusters': n_components,
            'metrics': metrics,
            'probabilities': gmm.predict_proba(self.scaled_features),
            'means': gmm.means_,
            'covariances': gmm.covariances_
        }
        
        print(f"✅ Gaussian Mixture Model обучен успешно")
        self._print_metrics(metrics, 'Gaussian Mixture Model')
        
        return self.results['gaussian_mixture']
    
    def _calculate_clustering_metrics(self, labels: np.ndarray, features: np.ndarray = None) -> Dict[str, float]:
        """Расчет метрик качества кластеризации"""
        if features is None:
            features = self.scaled_features
        
        metrics = {}
        
        
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            metrics['silhouette'] = silhouette_score(features, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
        else:
            metrics['silhouette'] = 0
            metrics['calinski_harabasz'] = 0
            metrics['davies_bouldin'] = float('inf')
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, float], model_name: str):
        """Вывод метрик качества"""
        print(f"📊 Метрики качества {model_name}:")
        print(f"   Силуэтный коэффициент: {metrics['silhouette']:.3f}")
        print(f"   Индекс Калински-Харабаша: {metrics['calinski_harabasz']:.2f}")
        print(f"   Индекс Дэвиса-Болдина: {metrics['davies_bouldin']:.3f}")
        
        if 'aic' in metrics:
            print(f"   AIC: {metrics['aic']:.2f}")
            print(f"   BIC: {metrics['bic']:.2f}")
            print(f"   Log-likelihood: {metrics['log_likelihood']:.2f}")
    
    def compare_models(self) -> pd.DataFrame:
        """Сравнение всех обученных моделей"""
        print("\n📊 СРАВНЕНИЕ МОДЕЛЕЙ КЛАСТЕРИЗАЦИИ")
        print("="*50)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if results is not None:
                row = {
                    'Модель': model_name.replace('_', ' ').title(),
                    'Количество кластеров': results['n_clusters'],
                    'Силуэтный коэффициент': results['metrics']['silhouette'],
                    'Индекс Калински-Харабаша': results['metrics']['calinski_harabasz'],
                    'Индекс Дэвиса-Болдина': results['metrics']['davies_bouldin']
                }
                
                
                if 'n_noise' in results:
                    row['Количество выбросов'] = results['n_noise']
                if 'aic' in results['metrics']:
                    row['AIC'] = results['metrics']['aic']
                    row['BIC'] = results['metrics']['bic']
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            print(comparison_df.round(3).to_string(index=False))
            
            
            best_model_idx = comparison_df['Силуэтный коэффициент'].idxmax()
            best_model = comparison_df.loc[best_model_idx, 'Модель']
            best_silhouette = comparison_df.loc[best_model_idx, 'Силуэтный коэффициент']
            
            print(f"\n🏆 Рекомендуемая модель: {best_model}")
            print(f"   Силуэтный коэффициент: {best_silhouette:.3f}")
            
            
            comparison_df.to_csv(config.get_output_path('model_comparison.csv'), index=False)
            print(f"💾 Сравнение моделей сохранено: {config.get_output_path('model_comparison.csv')}")
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """Получение лучшей модели по силуэтному коэффициенту"""
        if not self.results:
            raise ValueError("Нет обученных моделей для сравнения")
        
        best_model_name = None
        best_score = -1
        
        for model_name, results in self.results.items():
            if results is not None and results['metrics']['silhouette'] > best_score:
                best_score = results['metrics']['silhouette']
                best_model_name = model_name
        
        return best_model_name, self.results[best_model_name]
    
    def explain_model_choice(self, chosen_model: str) -> str:
        """Объяснение выбора модели"""
        descriptions = self.get_algorithm_descriptions()
        
        if chosen_model not in descriptions:
            return f"Модель {chosen_model} не найдена в описаниях"
        
        model_info = descriptions[chosen_model]
        results = self.results.get(chosen_model, {})
        
        explanation = f"""
🎯 ОБОСНОВАНИЕ ВЫБОРА МОДЕЛИ: {model_info['name']}

📝 Описание алгоритма:
{model_info['description']}

✅ Преимущества:
"""
        for pro in model_info['pros']:
            explanation += f"• {pro}\n"
        
        explanation += f"""
⚠️ Недостатки:
"""
        for con in model_info['cons']:
            explanation += f"• {con}\n"
        
        explanation += f"""
🎯 Лучше всего подходит для:
{model_info['best_for']}

📊 Результаты на наших данных:
"""
        if results:
            metrics = results['metrics']
            explanation += f"• Количество кластеров: {results['n_clusters']}\n"
            explanation += f"• Силуэтный коэффициент: {metrics['silhouette']:.3f}\n"
            explanation += f"• Индекс Калински-Харабаша: {metrics['calinski_harabasz']:.2f}\n"
            explanation += f"• Индекс Дэвиса-Болдина: {metrics['davies_bouldin']:.3f}\n"
        
        return explanation 