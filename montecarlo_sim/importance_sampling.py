"""
Módulo de Importance Sampling
==============================

Estimación de eventos raros en pipelines bioinformáticos mediante
muestreo por importancia.
"""

import numpy as np
from typing import Callable, Dict, Tuple
from scipy import stats


def target_distribution_pdf(x: np.ndarray, params: Dict) -> np.ndarray:
    """
    Densidad de probabilidad objetivo f(x): mezcla de distribuciones de cola pesada.
    
    Modela tiempos de ejecución de pipeline bioinformático con comportamiento
    de valores extremos.
    
    Parameters
    ----------
    x : np.ndarray
        Valores donde evaluar la densidad.
    params : dict
        Parámetros de la distribución:
        - 'weight1': peso de la primera componente (0 < w < 1)
        - 'mean1': media de la primera componente normal
        - 'std1': desviación estándar de la primera componente
        - 'shape': parámetro de forma de la distribución Weibull
        - 'scale': parámetro de escala de la distribución Weibull
    
    Returns
    -------
    np.ndarray
        Valores de densidad evaluados en x.
    
    Notes
    -----
    Usa una mezcla: w * Normal + (1-w) * Weibull
    La Weibull captura tiempos extremos (eventos catastróficos raros).
    """
    w1 = params.get('weight1', 0.7)
    mean1 = params.get('mean1', 10)
    std1 = params.get('std1', 2)
    shape = params.get('shape', 1.5)  # k de Weibull
    scale = params.get('scale', 15)   # lambda de Weibull
    
    # Componente 1: Normal (tiempos normales)
    normal_component = stats.norm.pdf(x, loc=mean1, scale=std1)
    
    # Componente 2: Weibull (tiempos extremos/catastróficos)
    weibull_component = stats.weibull_min.pdf(x, c=shape, scale=scale)
    
    # Mezcla
    return w1 * normal_component + (1 - w1) * weibull_component


def proposal_distribution_sample(n: int, params: Dict, seed: int = 42) -> np.ndarray:
    """
    Genera muestras de la distribución propuesta g(x).
    
    Usa una exponencial desplazada para concentrar muestras en región crítica.
    
    Parameters
    ----------
    n : int
        Número de muestras a generar.
    params : dict
        Parámetros de la distribución propuesta:
        - 'rate': tasa de la exponencial (lambda)
        - 'shift': desplazamiento para cubrir región crítica
    seed : int, default=42
        Semilla para reproducibilidad.
    
    Returns
    -------
    np.ndarray
        Muestras de la distribución propuesta.
    """
    np.random.seed(seed)
    rate = params.get('rate', 0.1)
    shift = params.get('shift', 15)
    
    # Exponencial desplazada: shift + Exp(rate)
    samples = shift + np.random.exponential(scale=1/rate, size=n)
    return samples


def proposal_distribution_pdf(x: np.ndarray, params: Dict) -> np.ndarray:
    """
    Densidad de probabilidad de la distribución propuesta g(x).
    
    Parameters
    ----------
    x : np.ndarray
        Valores donde evaluar la densidad.
    params : dict
        Parámetros de la distribución propuesta.
    
    Returns
    -------
    np.ndarray
        Valores de densidad evaluados en x.
    """
    rate = params.get('rate', 0.1)
    shift = params.get('shift', 15)
    
    # PDF de exponencial desplazada
    result = np.zeros_like(x, dtype=float)
    mask = x >= shift
    result[mask] = rate * np.exp(-rate * (x[mask] - shift))
    
    return result


def importance_sampling_estimator(
    threshold: float,
    n_samples: int = 100000,
    target_params: Dict = None,
    proposal_params: Dict = None,
    seed: int = 42
) -> Dict:
    """
    Estima P(X > threshold) usando Importance Sampling.
    
    Calcula la probabilidad de que el tiempo de ejecución supere un umbral
    crítico usando muestreo por importancia para reducir varianza.
    
    Parameters
    ----------
    threshold : float
        Umbral crítico de tiempo (límite máximo de horas).
    n_samples : int, default=100000
        Número de muestras para la estimación.
    target_params : dict, optional
        Parámetros de la distribución objetivo f(x).
    proposal_params : dict, optional
        Parámetros de la distribución propuesta g(x).
    seed : int, default=42
        Semilla para reproducibilidad.
    
    Returns
    -------
    dict
        Diccionario con resultados:
        - 'probability': estimación de P(X > threshold)
        - 'variance': varianza del estimador
        - 'std_error': error estándar de la estimación
        - 'samples': muestras generadas de g(x)
        - 'weights': pesos de importancia
        - 'confidence_interval_95': intervalo de confianza al 95%
    
    Examples
    --------
    >>> result = importance_sampling_estimator(threshold=25, n_samples=100000)
    >>> print(f"P(X > 25) = {result['probability']:.6f} ± {result['std_error']:.6f}")
    
    Notes
    -----
    El estimador de Importance Sampling es:
    
    E[I(X > t)] ≈ (1/n) * Σ I(X_i > t) * f(X_i) / g(X_i)
    
    donde X_i ~ g(x) y f(x)/g(x) es el peso de importancia.
    """
    # Parámetros por defecto
    if target_params is None:
        target_params = {
            'weight1': 0.7,
            'mean1': 10,
            'std1': 2,
            'shape': 1.5,
            'scale': 15
        }
    
    if proposal_params is None:
        proposal_params = {
            'rate': 0.1,
            'shift': threshold * 0.8  # Concentrar cerca del umbral
        }
    
    # Generar muestras de g(x)
    samples = proposal_distribution_sample(n_samples, proposal_params, seed)
    
    # Calcular pesos de importancia: w_i = f(X_i) / g(X_i)
    f_x = target_distribution_pdf(samples, target_params)
    g_x = proposal_distribution_pdf(samples, proposal_params)
    
    # Evitar división por cero
    g_x = np.maximum(g_x, 1e-10)
    weights = f_x / g_x
    
    # Indicador: 1 si X_i > threshold, 0 en otro caso
    indicator = (samples > threshold).astype(float)
    
    # Estimador de Importance Sampling
    weighted_indicators = indicator * weights
    probability = np.mean(weighted_indicators)
    
    # Varianza del estimador
    variance = np.var(weighted_indicators) / n_samples
    std_error = np.sqrt(variance)
    
    # Intervalo de confianza al 95%
    z_score = 1.96
    ci_lower = probability - z_score * std_error
    ci_upper = probability + z_score * std_error
    
    return {
        'probability': probability,
        'variance': variance,
        'std_error': std_error,
        'samples': samples,
        'weights': weights,
        'confidence_interval_95': (ci_lower, ci_upper)
    }


def standard_mc_estimator(
    threshold: float,
    n_samples: int = 100000,
    target_params: Dict = None,
    seed: int = 42
) -> Dict:
    """
    Estima P(X > threshold) usando Monte Carlo estándar (para comparación).
    
    Parameters
    ----------
    threshold : float
        Umbral crítico de tiempo.
    n_samples : int, default=100000
        Número de muestras para la estimación.
    target_params : dict, optional
        Parámetros de la distribución objetivo f(x).
    seed : int, default=42
        Semilla para reproducibilidad.
    
    Returns
    -------
    dict
        Diccionario con resultados:
        - 'probability': estimación de P(X > threshold)
        - 'variance': varianza del estimador
        - 'std_error': error estándar de la estimación
        - 'samples': muestras generadas de f(x)
        - 'confidence_interval_95': intervalo de confianza al 95%
    
    Notes
    -----
    Muestrea directamente de f(x) y calcula la fracción de muestras > threshold.
    Menos eficiente que Importance Sampling para eventos raros.
    """
    np.random.seed(seed)
    
    # Parámetros por defecto
    if target_params is None:
        target_params = {
            'weight1': 0.7,
            'mean1': 10,
            'std1': 2,
            'shape': 1.5,
            'scale': 15
        }
    
    # Muestrear directamente de f(x) usando mezcla
    w1 = target_params['weight1']
    n_normal = int(n_samples * w1)
    n_weibull = n_samples - n_normal
    
    # Componente normal
    normal_samples = np.random.normal(
        loc=target_params['mean1'],
        scale=target_params['std1'],
        size=n_normal
    )
    
    # Componente Weibull
    weibull_samples = stats.weibull_min.rvs(
        c=target_params['shape'],
        scale=target_params['scale'],
        size=n_weibull,
        random_state=seed
    )
    
    # Combinar muestras
    samples = np.concatenate([normal_samples, weibull_samples])
    np.random.shuffle(samples)
    
    # Indicador
    indicator = (samples > threshold).astype(float)
    
    # Estimador Monte Carlo estándar
    probability = np.mean(indicator)
    
    # Varianza (Bernoulli)
    variance = probability * (1 - probability) / n_samples
    std_error = np.sqrt(variance)
    
    # Intervalo de confianza al 95%
    z_score = 1.96
    ci_lower = probability - z_score * std_error
    ci_upper = probability + z_score * std_error
    
    return {
        'probability': probability,
        'variance': variance,
        'std_error': std_error,
        'samples': samples,
        'confidence_interval_95': (ci_lower, ci_upper)
    }


def compare_estimators(
    threshold: float,
    n_samples: int = 100000,
    target_params: Dict = None,
    proposal_params: Dict = None,
    seed: int = 42
) -> Dict:
    """
    Compara Importance Sampling vs Monte Carlo estándar.
    
    Parameters
    ----------
    threshold : float
        Umbral crítico para estimación.
    n_samples : int, default=100000
        Número de muestras para ambos métodos.
    target_params : dict, optional
        Parámetros de distribución objetivo.
    proposal_params : dict, optional
        Parámetros de distribución propuesta (solo para IS).
    seed : int, default=42
        Semilla para reproducibilidad.
    
    Returns
    -------
    dict
        Comparación con:
        - 'importance_sampling': resultados de IS
        - 'standard_mc': resultados de MC estándar
        - 'variance_reduction_factor': factor de reducción de varianza
        - 'efficiency_gain': ganancia en eficiencia computacional
    
    Notes
    -----
    La eficiencia relativa se calcula como:
    
    Eff = Var(MC) / Var(IS)
    
    Un valor > 1 indica que IS es más eficiente.
    """
    # Ejecutar ambos métodos
    is_result = importance_sampling_estimator(
        threshold, n_samples, target_params, proposal_params, seed
    )
    
    mc_result = standard_mc_estimator(
        threshold, n_samples, target_params, seed
    )
    
    # Calcular reducción de varianza
    variance_reduction = mc_result['variance'] / is_result['variance']
    
    # Ganancia de eficiencia (cuántas muestras MC necesitarías para igualar IS)
    efficiency_gain = variance_reduction
    
    return {
        'importance_sampling': is_result,
        'standard_mc': mc_result,
        'variance_reduction_factor': variance_reduction,
        'efficiency_gain': efficiency_gain
    }
