"""
Módulo de Simulación Monte Carlo Estándar
==========================================

Análisis estocástico de latencia en arquitecturas RAG.
"""

import numpy as np
from typing import Dict, Tuple


def simulate_rag_latency(
    n_simulations: int = 100000,
    embedding_params: Tuple[float, float] = (50, 10),
    db_search_params: Tuple[float, float] = (20, 60),
    llm_params: Tuple[float, float] = (100, 30),
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Simula el tiempo de respuesta total de un sistema RAG mediante Monte Carlo.
    
    El tiempo total se compone de tres etapas independientes:
    1. Generación de embeddings (distribución Normal)
    2. Búsqueda en base de datos vectorial (distribución Uniforme)
    3. Inferencia del LLM (distribución Lognormal)
    
    Parameters
    ----------
    n_simulations : int, default=100000
        Número de simulaciones Monte Carlo a realizar.
    embedding_params : tuple of float, default=(50, 10)
        Parámetros (media, desviación estándar) para la distribución Normal
        de generación de embeddings en milisegundos.
    db_search_params : tuple of float, default=(20, 60)
        Parámetros (mínimo, máximo) para la distribución Uniforme
        de búsqueda en DB vectorial en milisegundos.
    llm_params : tuple of float, default=(100, 30)
        Parámetros (media, sigma) para la distribución Lognormal
        de inferencia del LLM en milisegundos.
    seed : int, default=42
        Semilla para reproducibilidad de resultados.
    
    Returns
    -------
    dict
        Diccionario con las siguientes claves:
        - 'embedding_times': array con tiempos de generación de embeddings
        - 'db_search_times': array con tiempos de búsqueda en DB
        - 'llm_inference_times': array con tiempos de inferencia del LLM
        - 'total_latency': array con latencia total de cada simulación
        - 'percentile_95': float con el percentil 95 de latencia total
        - 'percentile_99': float con el percentil 99 de latencia total
        - 'mean': float con la media de latencia total
        - 'std': float con la desviación estándar de latencia total
    
    Examples
    --------
    >>> results = simulate_rag_latency(n_simulations=100000)
    >>> print(f"P95: {results['percentile_95']:.2f} ms")
    >>> print(f"P99: {results['percentile_99']:.2f} ms")
    
    Notes
    -----
    La distribución Lognormal para inferencia LLM se justifica porque la mayoría
    de consultas son rápidas, pero algunas generan colas largas de procesamiento,
    produciendo una distribución con cola pesada a la derecha.
    """
    # Establecer semilla para reproducibilidad
    np.random.seed(seed)
    
    # Etapa 1: Generación de embeddings (Normal)
    embedding_mean, embedding_std = embedding_params
    embedding_times = np.random.normal(
        loc=embedding_mean,
        scale=embedding_std,
        size=n_simulations
    )
    # Asegurar valores positivos
    embedding_times = np.maximum(embedding_times, 0)
    
    # Etapa 2: Búsqueda en DB vectorial (Uniforme)
    db_min, db_max = db_search_params
    db_search_times = np.random.uniform(
        low=db_min,
        high=db_max,
        size=n_simulations
    )
    
    # Etapa 3: Inferencia del LLM (Lognormal)
    llm_mean, llm_sigma = llm_params
    # Para Lognormal, necesitamos calcular mu y sigma de la distribución subyacente
    # Si queremos media=llm_mean, usamos: mu = log(mean) - 0.5*sigma^2
    llm_mu = np.log(llm_mean) - 0.5 * (llm_sigma / llm_mean) ** 2
    llm_sig = llm_sigma / llm_mean
    llm_inference_times = np.random.lognormal(
        mean=llm_mu,
        sigma=llm_sig,
        size=n_simulations
    )
    
    # Tiempo total de respuesta
    total_latency = embedding_times + db_search_times + llm_inference_times
    
    # Cálculo de estadísticas
    percentile_95 = np.percentile(total_latency, 95)
    percentile_99 = np.percentile(total_latency, 99)
    mean_latency = np.mean(total_latency)
    std_latency = np.std(total_latency)
    
    return {
        'embedding_times': embedding_times,
        'db_search_times': db_search_times,
        'llm_inference_times': llm_inference_times,
        'total_latency': total_latency,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'mean': mean_latency,
        'std': std_latency
    }


def analyze_sla_compliance(
    results: Dict[str, np.ndarray],
    sla_threshold: float
) -> Dict[str, float]:
    """
    Analiza el cumplimiento de un SLA basado en resultados de simulación.
    
    Parameters
    ----------
    results : dict
        Diccionario retornado por simulate_rag_latency().
    sla_threshold : float
        Umbral de tiempo máximo del SLA en milisegundos.
    
    Returns
    -------
    dict
        Diccionario con métricas de cumplimiento:
        - 'compliance_rate': porcentaje de peticiones bajo el umbral
        - 'violation_rate': porcentaje de peticiones sobre el umbral
        - 'sla_met': bool indicando si se cumple SLA del 99%
        - 'margin': diferencia entre P99 y el umbral SLA
    
    Examples
    --------
    >>> results = simulate_rag_latency()
    >>> sla_analysis = analyze_sla_compliance(results, sla_threshold=300)
    >>> print(f"Cumplimiento: {sla_analysis['compliance_rate']:.2f}%")
    """
    total_latency = results['total_latency']
    
    # Calcular tasa de cumplimiento
    compliance_count = np.sum(total_latency <= sla_threshold)
    compliance_rate = (compliance_count / len(total_latency)) * 100
    violation_rate = 100 - compliance_rate
    
    # Verificar si se cumple SLA (99% bajo el umbral)
    sla_met = compliance_rate >= 99.0
    
    # Margen de seguridad
    margin = sla_threshold - results['percentile_99']
    
    return {
        'compliance_rate': compliance_rate,
        'violation_rate': violation_rate,
        'sla_met': sla_met,
        'margin': margin
    }
