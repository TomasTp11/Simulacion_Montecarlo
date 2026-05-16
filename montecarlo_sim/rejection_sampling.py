"""
Módulo de Rejection Sampling
=============================

Generación de muestras de distribuciones complejas mediante
el algoritmo de muestreo por rechazo.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Union
from scipy import stats


def bimodal_target_pdf(
    x: np.ndarray,
    mu1: float = 100,
    sigma1: float = 15,
    mu2: float = 200,
    sigma2: float = 20
) -> np.ndarray:
    """
    Densidad de probabilidad objetivo bimodal no normalizada.
    
    Modela tráfico de peticiones a una API con dos picos:
    - Pico 1: tráfico en horas de oficina
    - Pico 2: sincronizaciones nocturnas de bases de datos
    
    p*(x) ∝ exp(-(x-μ₁)²/(2σ₁²)) + 0.5·exp(-(x-μ₂)²/(2σ₂²))
    
    Parameters
    ----------
    x : np.ndarray
        Valores donde evaluar la densidad.
    mu1 : float, default=100
        Media del primer pico (tráfico diurno).
    sigma1 : float, default=15
        Desviación estándar del primer pico.
    mu2 : float, default=200
        Media del segundo pico (tráfico nocturno).
    sigma2 : float, default=20
        Desviación estándar del segundo pico.
    
    Returns
    -------
    np.ndarray
        Valores de densidad no normalizada evaluados en x.
    
    Notes
    -----
    La función no está normalizada (no integra a 1), pero el algoritmo
    de Rejection Sampling funciona con densidades proporcionales.
    """
    # Primer pico (más alto): tráfico de oficina
    peak1 = np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    
    # Segundo pico (menor): tráfico nocturno (peso 0.5)
    peak2 = 0.5 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    
    return peak1 + peak2


def proposal_envelope_pdf(
    x: np.ndarray,
    envelope_type: str = 'normal',
    params: Dict = None
) -> np.ndarray:
    """
    Densidad de probabilidad de la distribución envolvente q(x).
    
    Parameters
    ----------
    x : np.ndarray
        Valores donde evaluar la densidad.
    envelope_type : str, default='normal'
        Tipo de distribución envolvente:
        - 'normal': Normal con gran varianza
        - 'uniform': Uniforme amplia
    params : dict, optional
        Parámetros de la distribución envolvente.
    
    Returns
    -------
    np.ndarray
        Valores de densidad evaluados en x.
    """
    if params is None:
        params = {}
    
    if envelope_type == 'normal':
        mean = params.get('mean', 150)
        std = params.get('std', 80)
        return stats.norm.pdf(x, loc=mean, scale=std)
    
    elif envelope_type == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 300)
        return stats.uniform.pdf(x, loc=low, scale=high - low)
    
    else:
        raise ValueError(f"Tipo de envolvente no soportado: {envelope_type}")


def find_envelope_constant(
    target_pdf: Callable,
    envelope_pdf: Callable,
    x_range: Tuple[float, float] = (0, 300),
    n_points: int = 10000
) -> float:
    """
    Encuentra la constante k tal que k·q(x) ≥ p*(x) en todo el dominio.
    
    Parameters
    ----------
    target_pdf : callable
        Función de densidad objetivo p*(x).
    envelope_pdf : callable
        Función de densidad envolvente q(x).
    x_range : tuple, default=(0, 300)
        Rango del dominio a evaluar.
    n_points : int, default=10000
        Número de puntos para encontrar el máximo.
    
    Returns
    -------
    float
        Constante k con un margen de seguridad del 5%.
    
    Notes
    -----
    Se busca k = max(p*(x) / q(x)) y se añade 5% de margen para garantizar
    que la envolvente cubra completamente la distribución objetivo.
    """
    x_test = np.linspace(x_range[0], x_range[1], n_points)
    
    p_values = target_pdf(x_test)
    q_values = envelope_pdf(x_test)
    
    # Evitar división por cero
    q_values = np.maximum(q_values, 1e-10)
    
    # Encontrar el máximo ratio
    ratio = p_values / q_values
    k = np.max(ratio)
    
    # Añadir margen de seguridad del 5%
    k *= 1.05
    
    return k


def rejection_sampling_bimodal(
    n_samples: int = 10000,
    mu1: float = 100,
    sigma1: float = 15,
    mu2: float = 200,
    sigma2: float = 20,
    envelope_type: str = 'normal',
    envelope_params: Dict = None,
    seed: int = 42,
    return_diagnostics: bool = False
) -> Union[np.ndarray, Dict]:
    """
    Genera muestras de la distribución bimodal usando Rejection Sampling.
    
    Implementa el algoritmo de muestreo por rechazo para generar muestras
    que sigan la distribución objetivo bimodal p*(x).
    
    Parameters
    ----------
    n_samples : int, default=10000
        Número de muestras aceptadas a generar.
    mu1 : float, default=100
        Media del primer pico de la distribución bimodal.
    sigma1 : float, default=15
        Desviación estándar del primer pico.
    mu2 : float, default=200
        Media del segundo pico de la distribución bimodal.
    sigma2 : float, default=20
        Desviación estándar del segundo pico.
    envelope_type : str, default='normal'
        Tipo de distribución envolvente ('normal' o 'uniform').
    envelope_params : dict, optional
        Parámetros de la distribución envolvente.
    seed : int, default=42
        Semilla para reproducibilidad.
    return_diagnostics : bool, default=False
        Si True, retorna diccionario con diagnósticos adicionales.
    
    Returns
    -------
    np.ndarray or dict
        Si return_diagnostics=False: array con muestras aceptadas.
        Si return_diagnostics=True: diccionario con:
        - 'samples': muestras aceptadas
        - 'acceptance_rate': tasa de aceptación empírica
        - 'k_constant': constante de la envolvente usada
        - 'n_proposals': número total de propuestas generadas
        - 'envelope_type': tipo de envolvente usada
    
    Examples
    --------
    >>> samples = rejection_sampling_bimodal(n_samples=10000)
    >>> print(f"Media de muestras: {np.mean(samples):.2f}")
    >>> 
    >>> # Con diagnósticos
    >>> result = rejection_sampling_bimodal(
    ...     n_samples=10000,
    ...     return_diagnostics=True
    ... )
    >>> print(f"Tasa de aceptación: {result['acceptance_rate']:.2%}")
    
    Notes
    -----
    Algoritmo de Rejection Sampling:
    
    1. Generar x ~ q(x)
    2. Generar u ~ Uniform(0, 1)
    3. Si u ≤ p*(x) / (k·q(x)), aceptar x; de lo contrario, rechazar
    4. Repetir hasta obtener n_samples muestras aceptadas
    
    La eficiencia del algoritmo depende críticamente de la elección de
    q(x) y k. Una envolvente más ajustada (menor k) resulta en mayor
    tasa de aceptación y menor desperdicio computacional.
    """
    np.random.seed(seed)
    
    # Configurar parámetros de la envolvente
    if envelope_params is None:
        if envelope_type == 'normal':
            envelope_params = {'mean': 150, 'std': 80}
        elif envelope_type == 'uniform':
            envelope_params = {'low': 0, 'high': 300}
    
    # Funciones de densidad
    def target(x):
        return bimodal_target_pdf(x, mu1, sigma1, mu2, sigma2)
    
    def envelope(x):
        return proposal_envelope_pdf(x, envelope_type, envelope_params)
    
    # Encontrar constante k
    k = find_envelope_constant(target, envelope, x_range=(0, 300))
    
    # Proceso de muestreo por rechazo
    accepted_samples = []
    n_proposals = 0
    
    # Generar muestras en lotes para eficiencia
    batch_size = max(n_samples * 10, 10000)  # Estimado generoso
    
    while len(accepted_samples) < n_samples:
        # Generar propuestas de q(x)
        if envelope_type == 'normal':
            proposals = np.random.normal(
                loc=envelope_params['mean'],
                scale=envelope_params['std'],
                size=batch_size
            )
        elif envelope_type == 'uniform':
            proposals = np.random.uniform(
                low=envelope_params['low'],
                high=envelope_params['high'],
                size=batch_size
            )
        
        # Filtrar propuestas en rango válido
        proposals = proposals[(proposals >= 0) & (proposals <= 300)]
        n_proposals += len(proposals)
        
        # Generar variables uniformes
        u = np.random.uniform(0, 1, size=len(proposals))
        
        # Calcular ratio de aceptación
        p_vals = target(proposals)
        q_vals = envelope(proposals)
        q_vals = np.maximum(q_vals, 1e-10)  # Evitar división por cero
        
        acceptance_ratio = p_vals / (k * q_vals)
        
        # Aceptar o rechazar
        accepted = proposals[u <= acceptance_ratio]
        accepted_samples.extend(accepted)
        
        # Ajustar batch_size si es necesario
        if len(accepted_samples) < n_samples * 0.5:
            batch_size = int(batch_size * 1.5)
    
    # Tomar solo las n_samples necesarias
    accepted_samples = np.array(accepted_samples[:n_samples])
    
    # Calcular tasa de aceptación empírica
    acceptance_rate = len(accepted_samples) / n_proposals
    
    if return_diagnostics:
        return {
            'samples': accepted_samples,
            'acceptance_rate': acceptance_rate,
            'k_constant': k,
            'n_proposals': n_proposals,
            'envelope_type': envelope_type,
            'envelope_params': envelope_params,
            'target_params': {
                'mu1': mu1, 'sigma1': sigma1,
                'mu2': mu2, 'sigma2': sigma2
            }
        }
    else:
        return accepted_samples


def plot_rejection_sampling_diagnostics(
    result: Dict,
    n_bins: int = 50,
    save_path: str = None
) -> None:
    """
    Genera visualizaciones diagnósticas del Rejection Sampling.
    
    Parameters
    ----------
    result : dict
        Diccionario retornado por rejection_sampling_bimodal con
        return_diagnostics=True.
    n_bins : int, default=50
        Número de bins para el histograma.
    save_path : str, optional
        Ruta para guardar la figura. Si None, solo muestra.
    
    Notes
    -----
    Esta función requiere matplotlib. No se importa por defecto
    para evitar dependencias pesadas en la librería.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib es requerido para visualizaciones. "
            "Instálalo con: pip install matplotlib"
        )
    
    samples = result['samples']
    target_params = result['target_params']
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histograma de muestras aceptadas
    counts, bins, _ = ax.hist(
        samples,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color='skyblue',
        edgecolor='black',
        label='Muestras aceptadas'
    )
    
    # Curva teórica objetivo
    x_theory = np.linspace(0, 300, 1000)
    p_theory = bimodal_target_pdf(
        x_theory,
        target_params['mu1'],
        target_params['sigma1'],
        target_params['mu2'],
        target_params['sigma2']
    )
    
    # Normalizar para comparar con histograma
    p_theory_normalized = p_theory / (np.trapezoid(p_theory, x_theory))
    
    ax.plot(
        x_theory,
        p_theory_normalized,
        'r-',
        linewidth=2,
        label='Densidad objetivo teórica'
    )
    
    ax.set_xlabel('Peticiones por minuto', fontsize=12)
    ax.set_ylabel('Densidad de probabilidad', fontsize=12)
    ax.set_title(
        f"Rejection Sampling - Tasa de aceptación: {result['acceptance_rate']:.2%}",
        fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
