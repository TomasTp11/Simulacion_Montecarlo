"""
Librería de Simulación Monte Carlo
===================================

Librería para simulaciones estadísticas usando el Método de Monte Carlo
y técnicas de muestreo avanzado.

Módulos disponibles:
- standard_mc: Simulación Monte Carlo estándar
- importance_sampling: Muestreo por importancia
- rejection_sampling: Muestreo por rechazo
"""

__version__ = "1.0.0"
__author__ = "Taller 3 - Simulación Montecarlo"

from .standard_mc import simulate_rag_latency, analyze_sla_compliance
from .importance_sampling import (
    importance_sampling_estimator, 
    standard_mc_estimator,
    compare_estimators
)
from .rejection_sampling import rejection_sampling_bimodal

__all__ = [
    'simulate_rag_latency',
    'analyze_sla_compliance',
    'importance_sampling_estimator',
    'standard_mc_estimator',
    'compare_estimators',
    'rejection_sampling_bimodal'
]
