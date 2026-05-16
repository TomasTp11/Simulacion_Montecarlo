# Guia de Instalacion y Uso Rapido

## Paso 1: Clonar el repositorio

```bash
git clone https://github.com/TomasTp11/Simulacion_Montecarlo.git
cd Simulacion_Montecarlo
```

## Paso 2: Instalar la libreria

```bash
pip install .
```

## Paso 3: Instalar dependencias para el notebook

```bash
pip install matplotlib jupyter
```

## Paso 4: Verificar que todo funciona

```bash
python test_library.py
```

Debe mostrar 3/3 tests pasados.

## Paso 5: Abrir el notebook

```bash
jupyter notebook Evaluacion_Simulacion.ipynb
```

Se abre el navegador automaticamente. Hacer clic en **Kernel → Restart & Run All** para ejecutar los 3 casos completos.

---

## Uso desde Python

### Caso 1: Monte Carlo Estandar
```python
from montecarlo_sim import simulate_rag_latency, analyze_sla_compliance

results = simulate_rag_latency(n_simulations=100000, seed=42)
print(f"P99: {results['percentile_99']:.2f} ms")

sla = analyze_sla_compliance(results, sla_threshold=300)
print(f"SLA cumplido: {sla['sla_met']}")
```

### Caso 2: Importance Sampling
```python
from montecarlo_sim import importance_sampling_estimator, compare_estimators

result = importance_sampling_estimator(threshold=25, n_samples=100000)
print(f"P(X > 25) = {result['probability']:.6f}")

comparison = compare_estimators(threshold=25, n_samples=100000)
print(f"Reduccion de varianza: {comparison['variance_reduction_factor']:.2f}x")
```

### Caso 3: Rejection Sampling
```python
from montecarlo_sim import rejection_sampling_bimodal

result = rejection_sampling_bimodal(
    n_samples=10000,
    mu1=100, sigma1=15,
    mu2=200, sigma2=20,
    return_diagnostics=True
)
print(f"Tasa de aceptacion: {result['acceptance_rate']:.2%}")
```
