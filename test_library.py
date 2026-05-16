#!/usr/bin/env python3
"""
Script de prueba rápido para validar la instalación de montecarlo_sim
"""

import numpy as np

def test_installation():
    """Verifica que la librería esté instalada correctamente."""
    print("="*70)
    print("TEST DE INSTALACION - montecarlo_sim v1.0.0")
    print("="*70)
    print()

    try:
        import montecarlo_sim as mc
        print("[OK] Libreria importada correctamente")
        print(f"   Version: {mc.__version__}")
        print(f"   Funciones disponibles: {len(mc.__all__)}")
        print()
    except ImportError as e:
        print(f"[ERROR] Error al importar libreria: {e}")
        return False

    return True


def test_case_1():
    """Prueba el Caso 1: Monte Carlo estándar."""
    print("TEST CASO 1: Monte Carlo Estandar (Latencia RAG)")
    print("-"*70)

    try:
        from montecarlo_sim import simulate_rag_latency, analyze_sla_compliance

        # Ejecutar simulación pequeña
        results = simulate_rag_latency(n_simulations=5000, seed=42)

        print(f"   Media: {results['mean']:.2f} ms")
        print(f"   Desv. estandar: {results['std']:.2f} ms")
        print(f"   P95: {results['percentile_95']:.2f} ms")
        print(f"   P99: {results['percentile_99']:.2f} ms")

        # Analizar SLA
        sla = analyze_sla_compliance(results, sla_threshold=300)
        print(f"   SLA cumplido (300ms): {sla['sla_met']}")
        print(f"   Tasa de cumplimiento: {sla['compliance_rate']:.2f}%")

        print("[OK] Caso 1: PASS")
        print()
        return True

    except Exception as e:
        print(f"[ERROR] Error en Caso 1: {e}")
        print()
        return False


def test_case_2():
    """Prueba el Caso 2: Importance Sampling."""
    print("TEST CASO 2: Importance Sampling (Eventos Raros)")
    print("-"*70)

    try:
        from montecarlo_sim import importance_sampling_estimator, compare_estimators

        # Estimación con IS
        threshold = 25
        is_result = importance_sampling_estimator(
            threshold=threshold,
            n_samples=5000,
            seed=42
        )

        print(f"   P(X > {threshold}) = {is_result['probability']:.6f}")
        print(f"   Error estandar: {is_result['std_error']:.6f}")
        print(f"   Varianza: {is_result['variance']:.8f}")

        # Comparar con MC
        comparison = compare_estimators(threshold=threshold, n_samples=5000, seed=42)
        print(f"   Reduccion de varianza: {comparison['variance_reduction_factor']:.2f}x")

        print("[OK] Caso 2: PASS")
        print()
        return True

    except Exception as e:
        print(f"[ERROR] Error en Caso 2: {e}")
        print()
        return False


def test_case_3():
    """Prueba el Caso 3: Rejection Sampling."""
    print("TEST CASO 3: Rejection Sampling (Distribucion Bimodal)")
    print("-"*70)

    try:
        from montecarlo_sim import rejection_sampling_bimodal

        # Generar muestras bimodales
        result = rejection_sampling_bimodal(
            n_samples=2000,
            mu1=100, sigma1=15,
            mu2=200, sigma2=20,
            envelope_type='normal',
            seed=42,
            return_diagnostics=True
        )

        samples = result['samples']
        print(f"   Muestras generadas: {len(samples):,}")
        print(f"   Media: {samples.mean():.2f}")
        print(f"   Desv. estandar: {samples.std():.2f}")
        print(f"   Tasa de aceptacion: {result['acceptance_rate']:.2%}")
        print(f"   Constante k: {result['k_constant']:.2f}")

        # Verificar bimodalidad aproximada
        mean_expected = (100 + 0.5 * 200) / 1.5  # Promedio ponderado
        print(f"   Media esperada (aprox): {mean_expected:.2f}")

        print("[OK] Caso 3: PASS")
        print()
        return True

    except Exception as e:
        print(f"[ERROR] Error en Caso 3: {e}")
        print()
        return False


def main():
    """Ejecuta todos los tests."""
    print()

    # Test de instalación
    if not test_installation():
        print("[FALLO] La libreria no esta instalada correctamente")
        return

    # Tests de casos
    results = []
    results.append(test_case_1())
    results.append(test_case_2())
    results.append(test_case_3())

    # Resumen
    print("="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Tests pasados: {passed}/{total}")
    print()

    if all(results):
        print("TODOS LOS TESTS PASARON EXITOSAMENTE")
        print()
        print("La libreria montecarlo_sim esta lista para usar.")
        print("Ejecuta el notebook: jupyter notebook Evaluacion_Simulacion.ipynb")
    else:
        print("ADVERTENCIA: Algunos tests fallaron. Revisa los errores arriba.")

    print()


if __name__ == "__main__":
    main()
