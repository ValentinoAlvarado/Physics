"""
Validacion del campo complejo y su visualizacion.

Ejecutar:  python -m pytest tests/test_field.py -v
o directamente:  python tests/test_field.py   (imprime el informe completo)
"""

import numpy as np
import pytest

from src.field.analysis import (convergence_test, find_phase_defects,
                                health_check, phase_gradient_cartesian,
                                symmetry_test)
from src.field.coloring import ColorConfig, DomainColoring
from src.field.scattering import ScatteringField
from src.physics.background import DrainingBathtub

OMEGA = 1.2
M_TEST = 6


def _grid(nr=90, nphi=120, r_lo=1.05, r_hi=12.0):
    r = np.linspace(r_lo, r_hi, nr)
    phi = np.linspace(-np.pi, np.pi, nphi)
    return np.meshgrid(r, phi, indexing="ij")


def _field(B, m_max=M_TEST):
    bg = DrainingBathtub(A=1.0, B=B, c=1.0)
    return ScatteringField(bg, omega=OMEGA, m_max=m_max, r_max=13.0,
                           n_out=700)


# ================================================================= TEST 1
def test_symmetry_B_zero():
    """Con B=0 el campo debe ser simetrico bajo phi -> -phi a precision de
    maquina. Este test es el arbitro de la normalizacion de ondas parciales:
    detecto con el que c_m = i^m era INCORRECTO en nuestra convencion (el
    correcto es c_m = (-1)^m, porque H_{-m} = H_m sin el factor de paridad
    que tienen las Bessel)."""
    R, PHI = _grid()
    res = symmetry_test(_field(0.0), R, PHI, tol=1e-10)
    assert res.asymmetry < 1e-10, f"asimetria espuria con B=0: {res.asymmetry:.2e}"


# ================================================================= TEST 2
def test_symmetry_broken_B_nonzero():
    """Con B != 0 la simetria DEBE romperse: es el arrastre de marco."""
    R, PHI = _grid()
    res = symmetry_test(_field(0.7), R, PHI)
    assert res.asymmetry > 1e-3, (
        f"con B=0.7 no hay ruptura de simetria ({res.asymmetry:.2e}): "
        "el termino (omega - m B/r^2)^2 no esta distinguiendo +m de -m")


# ================================================================= TEST 3
def test_convergence_in_M():
    """La truncacion debe converger al aumentar M."""
    R, PHI = _grid(nr=60, nphi=80)
    sf = _field(0.7, m_max=20)
    res = convergence_test(sf, R, PHI, m_values=(5, 10, 15, 20))
    # monotona no estricta y ultima diferencia nula (es la referencia)
    assert res.rel_diff[-1] < 1e-12
    assert res.rel_diff[0] > res.rel_diff[-2], (
        f"no converge: {list(zip(res.m_values, res.rel_diff))}")


# ================================================================= TEST 4
def test_no_nan_inf_in_color():
    """El coloreado no debe producir NaN/Inf ni saturar por outliers."""
    R, PHI = _grid()
    psi = _field(0.7).evaluate(R, PHI)
    h = health_check(psi)
    assert h.n_inf == 0
    rgb = DomainColoring(ColorConfig()).rgb(psi)
    assert np.isfinite(rgb).all(), "el domain coloring produjo NaN/Inf"
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_color_survives_all_nan_input():
    """Caso degenerado: entrada toda NaN no debe reventar."""
    psi = np.full((10, 10), np.nan, dtype=complex)
    rgb = DomainColoring(ColorConfig()).rgb(psi)
    assert np.isfinite(rgb).all()


# ================================================================= TEST 5
def test_zeros_preserved_in_phase_map():
    """Alrededor de un cero la fase debe barrer 2*pi (carga topologica).

    Se comprueba sobre un campo analitico de control con un cero conocido,
    Psi = z, cuya carga es +1 por construccion.
    """
    y, x = np.mgrid[-1:1:60j, -1:1:60j]
    z = x + 1j * y
    defects = find_phase_defects(z, x, y)
    assert len(defects) >= 1, "no se detecto la dislocacion de Psi = z"
    assert any(d.charge == 1 for d in defects), (
        f"carga topologica incorrecta: {[d.charge for d in defects]}")


def test_zeros_not_treated_as_poles():
    """|Psi| -> 0 debe dar luminancia baja, no saturacion."""
    y, x = np.mgrid[-1:1:80j, -1:1:80j]
    z = x + 1j * y
    rgb = DomainColoring(ColorConfig()).rgb(z)
    c = rgb.shape[0] // 2
    center_lum = rgb[c, c].mean()
    edge_lum = rgb[c, 0].mean()
    assert center_lum < edge_lum, (
        "el cero no aparece oscuro: se esta tratando como singularidad")


# ================================================================= TEST 7
def test_phase_gradient_is_local_wavevector():
    """Para una onda plana Psi = e^{i k.x}, grad(arg Psi) debe dar k exacto.

    Es la cantidad que aparece en omega = v.k + c|k|, la misma relacion de
    dispersion que integra el ray tracer.
    """
    y, x = np.mgrid[-2:2:100j, -2:2:100j]
    kx0, ky0 = 3.0, -1.5
    psi = np.exp(1j * (kx0 * x + ky0 * y))
    pg = phase_gradient_cartesian(psi, x, y)
    inner = (slice(2, -2), slice(2, -2))
    assert np.allclose(pg.kx[inner], kx0, atol=1e-2), np.nanmax(pg.kx[inner])
    assert np.allclose(pg.ky[inner], ky0, atol=1e-2), np.nanmax(pg.ky[inner])


# ================================================================= TEST 6
def test_raytracer_untouched():
    """La visualizacion no debe alterar el ray tracer ni la fisica existente:
    el potencial deformado sigue reduciendo al no deformado."""
    from src.mathematics.potential import (V_effective,
                                           V_effective_undeformed)
    from src.physics.deformation import NullDeformation
    bg = DrainingBathtub(A=1.0, B=0.7, c=1.0)
    null = NullDeformation(r_h=bg.r_h)
    r = np.linspace(1.01, 30.0, 400)
    for m in (0, 1, 2):
        assert np.allclose(V_effective(r, 0.5 - 0.2j, m, bg, null),
                           V_effective_undeformed(r, 0.5 - 0.2j, m, bg),
                           rtol=1e-12)


def test_field_does_not_mutate_background():
    """Evaluar el campo no debe modificar el objeto de fondo."""
    bg = DrainingBathtub(A=1.0, B=0.7, c=1.0)
    before = (bg.A, bg.B, bg.c, bg.r_h, bg.Omega_H)
    sf = ScatteringField(bg, omega=OMEGA, m_max=3, r_max=13.0, n_out=400)
    R, PHI = _grid(nr=30, nphi=40)
    sf.evaluate(R, PHI)
    assert (bg.A, bg.B, bg.c, bg.r_h, bg.Omega_H) == before


if __name__ == "__main__":
    print("=" * 68)
    print("VALIDACION DEL CAMPO COMPLEJO")
    print("=" * 68)
    R, PHI = _grid()

    print("\nTEST 1/2 — simetria phi -> -phi")
    for B in (0.0, 0.7, 1.2):
        res = symmetry_test(_field(B), R, PHI)
        tag = ("simetrico (esperado)" if B == 0 else
               "ruptura por arrastre de marco (esperado)")
        print(f"  B={B:.1f}:  asimetria = {res.asymmetry:.3e}   {tag}")

    print("\nTEST 3 — convergencia en M (referencia = M mayor)")
    sf = _field(0.7, m_max=30)
    conv = convergence_test(sf, *_grid(nr=60, nphi=80),
                            m_values=(5, 10, 15, 20, 30))
    for M, d in zip(conv.m_values, conv.rel_diff):
        print(f"  M={M:3d}:  diff rel = {d:.3e}")
    print(f"  -> M sugerido (tol 1e-3): {conv.suggested_m}")

    print("\nTEST 4 — salud numerica")
    psi = _field(0.7).evaluate(R, PHI)
    h = health_check(psi)
    print(f"  NaN={h.n_nan}  Inf={h.n_inf}  de {h.n_total}")
    print(f"  |Psi| en [{h.min_abs_nonzero:.3e}, {h.max_abs:.3e}]")
    rgb = DomainColoring(ColorConfig()).rgb(psi)
    print(f"  RGB finito: {bool(np.isfinite(rgb).all())}  "
          f"rango [{rgb.min():.3f}, {rgb.max():.3f}]")

    print("\nTEST 5 — dislocaciones de fase")
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    dfs = find_phase_defects(psi, X, Y)
    print(f"  dislocaciones detectadas: {len(dfs)}")
    if dfs:
        cargas = {}
        for d in dfs:
            cargas[d.charge] = cargas.get(d.charge, 0) + 1
        print(f"  cargas topologicas: {cargas}")
        print(f"  (deben ser enteras; suma = {sum(d.charge for d in dfs)})")

    print("\nTEST 7 — gradiente de fase sobre onda plana de control")
    y, x = np.mgrid[-2:2:100j, -2:2:100j]
    pg = phase_gradient_cartesian(np.exp(1j * (3.0 * x - 1.5 * y)), x, y)
    inner = (slice(2, -2), slice(2, -2))
    print(f"  k recuperado = ({np.nanmean(pg.kx[inner]):.4f}, "
          f"{np.nanmean(pg.ky[inner]):.4f})   esperado = (3.0000, -1.5000)")
    print("\n" + "=" * 68)
