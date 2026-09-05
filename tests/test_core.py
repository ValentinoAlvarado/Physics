"""Validaciones del nucleo fisico-matematico.

Ejecutar:  python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.mathematics.potential import (
    V_effective,
    V_effective_undeformed,
    g_metric,
)
from src.physics.background import DrainingBathtub
from src.physics.deformation import (
    GaussianDeformation,
    NullDeformation,
    PolynomialDeformation,
)

BG = DrainingBathtub(A=1.0, B=0.7, c=1.0)


# ------------------------------------------------------------------ geometria


def test_radios():
    assert BG.r_h == pytest.approx(1.0)
    assert BG.r_e == pytest.approx(np.hypot(1.0, 0.7))
    assert BG.r_e > BG.r_h  # ergosfera fuera del horizonte


def test_omega_h_dimensional():
    # Omega_H = B c^2 / A^2 = B / r_h^2
    assert BG.Omega_H == pytest.approx(BG.B / BG.r_h**2)


def test_kappa_convenciones_difieren_en_c():
    bg = DrainingBathtub(A=2.0, B=0.5, c=3.0)
    assert bg.surface_gravity_visser / bg.surface_gravity == pytest.approx(bg.c)


# ----------------------------------------------------- condiciones de frontera


@pytest.mark.parametrize(
    "defo",
    [
        PolynomialDeformation(r_h=1.0, a=[0.3]),
        PolynomialDeformation(r_h=1.0, a=[0.3, -0.15, 0.05]),
        GaussianDeformation(r_h=1.0, eps=0.4, sigma=2.0),
        GaussianDeformation(r_h=1.0, eps=-0.2, sigma=0.5),
    ],
)
def test_condiciones_horizonte(defo):
    """h(r_h) = 0 y h'(r_h) = 0 de forma exacta."""
    defo.check_boundary_conditions()


@pytest.mark.parametrize(
    "defo",
    [
        PolynomialDeformation(r_h=1.0, a=[0.3, -0.15]),
        GaussianDeformation(r_h=1.0, eps=0.4, sigma=2.0),
    ],
)
def test_horizonte_preservado(defo):
    """g(r_h) = 0 exactamente: el horizonte no se mueve bajo deformacion."""
    assert g_metric(BG.r_h, BG, defo) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    "defo",
    [
        PolynomialDeformation(r_h=1.0, a=[0.3, -0.15]),
        GaussianDeformation(r_h=1.0, eps=0.4, sigma=2.0),
    ],
)
def test_gravedad_superficial_preservada(defo):
    """kappa_h = |dv_r/dr|_{r_h} invariante, por diferencias finitas."""

    def v_r(r):
        return -(BG.A / r) * (1.0 + defo.h(r))

    eps = 1e-6
    num = (v_r(BG.r_h + eps) - v_r(BG.r_h - eps)) / (2 * eps)
    assert abs(num) == pytest.approx(BG.surface_gravity, rel=1e-6)


def test_dh_consistente_con_h():
    """dh/dr analitica vs. diferencias finitas."""
    for defo in (
        PolynomialDeformation(r_h=1.0, a=[0.3, -0.15, 0.05]),
        GaussianDeformation(r_h=1.0, eps=0.4, sigma=2.0),
    ):
        r = np.linspace(1.05, 6.0, 40)
        eps = 1e-6
        num = (defo.h(r + eps) - defo.h(r - eps)) / (2 * eps)
        assert np.allclose(defo.dh(r), num, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------- reduccion del potencial


@pytest.mark.parametrize("m", [0, 1, 2, 3])
@pytest.mark.parametrize("omega", [0.1, 0.5 + 0.2j, 1.3 - 0.4j])
def test_reduccion_potencial_no_deformado(m, omega):
    """V_h con h=0 debe coincidir con la forma cerrada del DBT estandar.

    Esta es la validacion mas fuerte de la derivacion de V_h.
    """
    null = NullDeformation(r_h=BG.r_h)
    r = np.linspace(1.01, 30.0, 500)
    a = V_effective(r, omega, m, BG, null)
    b = V_effective_undeformed(r, omega, m, BG)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("m", [1, 2])
def test_reduccion_para_varios_fondos(m):
    """La reduccion se mantiene con c != 1 y A != 1."""
    for bg in (
        DrainingBathtub(A=2.3, B=1.1, c=0.8),
        DrainingBathtub(A=0.5, B=-0.4, c=1.7),
    ):
        null = NullDeformation(r_h=bg.r_h)
        r = np.linspace(bg.r_h * 1.01, bg.r_h * 30, 400)
        w = 0.6 - 0.1j
        assert np.allclose(
            V_effective(r, w, m, bg, null),
            V_effective_undeformed(r, w, m, bg),
            rtol=1e-12,
            atol=1e-14,
        )


def test_deformacion_modifica_el_potencial():
    """Control negativo: con h != 0 el potencial DEBE diferir del de referencia.

    Si esto pasara, la familia no estaria deformando nada y el estudio de
    identificabilidad seria vacuo.
    """
    defo = GaussianDeformation(r_h=BG.r_h, eps=0.4, sigma=2.0)
    r = np.linspace(1.05, 8.0, 300)
    w, m = 0.5, 2
    assert not np.allclose(
        V_effective(r, w, m, BG, defo),
        V_effective_undeformed(r, w, m, BG),
        rtol=1e-6,
    )
