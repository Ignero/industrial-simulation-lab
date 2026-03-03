"""
Author: Jesús Valera Echeverria

Description:
Linearization of non-isothermal CSTR around steady state.
Computes Jacobian and eigenvalues.
"""

import numpy as np
from scipy.optimize import root
from models.cstr import cstr_rhs


# Parameters
params = dict(
    V=10.0,
    F=1.0,
    rho=1000.0,
    Cp=4180.0,
    A=1.0e7,
    Ea=80000.0,
    R=8.314,
    delta_H=-50000.0,
)

def C_in(t): return 2.0
def T_in(t): return 350.0
def Q(t): return 0.0


# Steady state
def steady_state_equations(y):
    return cstr_rhs(
        0.0, y,
        **params,
        C_in=C_in,
        T_in=T_in,
        Q=Q
    )


y_guess = [1.0, 360.0]
sol = root(steady_state_equations, y_guess)

C_ss, T_ss = sol.x

print("Steady State:")
print(C_ss, T_ss)


# Numerical Jacobian
def numerical_jacobian(f, y0, eps=1e-6):
    n = len(y0)
    J = np.zeros((n, n))

    for i in range(n):
        y_plus = y0.copy()
        y_minus = y0.copy()
        y_plus[i] += eps
        y_minus[i] -= eps

        f_plus = np.array(f(y_plus))
        f_minus = np.array(f(y_minus))

        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J


A = numerical_jacobian(
    steady_state_equations,
    np.array([C_ss, T_ss])
)

print("\nA matrix:")
print(A)

print("\nEigenvalues:")
print(np.linalg.eigvals(A))