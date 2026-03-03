"""
Author: Jesús Valera Echeverria

Description:
Non-isothermal CSTR model.
Contains mass and energy balances.
"""

import numpy as np


def cstr_rhs(t, y, V, F, rho, Cp, A, Ea, R, delta_H,
             C_in, T_in, Q):

    C, T = y

    # Allow inputs to be functions or constants
    Cfeed = C_in(t) if callable(C_in) else C_in
    Tfeed = T_in(t) if callable(T_in) else T_in
    Qval = Q(t) if callable(Q) else Q

    # Reaction rate
    k = A * np.exp(-Ea / (R * T))
    r = k * C

    # Mass balance
    dCdt = (F / V) * (Cfeed - C) - r

    # Energy balance
    dTdt = (
        (F / V) * (Tfeed - T)
        + Qval / (rho * Cp * V)
        + (-delta_H / (rho * Cp)) * r
    )

    return [dCdt, dTdt]