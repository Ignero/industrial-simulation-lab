"""
Open-loop CSTR simulation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cstr_pi import cstr_rhs


def main():
    os.makedirs("figures", exist_ok=True)

    # Parameters (consistent units: J/mol)
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

    # Inputs
    def C_in(t): return 2.0
    def T_in(t): return 350.0
    def Q(t): return 0.0

    y0 = [0.5, 350.0]
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 400)

    rhs = lambda t, y: cstr_rhs(t, y, **params, C_in=C_in, T_in=T_in, Q=Q)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="BDF")

    t = sol.t
    C = sol.y[0]
    T = sol.y[1]

    plt.figure()
    plt.plot(t, C)
    plt.xlabel("Time (s)")
    plt.ylabel("C (kg/m3)")
    plt.title("CSTR Open-Loop: Concentration")
    plt.grid()
    plt.savefig("figures/open_loop_C.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(t, T)
    plt.xlabel("Time (s)")
    plt.ylabel("T (K)")
    plt.title("CSTR Open-Loop: Temperature")
    plt.grid()
    plt.savefig("figures/open_loop_T.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()